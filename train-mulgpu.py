import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import numpy as np
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol
import model.base 
from model.base import Transformer
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse
from tqdm import tqdm
import data 
from data import ProcessData
import torch_geometric 
from torch_geometric.loader import DataLoader as gDataLoader
import utils 
from utils import monotonic_annealer, get_mask, parallel_f, seed_torch, cyclic_annealer

    

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/moses_train.txt')
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--batch', type=int, default=128)

parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_latent', type=int, default=256)
parser.add_argument('--d_ff', type=int, default=1024)
parser.add_argument('--e_heads', type=int, default=1)
parser.add_argument('--d_heads', type=int, default=8)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--kl_type', type=str, default="monotonic")
parser.add_argument('--kl_w_start', type=float, default=0.0005)
parser.add_argument('--kl_w_end', type=float, default=0.005)
parser.add_argument('--kl_cycle', type=int, default=4)
parser.add_argument('--kl_ratio', type=float, default=0.9)
parser.add_argument('--save_name', type=str, default='mulgpu')
arg = parser.parse_args()

print(f'Number of GPUs: {torch.cuda.device_count()}')


print('\nArguments:')
for name, value in arg.__dict__.items() :
    print(f'\t{name}: {value}')

Data = ProcessData(arg.data_path, arg.max_len)
data_list, smi_list, vocab, inv_vocab, gvocab, max_len = (Data.process(),
                                                         Data.smi_list,
                                                         Data.vocab,
                                                         Data.inv_vocab,
                                                         Data.gvocab,
                                                         Data.max_len)

print(f'\nNumber of data: {len(data_list)}')
print(f'\nSMILES Vocab:\n\t {vocab}')
print(f'\nGraph Vocab:\n\t {gvocab}')

train_loader = gDataLoader(data_list, batch_size=arg.batch, shuffle=True)  


def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss
def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def get_valid(smi) : 
    return smi if get_mol(smi) else None 
def get_novel(smi) : 
    return smi if smi not in smi_list else None 

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, model, train_data, optimizer, gpu_id, save_every=10):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
    


    def _run_batch(self, source, targets, targets_mask):
        self.optimizer.zero_grad()
        output, mu, sigma = self.model(source, targets[:, :-1], None, targets_mask)
        loss, _, _ = loss_fn(output, targets[:, 1:], mu, sigma, self.beta)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        for src in tqdm(self.train_data):

            src = src.to(self.gpu_id)
            tgt = src.clone().smi
            tgt_mask = get_mask(tgt[:, :-1], vocab) 
            # source = source.to(self.gpu_id)
            # targets = source.clone()
            # source_mask = (source != vocab['<PAD>']).unsqueeze(-2)
            # targets_mask = get_mask(targets[:, :-1], vocab)
            self._run_batch(src, tgt, tgt_mask)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        if arg.kl_type == 'monotonic' : 
            annealer = monotonic_annealer(arg.n_epochs, 0, arg.kl_w_start, arg.kl_w_end)
        elif arg.kl_type == 'cyclic' :
            annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.n_epochs, arg.kl_cycle, arg.kl_ratio)
        for epoch in range(max_epochs):
            self.beta = annealer[epoch]
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    train_set = data_list  # load your dataset
    model = Transformer(arg.d_model, arg.d_latent, arg.d_ff, arg.e_heads, arg.d_heads, arg.n_layers, arg.dropout, vocab, gvocab)

    optimizer = torch.optim.Adam(model.parameters(), lr = arg.lr, weight_decay=1e-6)
    return train_set, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return gDataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, arg.batch)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

print('#########################################################################')
print('############################## TRAINING #################################')
print('#########################################################################')

if __name__ == "__main__":
   world_size = torch.cuda.device_count()
   mp.spawn(main, args=(world_size, arg.n_epochs, 1,), nprocs=world_size)








