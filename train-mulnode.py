import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data import ProcessData
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader as gDataLoader
from utils import monotonic_annealer, get_mask, parallel_f, seed_torch, cyclic_annealer
from data import ProcessData
from model.base import Transformer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
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
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--kl_type', type=str, default="monotonic")
parser.add_argument('--kl_w_start', type=float, default=0.0005)
parser.add_argument('--kl_w_end', type=float, default=0.005)
parser.add_argument('--kl_cycle', type=int, default=4)
parser.add_argument('--kl_ratio', type=float, default=0.9)

parser.add_argument('--save_name', type=str, default='your_save_name')
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--save_epoch', type=int, default=0)

arg = parser.parse_args()


if not os.path.exists(f'checkpoint/{arg.save_name}') : 
    os.makedirs(f'checkpoint/{arg.save_name}')




Data = ProcessData('data/moses_train.txt', arg.max_len)
data_list, smi_list, vocab, inv_vocab, gvocab, max_len = (Data.process(),
                                                         Data.smi_list,
                                                         Data.vocab,
                                                         Data.inv_vocab,
                                                         Data.gvocab,
                                                         Data.max_len)

config = vars(arg)
config['vocab'] = vocab 
config['gvocab'] = gvocab
config['max_token_len'] = max_len 
torch.save(vars(arg), f'checkpoint/{arg.save_name}/config.pt')

print(f'\nNumber of data: {len(data_list)}')
print(f'\nSMILES Vocab:\n\t {vocab}')
print(f'\nGraph Vocab:\n\t {gvocab}')


def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0

        if arg.resume : 
            if os.path.exists(f'checkpoint/{arg.save_name}/snapshot_{arg.save_epoch}.pt'): 
                self._load_snapshot(f'checkpoint/{arg.save_name}/snapshot_{arg.save_epoch}.pt')
            else : 
                print(f"No snapshot found at checkpoint/{arg.save_name}/snapshot_{arg.save_epoch}.pt")
                

        # self.snapshot_path = snapshot_path
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, targets_mask):
        self.optimizer.zero_grad()
        output, mu, sigma = self.model(source, targets[:, :-1], None, targets_mask)
        loss, _, _ = loss_fn(output, targets[:, 1:], mu, sigma, self.beta)
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        clip_grad_norm_(self.model.parameters(), 0.5)



    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        
        for src in tqdm(self.train_data) :
            src = src.to(self.local_rank)
            tgt = src.clone().smi
            tgt_mask = get_mask(tgt[:, :-1], vocab) 
            self._run_batch(src, tgt, tgt_mask)



    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }

        torch.save(snapshot, f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')
        print(f"Epoch {epoch} | Training snapshot saved at checkpoint/{arg.save_name}/snapshot_{epoch}.pt")

    def train(self, max_epochs: int):
        if arg.kl_type == 'monotonic' : 
            annealer = monotonic_annealer(arg.n_epochs, 0, arg.kl_w_start, arg.kl_w_end)
        elif arg.kl_type == 'cyclic' :
            annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.n_epochs, arg.kl_cycle, arg.kl_ratio)
        for epoch in range(self.epochs_run, max_epochs):
            self.beta = annealer[epoch]
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_set = data_list  
    model = Transformer(arg.d_model, arg.d_latent, arg.d_ff, arg.e_heads, arg.d_heads, arg.n_layers, arg.dropout, vocab, gvocab)
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.lr, weight_decay=1e-6)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return gDataLoader(
        dataset,
        batch_size=arg.batch,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = f"snapshot.pt"):
    ddp_setup()
    seed_torch()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()





main(arg.save_every, arg.n_epochs, arg.batch)
