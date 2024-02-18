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
from data import ProcessData
from model.base import Transformer
import argparse
import multiprocessing
import metrics

def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
def get_mask( target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default="yourmodel")
parser.add_argument('--save_epoch', type=int, default=100)
parser.add_argument('--use_all', type=bool, default=False)
arg = parser.parse_args()


if not os.path.exists(f'checkpoint/{arg.save_name}') : 
    print('Path not exists')
    exit()



Data = ProcessData('data/moses_train.txt', 0)
data_list, smi_list, vocab, inv_vocab, gvocab, max_len = (Data.process(),
                                                         Data.smi_list,
                                                         Data.vocab,
                                                         Data.inv_vocab,
                                                         Data.gvocab,
                                                         Data.max_len)

config = torch.load(f'checkpoint/{arg.save_name}/config.pt')


class GenSet(Dataset) :
    def __init__(self, num_gen, d_latent) : 
        self.num_gen = num_gen
        self.d_latent = d_latent
    def __len__(self) : 
        return self.num_gen
    
    def __getitem__(self, idx) : 
        z = torch.randn(self.d_latent)
        tgt = torch.zeros(1, dtype=torch.long)
        return z, tgt 



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

        self.model = DDP(self.model, device_ids=[self.local_rank])

    # def _load_snapshot(self, snapshot_path):
    #     loc = f"cuda:{self.local_rank}"
    #     snapshot = torch.load(snapshot_path, map_location=loc)
    #     self.model.load_state_dict(snapshot["MODEL_STATE"])
    #     self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    #     self.epochs_run = snapshot["EPOCHS_RUN"]
    #     print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, targets_mask):
        # self.optimizer.zero_grad()
        pred = self.model.module.inference(source, targets, None, targets_mask)
        # loss, _, _ = loss_fn(output, targets[:, 1:], mu, sigma, self.beta)
        # loss.backward()
        # self.optimizer.step()
        return pred


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')['MODEL_STATE'],strict=False)

        gen_mol = torch.empty(0)
        with torch.no_grad() :
            for z, tgt in self.train_data :
                z = z.to(self.local_rank)
                tgt = tgt.to(self.local_rank)

                for _ in range(max_len - 1) : 
                    pred = self._run_batch(z, tgt, get_mask(tgt,vocab).to(self.local_rank))
                    _, idx = torch.topk(pred, 1, dim=-1)
                    idx = idx[:, -1, :]
                    tgt = torch.cat([tgt, idx], dim=1)
                gen_mol = torch.cat([gen_mol, tgt.detach().cpu()], dim=0)
            gen_mol = gen_mol.detach().cpu().tolist()
            gen_mol = parallel_f(read_gen_smi, gen_mol)
            for i in gen_mol: 
                print(i)
            # result = metrics.get_all_metrics(gen_mol)
            # print(f'Epoch {epoch}:\n{result}')
            torch.cuda.empty_cache()


    # def _save_snapshot(self, epoch):
    #     snapshot = {
    #         "MODEL_STATE": self.model.module.state_dict(),
    #         "OPTIMIZER_STATE": self.optimizer.state_dict(),
    #         "EPOCHS_RUN": epoch,
    #     }

    #     torch.save(snapshot, f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')
    #     print(f"Epoch {epoch} | Training snapshot saved at checkpoint/{arg.save_name}/snapshot_{epoch}.pt")

    def train(self, max_epochs: int):
        self.model.eval()
        for epoch in range(self.epochs_run, max_epochs):
            if epoch > 90 : 
                self._run_epoch(epoch)
            # if self.local_rank == 0 and epoch % self.save_every == 0:
            #     # self._save_snapshot(epoch)
            #     pass


def load_train_objs():
    train_set = GenSet(30000, config['d_latent'])  
    model = Transformer(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=vocab,
        gvocab=gvocab)
    optimizer = None
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = f"snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()





main(0, 100, 2000)
