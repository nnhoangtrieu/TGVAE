import torch 
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torch_geometric 
from torch_geometric.loader import DataLoader as gDataLoader
import argparse
import data 
from data import ProcessData
import model.base
from model.base import Transformer
import utils 
from utils import monotonic_annealer, get_mask, parallel_f, seed_torch, cyclic_annealer
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol 
import datetime 
import os
from tqdm import tqdm
import metrics


def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def get_valid(smi) : 
    return smi if get_mol(smi) else None 
def get_novel(smi) : 
    return smi if smi not in smi_list else None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
rdkit.rdBase.DisableLog('rdApp.*') # Disable rdkit warnings
seed_torch()

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
# parser.add_argument('--kl_start', type=int, default=0)
parser.add_argument('--kl_type', type=str, default="monotonic")
parser.add_argument('--kl_w_start', type=float, default=0.0005)
parser.add_argument('--kl_w_end', type=float, default=0.005)
parser.add_argument('--kl_cycle', type=int, default=4)
parser.add_argument('--kl_ratio', type=float, default=0.9)
parser.add_argument('--save_name', type=str, default='your save name')
arg = parser.parse_args()



if arg.kl_type == 'cyclic':
    writer = SummaryWriter(f"{arg.save_name}/LEN {arg.max_len} LAYER {arg.n_layers} EPOCH {arg.n_epochs} TYPE {arg.kl_type} START {arg.kl_w_start} END {arg.kl_w_end} CYCLE {arg.kl_cycle} RATIO {arg.kl_ratio}")
elif arg.kl_type == 'monotonic' :
    writer = SummaryWriter(f"{arg.save_name}/LEN {arg.max_len} LAYER {arg.n_layers} EPOCH {arg.n_epochs} TYPE {arg.kl_type} START {arg.kl_w_start} END {arg.kl_w_end}")


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


model = Transformer(arg.d_model, arg.d_latent, arg.d_ff, arg.e_heads, arg.d_heads, arg.n_layers, arg.dropout, vocab, gvocab).to(device)
optim = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-6)
def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss


if arg.kl_type == 'monotonic' : 
    annealer = monotonic_annealer(arg.n_epochs, 0, arg.kl_w_start, arg.kl_w_end)
elif arg.kl_type == 'cyclic' :
    annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.n_epochs, arg.kl_cycle, arg.kl_ratio)

print(f'\n\nKL Annealer: {annealer}')
# annealer = monotonic_annealer(arg.n_epochs, arg.kl_start, arg.kl_w_start, arg.kl_w_end)
# annealer = cyclic_annealer(arg.kl_w_start, arg.kl_w_end, arg.n_epochs, arg.kl_cycle, arg.kl_ratio)




print('\n\n\n')
print('#########################################################################')
print('######################### START TRAINING ################################')
print('#########################################################################')
print('\n\n\n')

for epoch in range(arg.n_epochs) :
    train_loss, val_loss, recon_loss, kl_loss = 0, 0, 0, 0
    beta = annealer[epoch]

    model.train()
    # for src in train_loader :
    for src in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{arg.n_epochs} - Train') : 
        src = src.to(device)
        tgt = src.clone().smi.to(device)
        tgt_mask = get_mask(tgt[:, :-1], vocab) 
        pred, mu, sigma = model(src, tgt[:, :-1], None, tgt_mask)
        loss, recon, kl = loss_fn(pred, tgt[:, 1:], mu, sigma, beta)

        # train_loss += loss.item()
        # recon_loss += recon.item()
        # kl_loss += kl.item()

        loss.backward(), optim.step(), optim.zero_grad(), clip_grad_norm_(model.parameters(), 0.5)

    if epoch < 30 : 
        continue
    model.eval()
    gen_mol = torch.empty(0).to(device)
    with torch.no_grad() : 
        # for _ in range(60) :
        for _ in tqdm(range(60), desc='Generating Molecules...') : 
            z = torch.randn(500, arg.d_latent).to(device)
            tgt = torch.zeros(500, 1, dtype=torch.long).to(device)

            for _ in range(max_len - 1) : 
                pred = model.inference(z, tgt, None, get_mask(tgt, vocab).to(device))
                _, idx = torch.topk(pred, 1, dim=-1)
                idx = idx[:, -1, :]
                tgt = torch.cat([tgt, idx], dim=1)

            gen_mol = torch.cat([gen_mol, tgt], dim=0)
        gen_mol = gen_mol.tolist() 
        gen_mol = parallel_f(read_gen_smi, gen_mol)
        try : 
            result = metrics.get_all_metrics(gen_mol)

            valid = result['valid']
            unique1000 = result['unique@1000']
            unique10000 = result['unique@10000']
            print(result)

        except Exception as e :
            print(e)
            valid = 0 
            unique1000 = 0
            unique10000 = 0


        writer.add_scalar('Metric/valid', valid, epoch)
        writer.add_scalar('Metric/unique1000', unique1000, epoch)
        writer.add_scalar('Metric/unique10000', unique10000, epoch)
        # gen_mol = parallel_f(read_gen_smi, gen_mol)
        # valid_mol = parallel_f(get_valid, gen_mol)
        # valid_mol = [m for m in valid_mol if m != None]
        # unique_mol = set(valid_mol)

        # uniqueness = (len(unique_mol) / len(valid_mol)) * 100 if valid_mol else 0
        # novel_mol = [m for m in parallel_f(get_novel, unique_mol) if m is not None]
        # novelty = (len(novel_mol) / len(unique_mol)) * 100 if unique_mol else 0
        # validity = (len(valid_mol) / 30000) * 100 


        # with open(f'genmol-train/{cur_time}.txt', 'a') as file : 
        #     if epoch == 0 : 
        #         file.write('Model Parameters:\n')
        #         for name, value in arg.__dict__.items() : 
        #             file.write(f'\t{name} : {value}\n')
        #     file.write(f"Epoch: {epoch + 1} --- Train Loss: {train_loss / len(train_loader):3f}\n")
        #     file.write(f'Validity: {validity:.2f}% --- Uniqueness: {uniqueness:.2f}% --- Novelty: {novelty:.2f}%')

        #     for i, m in enumerate(set(novel_mol)) : 
        #         file.write(f'{i+1}. {m}\n')

    # writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
    # writer.add_scalar('Loss/Reconstruction', recon_loss / len(train_loader), epoch)
    # writer.add_scalar('Loss/KL', kl_loss / len(train_loader), epoch)
    # writer.add_scalar('Metric/Uniqueness', uniqueness, epoch)
    # writer.add_scalar('Metric/Validity', validity, epoch)

    # for i, m in enumerate(set(novel_mol)) : 
    #     print(f'{i+1}. {m}')
    print(f'Epoch: {epoch + 1}:')
    # print(f'\tTrain Loss: {train_loss / len(train_loader):.3f} --- Reconstruction Loss: {recon_loss / len(train_loader):.3f} --- KL Loss: {kl_loss / len(train_loader):.3f} --- Beta: {beta:5f}')
    # print(f'\tValidity: {validity:.2f}% --- Uniqueness: {uniqueness:.2f}% --- Novelty: {novelty:.2f}%\n')
    print(f'\tValidity: {valid:.2f}% --- Uniquene@1000: {unique1000:.2f}% --- Unique@10000: {unique10000:.2f}%\n')
