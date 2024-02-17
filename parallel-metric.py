import argparse
import torch 
from model.base import Transformer
from torch.utils.tensorboard import SummaryWriter
import os
from utils import get_mask
import multiprocessing
import metrics
import time

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)
def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser() 
parser.add_argument('--save_name', type=str, default='your_save_name')

arg = parser.parse_args()


if not os.path.exists(f'checkpoint/{arg.save_name}') : 
    print('Path not exists')
    exit()


config = torch.load(f'checkpoint/{arg.save_name}/config.pt')
inv_vocab = {v: k for k, v in config['vocab'].items()}

model = Transformer(
    d_model=config['d_model'],
    d_latent=config['d_latent'],
    d_ff=config['d_ff'],
    e_heads=config['e_heads'],
    d_heads=config['d_heads'],
    num_layer=config['n_layers'],
    dropout=config['dropout'],
    vocab=config['vocab'],
    gvocab=config['gvocab']
).to(device)

writer = SummaryWriter(f'tensorboard/{arg.save_name}/result')

processed = []

while True :
    model.eval() 
    print('Sleeping')
    time.sleep(10)
    for epoch in range(len(os.listdir(f'checkpoint/{arg.save_name}')) - 1) :
        if epoch not in processed : 
            if epoch < 30 : 
                continue
            processed.append(epoch)
            model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')['MODEL_STATE'])
            model.eval()
            print(f'Loaded model {arg.save_name} at epoch {epoch}')

            gen_mol = torch.empty(0).to(device)
            with torch.no_grad() : 
                for _ in range(10) : 
                    z = torch.randn(3000, config['d_latent']).to(device)
                    tgt = torch.zeros(3000, 1, dtype=torch.long).to(device) 

                    for _ in range(config['max_token_len'] -1) : 
                        pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
                        _, idx = torch.topk(pred, 1, dim=-1)
                        idx = idx[:, -1, :]
                        tgt = torch.cat([tgt, idx], dim=1)
                    gen_mol = torch.cat([gen_mol, tgt], dim=0)
                    torch.cuda.empty_cache()

                gen_mol = gen_mol.tolist()
                gen_mol = parallel_f(read_gen_smi, gen_mol)
                result = metrics.get_all_metrics(gen_mol)

                for name, value in result.items() : 
                    writer.add_scalar(name, value, epoch)

                with open(f'genmol/{arg.save_name}.txt','a') as f :
                    f.write(f'Epoch {epoch}\n')
                    for name, value in result.items() :
                        f.write(f'{name}: {value}\n')
                    for i, mol in enumerate(gen_mol) : 
                        f.write(f'{i}. {mol}\n')
