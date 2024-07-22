import argparse 
import json 
import torch
import os
from model.main import TGVAE
from tqdm import tqdm 
from helper.utils import get_mask, read_genmol, write_genmol, load_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='res')
parser.add_argument('-e', '--epoch', type=int, default=0) 
parser.add_argument('-d', '--divisor', type=int, default=20)
arg = parser.parse_args()


model_folder = f'./output/checkpoint/{arg.name}'
processed_folder = 'data/processed'
output_folder = f'output/generate/{arg.name}'

config = load_file(f'{model_folder}/config.json')
node_vocab = load_file(f'{processed_folder}/{config['train']}/node_vocab.json')
edge_vocab = load_file(f'{processed_folder}/{config['train']}/edge_vocab.json')
smi_vocab = load_file(f'{processed_folder}/{config['train']}/smi_vocab.json')
maxlen = load_file(f'{processed_folder}/{config['train']}/max_len.pt')
inv_smi_vocab = {v:k for k,v in smi_vocab.items()}





model = TGVAE(d_model=config['d_model'],
              d_ff=config['d_ff'],
              edge_dim=config['edge_dim'],
              n_head=config['n_head'],
              dropout=config['dropout'],
              n_layer=config['n_layer'],
              smi_vocab_size=len(smi_vocab),
              node_vocab_size=len(node_vocab),
              edge_vocab_size=len(edge_vocab),
              encoder_mode=config['encoder_mode'],
              pool_mode=config['pool_mode'],
              gnn_mode=config['gnn_mode']).to(device)




os.makedirs(output_folder, exist_ok=True)
for e in range(1, 101) : 
    if os.path.exists(f'{output_folder}/E{e}.txt') :
        continue
    if arg.epoch : 
        e = arg.epoch
    model.load_state_dict(torch.load(f'{model_folder}/snapshot_{e}.pt')['MODEL_STATE'])

    model.eval()
    genmol = torch.empty(0).to(device) 
    with torch.no_grad() :
        for _ in tqdm(range(arg.divisor), desc=f'Generating molecules epoch {e}') :
            z = torch.randn(30000 // arg.divisor, config['d_model'][0]).to(device)
            tgt = torch.zeros(30000 // arg.divisor, 1, dtype=torch.long).to(device)

            for _ in range(maxlen-1) : 
                pred = model.inference(z, tgt, get_mask(tgt, smi_vocab).to(device))
                _, idx = torch.topk(pred, 1, dim=-1)
                idx = idx[:, -1, :]
                tgt = torch.cat([tgt, idx], dim=1)

            genmol = torch.cat([genmol, tgt], dim=0)
            torch.cuda.empty_cache()

        genmol = [read_genmol(m, inv_smi_vocab) for m in genmol.tolist()]
        write_genmol(genmol, f'{output_folder}/E{e}.txt')

        if arg.epoch : 
            break




