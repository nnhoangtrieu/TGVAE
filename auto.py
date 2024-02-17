import os
import time
import torch 
from torch.utils.tensorboard import SummaryWriter
from model.base import Transformer
import re
from utils import *
import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_folders(path):
    folders = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders

directory_path = "checkpoint"

folders = get_folders(directory_path)


dic = {f:['config.pt'] for f in folders}



while True : 
    for folder in folders : 
        config = torch.load(f'checkpoint/{folder}/config.pt')
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

        for snapshot in os.listdir(f'checkpoint/{folder}') : 
            if snapshot not in dic[folder] : 
                dic[folder].append(snapshot)

                model.load_state_dict(torch.load(f'checkpoint/{folder}/{snapshot}')['MODEL_STATE'])
                model.eval()
                print(f'Loaded model {folder} at {snapshot}')

                epoch = int(re.findall(r'\d+', snapshot)[0])
                gen_mol = torch.empty(0).to(device)
                with torch.no_grad() : 
                    for _ in range(1) : 
                        z = torch.randn(500, config['d_latent']).to(device)
                        tgt = torch.zeros(500, 1, dtype=torch.long).to(device) 

                        for _ in range(config['max_token_len'] -1) : 
                            pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
                            _, idx = torch.topk(pred, 1, dim=-1)
                            idx = idx[:, -1, :]
                            tgt = torch.cat([tgt, idx], dim=1)
                        gen_mol = torch.cat([gen_mol, tgt], dim=0)
                    gen_mol = gen_mol.tolist()
                    gen_mol = parallel_f(read_gen_smi, gen_mol)
                    # result = metrics.get_all_metrics(gen_mol)
                    print(gen_mol)


    
    
    
    
    time.sleep(0.5)