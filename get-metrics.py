import os 
import torch 
import argparse
from model.base import Transformer 
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import metrics
from tqdm import tqdm
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
parser.add_argument('--save_path', type=str, default='None')
parser.add_argument('--save_name', type=str, default="None")
parser.add_argument('--save_epoch', type=int, default=-1)
arg = parser.parse_args()


if not os.path.exists(f'checkpoint/single-gpu/{arg.save_name}') and arg.save_name != 'None' : 
    print('Path not exists')
    exit()

if not os.path.exists(f'genmol/single-gpu/{arg.save_name}') :
    os.makedirs(f'genmol/single-gpu/{arg.save_name}')

config = torch.load(f'checkpoint/single-gpu/{arg.save_name}/config.pt') if arg.save_name != 'None' else torch.load(f'{arg.save_path}/config.pt')
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


writer = SummaryWriter(f'tensorboard/single-gpu/{arg.save_name}')

model.eval()
if arg.save_epoch == -1 :
    for epoch in range(1, config['n_epochs'] + 1) :
        gen_mol = torch.empty(0).to(device)
        if epoch <= 29 :
            with open(f'genmol/single-gpu/{arg.save_name}/e_{epoch}', 'w') as f :
                f.write(f'Epoch {epoch} skipped...')
                print(f'Epoch {epoch} skipped...')
                continue
        if epoch <= len(os.listdir(f'genmol/single-gpu/{arg.save_name}')) : 
            print(f'Epoch {epoch} already exists')
            continue  
        else :
            try :
                model.load_state_dict(torch.load(f'checkpoint/single-gpu/{arg.save_name}/snapshot_{epoch}.pt')['MODEL_STATE'])
                print(f'Loaded snapshot_{epoch}.pt')
            except :
                print("Snapshot not found")
                continue

            with torch.no_grad() : 
                print(f'Epoch {epoch} generating molecules...')
                for _ in range(10) :
                    z = torch.randn(3000, config['d_latent']).to(device)
                    tgt = torch.zeros(3000, 1, dtype=torch.long).to(device)

                    for _ in range(config['max_token_len']-1) : 
                        pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
                        _, idx = torch.topk(pred, 1, dim=-1)
                        idx = idx[:, -1, :]
                        tgt = torch.cat([tgt, idx], dim=1)

                    gen_mol = torch.cat([gen_mol, tgt], dim=0)
                    torch.cuda.empty_cache()
                gen_mol = gen_mol.tolist() 

                gen_mol = parallel_f(read_gen_smi, gen_mol)
                result = metrics.get_all_metrics(gen_mol, k=(10000, 20000, 25000, 30000))

                for name, value in result.items() : 
                    writer.add_scalar(name, value, epoch)
                    print(f'\t{name}: {value:.4f}')

                with open(f'genmol/single-gpu/{arg.save_name}/e_{epoch}', 'w') as f : 
                    for i, mol in enumerate(gen_mol[:1000]) : 
                        f.write(f'{i+1}. {mol}\n')
                    f.write(f'Epoch {epoch}:\n{result}')




else : 
    try :
        if arg.save_name == 'None' : 
            model.load_state_dict(torch.load(f'{arg.save_path}/snapshot_{arg.save_epoch}.pt')['MODEL_STATE'])
        else :
            model.load_state_dict(torch.load(f'checkpoint/single-gpu/{arg.save_name}/snapshot_{arg.save_epoch}.pt')['MODEL_STATE'])
        print(f'Loaded snapshot_{arg.save_epoch}.pt')
    except :
        print("Snapshot not found")
        exit()

    gen_mol = torch.empty(0).to(device)
    
    with torch.no_grad() : 
        print(f'Epoch {arg.save_epoch} generating molecules...')
        for _ in tqdm(range(30)) :
            z = torch.randn(1000, config['d_latent']).to(device)
            tgt = torch.zeros(1000, 1, dtype=torch.long).to(device)

            for _ in range(config['max_token_len']-1) : 
                pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
                _, idx = torch.topk(pred, 1, dim=-1)
                idx = idx[:, -1, :]
                tgt = torch.cat([tgt, idx], dim=1)

            gen_mol = torch.cat([gen_mol, tgt], dim=0)
            torch.cuda.empty_cache()
        gen_mol = gen_mol.tolist() 

        gen_mol = parallel_f(read_gen_smi, gen_mol)
        # for i, mol in enumerate(gen_mol[:100]) :
        #     print(f'{i+1}. {mol}')
        result = metrics.get_all_metrics(gen_mol, k=(10000, 20000, 25000, 30000))



        for name, value in result.items() : 
            print(f'\t{name}: {value:.4f}')