import os 
import torch 
import argparse 
from model.base import Transformer
import multiprocessing 
import metrics
from tqdm import tqdm
import datetime 
current = datetime.datetime.now()


parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--num_gen', type=int, default=30000)
parser.add_argument('--get_metric', type=int, default=1)
arg = parser.parse_args()

if not os.path.exists(f'checkpoint/{arg.save_name}') and arg.save_name != 'None' :
    print('Path not exists')
    print('Please look into folder checkpoint/single-gpu and choose the name of folder that you want to load the model from')
    exit()

print('Model found, loading model...\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Currently using {device}\n')

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

try :
    model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/model.pt')['MODEL_STATE'])
    print('Model loaded successfully')
except : 
    print('Model not found')
    exit()



model.eval() 
gen_mol = torch.empty(0).to(device)
with torch.no_grad() :
    for _ in tqdm(range(arg.num_gen // 1000), desc='Generating molecules') :
        z = torch.randn(arg.num_gen // (arg.num_gen // 1000), config['d_latent']).to(device)
        tgt = torch.zeros(arg.num_gen // (arg.num_gen // 1000), 1, dtype=torch.long).to(device)

        for _ in range(config['max_token_len']-1) : 
            pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
            _, idx = torch.topk(pred, 1, dim=-1)
            idx = idx[:, -1, :]
            tgt = torch.cat([tgt, idx], dim=1)

        gen_mol = torch.cat([gen_mol, tgt], dim=0)
        torch.cuda.empty_cache()
    gen_mol = gen_mol.tolist() 
    gen_mol = parallel_f(read_gen_smi, gen_mol)

    print('Generated Molecules: ')
    for i, mol in enumerate(gen_mol) : 
        print(f'{i+1}. {mol}')

    with open(f'genmol{current}', 'w') as f :
        for i, mol in enumerate(gen_mol) : 
            f.write(f'{i+1}. {mol}\n')

    if arg.get_metric == 1 :
        print('Calculating metrics...')
        result = metrics.get_all_metrics(gen_mol, k=(10000, 20000, 25000, 30000))
    else :
        print('Skip calculating metrics...')
        exit()

    for name, value in result.items() : 
        print(f'\t{name}: {value:.4f}')






