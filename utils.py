import re
import os
import json
import torch
import random
import argparse
import datetime
import numpy as np
import os.path as op
import torch.nn.functional as F
from tqdm import tqdm 
from rdkit import Chem 
from model.main import TGVAE
from argparse import Namespace
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence


edge_vocab = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}

path_script = op.dirname(op.abspath(__file__))
path_checkpoint_folder = op.join(path_script, 'checkpoint')
path_data_folder = op.join(path_script, 'data') 




def get_generate_path(arg) : 
    path_script = op.dirname(op.abspath(__file__))
    path_checkpoint = op.join(path_script, 'checkpoint', arg.name)
    path_config = arg.path_config if arg.path_config else op.join(path_checkpoint, 'config.json')
    path_snapshot = arg.path_snapshot if arg.path_snapshot else op.join(path_checkpoint, f'snapshot_{arg.snapshot}.pt') 
    return path_config, path_snapshot

def get_train_config() : 
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('-tr', '--train', type=str, default='moses_train.txt')
    parser.add_argument('-de', '--dim_encoder', type=int, default=512)
    parser.add_argument('-dd', '--dim_decoder', type=int, default=512)
    parser.add_argument('-dl', '--dim_latent', type=int, default=256)
    parser.add_argument('-def', '--dim_encoder_ff', type=int, default=512)
    parser.add_argument('-ddf', '--dim_decoder_ff', type=int, default=512)
    parser.add_argument('-nel', '--num_encoder_layer', type=int, default=4)
    parser.add_argument('-ndl', '--num_decoder_layer', type=int, default=4)
    parser.add_argument('-neh', '--num_encoder_head', type=int, default=1)
    parser.add_argument('-ndh', '--num_decoder_head', type=int, default=16)
    parser.add_argument('-doe', '--dropout_encoder', type=float, default=0.3)
    parser.add_argument('-dog', '--dropout_gat', type=float, default=0.3)
    parser.add_argument('-dod', '--dropout_decoder', type=float, default=0.3)

    # Training hyperparameters
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-gc', '--gradient_clipping', type=float, default=5.0)

    # Loss function hyperparametersd
    parser.add_argument('-lkl', '--loss_kl', type=str, default='mean')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
    parser.add_argument('-aes', '--anneal_epoch_start', type=int, default=0)
    parser.add_argument('-aws', '--anneal_weight_start', type=float, default=0.00005)
    parser.add_argument('-awe', '--anneal_weight_end', type=float, default=1.0)

    # Other hyperparameters
    parser.add_argument('-se', '--save_every', type=int, default=1)
    parser.add_argument('-ge', '--generate_every', type=int, default=1)
    parser.add_argument('-ss', '--start_save', type=int, default=5)
    parser.add_argument('-sg', '--start_generate', type=int, default=5)
    parser.add_argument('-n', '--name', type=str, default='experiment_1')

    config = parser.parse_args()

    config.trained_epoch = 0
    config.path_generate_folder = op.join(path_data_folder, 'generate', config.name)
    config.path_checkpoint_folder = op.join(path_checkpoint_folder, config.name)
    config.path_train_raw_file = op.join(path_data_folder, 'raw', config.train)
    config.path_train_processed_folder = op.join(path_data_folder, 'processed', op.splitext(config.train)[0])
    


    if op.exists(op.join(path_checkpoint_folder, config.name, 'config.json')) : 
        print(f'Resume training of training named {config.name}')
        return Namespace(**load(op.join(path_checkpoint_folder, config.name, 'config.json'))) # Load previous config 
    else : 
        os.makedirs(config.path_checkpoint_folder, exist_ok=True)
        os.makedirs(config.path_generate_folder, exist_ok=True)
        return config

def get_dataset(path_raw, path_processed) : 
    if op.exists(op.join(path_processed, 'data.pt')) :
        return load_processed_data(path_processed) 
    else : 
        raw_smi = read_smi(path_raw)
        smi, node_feature, edge_index, edge_attr, vocab_smi, vocab_graph, max_token = process_data(raw_smi)
        dataset = [MyData(x=nf, edge_index=ei,edge_attr=ea, smi=s) for nf, ei, ea, s in zip(node_feature, edge_index, edge_attr, smi)]
        save_processed_data(path_processed, dataset, vocab_smi, vocab_graph, max_token)
        return dataset, vocab_smi, vocab_graph, max_token

def save_processed_data(path, dataset, vocab_smi, vocab_graph, max_token) : 
    os.makedirs(path, exist_ok=True)
    with tqdm(total=4, desc='Saving processed data', unit='file') as pbar : 
        save(vocab_smi, op.join(path, 'vocab_smi.json')); pbar.update(1) 
        save(vocab_graph, op.join(path, 'vocab_graph.json')); pbar.update(1) 
        save(max_token, op.join(path, 'max_token.pt')); pbar.update(1)
        save(dataset, op.join(path, 'data.pt')); pbar.update(1) 

def load_processed_data(path) :
    with tqdm(total=4, desc='Loading processed data', unit='file') as pbar : 
       vocab_smi = load(op.join(path, 'vocab_smi.json')); pbar.update(1)
       vocab_graph = load(op.join(path, 'vocab_graph.json')); pbar.update(1)
       max_token = load(op.join(path, 'max_token.pt')); pbar.update(1)
       dataset = load(op.join(path, 'data.pt')); pbar.update(1)
    return dataset, vocab_smi, vocab_graph, max_token

def get_model(config, device, generate_snapshot=None) :
    model = TGVAE(dim_encoder=config.dim_encoder,
                    dim_decoder=config.dim_decoder,
                    dim_encoder_ff=config.dim_encoder_ff,
                    dim_latent=config.dim_latent,
                    dim_decoder_ff=config.dim_decoder_ff,
                    num_encoder_layer=config.num_encoder_layer,
                    num_decoder_layer=config.num_decoder_layer,
                    num_encoder_head=config.num_encoder_head,
                    num_decoder_head=config.num_decoder_head,
                    dropout_encoder=config.dropout_encoder,
                    dropout_gat=config.dropout_gat,
                    dropout_decoder=config.dropout_decoder,
                    size_graph_vocab=len(config.vocab_graph),
                    size_smi_vocab=len(config.vocab_smi),
                    device=device).to(device)
    if generate_snapshot : 
        model.load_state_dict(generate_snapshot['MODEL_STATE'])
        return model
    if config.trained_epoch == 0 : 
        optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else : 
        snapshot = load(op.join(config.path_checkpoint_folder, f'snapshot_{config.trained_epoch}.pt'))
        model.load_state_dict(snapshot['MODEL_STATE'])
        optim = torch.optim.Adam(model.parameters())
        optim.load_state_dict(snapshot['OPTIMIZER_STATE'])

    annealer = monotonic_annealer(config.epoch, 
                                  config.anneal_epoch_start,
                                  config.anneal_weight_start,
                                  config.anneal_weight_end)
    
    return model, optim, annealer

def convert_data(data, smi_vocab, device=None) : 
    inp_graph, smi = data.to(device), data.smi.to(device) 
    inp_smi, tgt_smi = smi[:, :-1], smi[:, 1:]
    inp_smi_mask = get_mask(inp_smi, smi_vocab)
    return inp_graph, inp_smi, inp_smi_mask.to(device), tgt_smi

def checkpoint(model, optim, epoch, config) : 
    snapshot = {'MODEL_STATE': model.state_dict(), 'OPTIMIZER_STATE': optim.state_dict()}
    save(snapshot, op.join(config.path_checkpoint_folder, f'snapshot_{epoch}.pt'))
    update_config(config, {'trained_epoch': epoch})


# Annealer
def monotonic_annealer(n_epoch, epoch_start, weight_start, weight_end):
    inc = (weight_end - weight_start) / (n_epoch - epoch_start)
    annealing_weights = []
    for i in range(n_epoch):
        k = (i - epoch_start) if i >= epoch_start else 0
        annealing_weights.append(weight_start + k * inc)
    return annealing_weights

def cyclic_annealer(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch) * stop
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)
    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


# Loss function 
def recon_loss_fn(pred, tgt, vocab, weight) :
    return F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['[PAD]'], weight=weight)

def kl_loss_fn(mu, sigma, loss_kl) : 
    if loss_kl == 'mean' : return -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
    elif loss_kl == 'sum' : return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / mu.size(0)

def loss_fn(out, tgt, beta, config, weight=None) :
    pred, mu, sigma = out
    return recon_loss_fn(pred, tgt, config.vocab_smi, weight) + beta * kl_loss_fn(mu, sigma, config.loss_kl)


# Data processing
def process_smi(smi, vocab, tokenizer) : 
    out = []
    for t in tokenizer(smi) : 
        if t not in vocab : vocab[t] = len(vocab)
        out.append(vocab[t])
    out = [vocab['[START]']] + out + [vocab['[END]']]
    return torch.tensor(out, dtype=torch.long)

def process_graph(smi, graph_vocab, edge_vocab) : 
    mol = Chem.MolFromSmiles(smi)
    node_feature, edge_index, edge_attr = [], [], []

    for atom in mol.GetAtoms() : 
        symbol = atom.GetSymbol() 
        if symbol not in graph_vocab : graph_vocab[symbol] = len(graph_vocab)
        node_feature.append(graph_vocab[symbol])

    for bond in mol.GetBonds() : 
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[b, e], [e, b]]
        edge_attr += [edge_vocab[str(bond.GetBondType())]] * 2

    node_feature = torch.tensor(node_feature)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().view(2, -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    return node_feature, edge_index, edge_attr

def process_data(data) : 
    smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
    graph_vocab = {}

    smi_list = []
    node_feature_list, edge_index_list, edge_attr_list = [], [], []

    for smi in tqdm(data, desc='Processing data') : 

        tokenized_smi = process_smi(smi, smi_vocab)
        node_feature, edge_index, edge_attr = process_graph(smi, graph_vocab, edge_vocab)

        smi_list.append(tokenized_smi)
        node_feature_list.append(node_feature)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    smi_list = pad_sequence(smi_list, batch_first=True, padding_value=smi_vocab['[PAD]'])

    return smi_list, node_feature_list, edge_index_list, edge_attr_list, smi_vocab, graph_vocab, smi_list.shape[1]

def token2smi(token, inv_vocab_smi) : 
    smiles = ''.join([inv_vocab_smi[t] for t in token])
    smiles = re.sub(r"\[START\]|\[PAD\]|\[END\]", "", smiles)
    return smiles 

def convert_token(token, vocab_smi) : 
    inv_vocab_smi = {v:k for k, v in vocab_smi.items()}
    token = token.tolist() 
    smiles = [token2smi(t, inv_vocab_smi) for t in token]
    return smiles

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smi)]
    assert smi == ''.join(tokens), ("{} could not be joined".format(smi))
    return tokens

def read_smi(path, delimiter='\t', titleLine=False) : 
    result = [] 
    if path.endswith('.txt') :
        with open(path, 'r') as f : 
            for smi in tqdm(f.readlines(), desc='Reading SMILES') : 
                if Chem.MolFromSmiles(smi) is not None : 
                    result.append(smi.strip())
    elif path.endswith('.sdf') :
        supplier = Chem.SDMolSupplier(path)
        for mol in tqdm(supplier, desc='Reading SMILES') : 
            if mol is None : 
                continue 
            result.append(Chem.MolToSmiles(mol))
    elif path.endswith('.smi') :
        supplier = Chem.SmilesMolSupplier(path, delimiter=delimiter, titleLine=titleLine)
        for mol in tqdm(supplier, desc='Reading SMILES') : 
            if mol is None : 
                continue 
            result.append(Chem.MolToSmiles(mol))
    return result

class MyData(Data) : 
    def __cat_dim__(self, key, value, *args, **kwargs) : 
        if key == 'smi' :
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 

# Model functions
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def get_mask(target, smi_vocab) :
    mask = (target != smi_vocab['[PAD]']).unsqueeze(-2)
    return mask & subsequent_mask(target.size(-1)).type_as(mask.data)

def generate_molecule(model, config, num_gen, path, batch=500) : 
    model.eval()
    with torch.no_grad() :
        for _ in tqdm(range(num_gen // batch), desc='Generating') : 
            smi_token = model.generate(config, num_gen=batch)
            smi = convert_token(smi_token, config.vocab_smi)
            save(smi, path, mode='a')

# Utils
def save(data, path, mode='w') : 
    if path.endswith('.pt') :
        torch.save(data, path)
    elif path.endswith('.json') :
        with open(path, 'w') as f : 
            json.dump(data, f, indent=4)
    elif path.endswith('.txt') : 
        with open(path, mode) as f : 
            for line in data : 
                f.write(line+'\n')

def load(file) : 
    if file.endswith('.pt') : 
        return torch.load(file)
    elif file.endswith('.json') :
        with open(file, 'r') as f : 
            return json.load(f)
    elif file.endswith('.txt') :
        with open(file, 'r') as f : 
            return [line.strip() for line in f.readlines()]

def update_config(config, dic) : 
    path_config = config.path_checkpoint_folder
    config = vars(config)
    for key, value in dic.items() : 
        config[key] = value
    save(config, op.join(path_config, 'config.json'))

def set_seed(seed):
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's CUDA seed (single GPU)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    