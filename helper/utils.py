import json
import re 
import os
import torch 
import torch.nn.functional as F 
import random 
import numpy as np
from tqdm import tqdm 
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data


def seed_torch(seed=910):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def get_token(smi_list) : 
    smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
    token_list = [] 
    maxlen = 0 

    for smi in tqdm(smi_list, desc='Tokenizing SMILES') : 
        token = []
        smi = tokenizer(smi)

        # Get token
        for char in smi : 
            if char not in smi_vocab : 
                smi_vocab[char] = len(smi_vocab)
            token.append(smi_vocab[char])
        token = [smi_vocab['[START]']] + token + [smi_vocab['[END]']]

        # Get longest token 
        if len(token) > maxlen :
            maxlen = len(token)
        token_list.append(token)
    for i, token in enumerate(token_list) : 
            token_list[i] = torch.tensor(token + [smi_vocab['[PAD]']] * (maxlen - len(token)), dtype=torch.long)
    
    return token_list, smi_vocab, maxlen

def preprocess(smi_list, output_folder=None) : 
    smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
    node_vocab = {'[MASK]': 0}
    # edge_vocab = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
    edge_vocab = {'[MASK]': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}
    token_list, nf_list, ei_list, ew_list = [], [], [], []
    max_len = 0 

    for i, smi in tqdm(enumerate(smi_list), desc='Processing data...') : 
        token, nf, ei, ew, temp = [], [], [], [], []
        mol = MolFromSmiles(smi)
        smi = tokenizer(smi)

        if mol is None : 
            continue

        # Get token
        for char in smi : 
            if char not in smi_vocab : 
                smi_vocab[char] = len(smi_vocab)
            token.append(smi_vocab[char])
        token = [smi_vocab['[START]']] + token + [smi_vocab['[END]']]

        # Get longest token 
        if len(token) > max_len :
            max_len = len(token)

        # Get graph vocab, node features
        for atom in mol.GetAtoms() : 
            symbol = atom.GetSymbol()
            if symbol not in node_vocab : 
                node_vocab[symbol] = len(node_vocab)
            nf.append(node_vocab[symbol])
        
        # Get edge index, edge weight
        for bond in mol.GetBonds() :
            b = bond.GetBeginAtomIdx()
            e = bond.GetEndAtomIdx()
            ei.append([b,e])
            temp.append([e, b])
            ew.append(edge_vocab[str(bond.GetBondType())])
        ei.extend(temp)
        ew = ew * 2

        nf = torch.tensor(nf)


        ei = torch.tensor(ei, dtype=torch.int64).T
        if ei.dim() != 2 :
            print(i)
            continue
        ew = torch.tensor(ew)

        token_list.append(token), nf_list.append(nf), ei_list.append(ei), ew_list.append(ew)

    # Pad token 
    for i, token in enumerate(token_list) : 
        token_list[i] = torch.tensor(token + [smi_vocab['[PAD]']] * (max_len - len(token)), dtype=torch.long)
    

    if output_folder is not None :
        os.makedirs(output_folder, exist_ok=True)
        save_file(smi_vocab, f'{output_folder}/smi_vocab.json')
        save_file(node_vocab, f'{output_folder}/node_vocab.json')
        save_file(edge_vocab, f'{output_folder}/edge_vocab.json')
        save_file(token_list, f'{output_folder}/token_list.pt')
        save_file(nf_list, f'{output_folder}/nf_list.pt')
        save_file(ei_list, f'{output_folder}/ei_list.pt')
        save_file(ew_list, f'{output_folder}/ew_list.pt')
        save_file(max_len, f'{output_folder}/max_len.pt')


    return nf_list, ei_list, ew_list, token_list, smi_vocab, node_vocab, edge_vocab, max_len


class MyData(Data) : 
    def __cat_dim__(self, key, value, *args, **kwargs) : 
        if key == 'smi' :
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 
    

def loss_fn(pred, tgt, mu, sigma, beta, vocab, batch) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['[PAD]'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss



def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def get_mask(target, smi_vocab) :
        mask = (target != smi_vocab['[PAD]']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)

def cyclic_annealer(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch) * stop
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def monotonic_annealer(n_epoch, kl_start, kl_w_start, kl_w_end):
    i_start = kl_start
    w_start = kl_w_start
    w_max = kl_w_end

    inc = (w_max - w_start) / (n_epoch - i_start)

    annealing_weights = []
    for i in range(n_epoch):
        k = (i - i_start) if i >= i_start else 0
        annealing_weights.append(w_start + k * inc)

    return annealing_weights


def genmol_to_smi(t, inv_smi_vocab) : 
    smiles = ''.join([inv_smi_vocab[i] for i in t])
    smiles = smiles.replace("[START]", "").replace("[PAD]", "").replace("[END]","")
    return smiles 


def load_file(file) : 
    if file.endswith('.pt') : 
        return torch.load(file)
    elif file.endswith('.json') :
        with open(file, 'r') as f : 
            return json.load(f)
    elif file.endswith('.txt') :
        with open(file, 'r') as f : 
            return [line.strip() for line in f.readlines()]
    
def save_file(data, path) : 
    if path.endswith('.pt') :
        torch.save(data, path)
    elif path.endswith('.json') :
        with open(path, 'w') as f : 
            json.dump(data, f, indent=4)
    elif path.endswith('.txt') : 
        with open(path, 'w') as f : 
            for line in data : 
                f.write(line+'\n')


def read_smi_file(file) : 
    with open(file, 'r') as f : 
        return [line.strip() for line in f.readlines()]
    

def write_genmol(genmol, path) : 
    with open(path, 'w') as f : 
        for mol in genmol : 
            f.write(mol+'\n')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 