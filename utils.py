import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path
import numpy as np 
import torch 
import multiprocessing
import re 
import random 
import os

def seed_torch(seed=910):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

def get_pharmacophore(smi, feature) : 
    "feature: Acceptor, Donor, Hydrophobe, Aromatic"
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 
    
    features = feature_factory.GetFeaturesForMol(mol_h, includeOnly=feature)
    coor = [f.GetPos() for f in features]
    coor = [[i.x,i.y,i.z] for i in coor]
    return coor



def get_coor(smi) :
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    mol = rdkit.Chem.MolFromSmiles(smi)
    num_atom = mol.GetNumAtoms()
    mol_h = rdkit.Chem.AddHs(mol)
    rdkit.Chem.rdDistGeom.EmbedMolecule(mol_h)
    conformer = mol_h.GetConformer()
    coor = conformer.GetPositions()
    
    features = feature_factory.GetFeaturesForMol(mol_h, includeOnly="Acceptor")
    p_coor = [f.GetPos() for f in features] 
    p_coor = [[i.x, i.y, i.z] for i in p_coor]
    return np.round(coor, 2), np.round(p_coor, 2)


def get_atom_mat(smi, vocab) : 
    mol = get_mol(smi) 
    mat = rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol)

    for i, atom in enumerate(mol.GetAtoms()) : 
        neighbor = list(atom.GetNeighbors())
        symbol = [(n.GetSymbol(), n.GetIdx()) for n in neighbor]
        for s in symbol : 
            mat[i][s[1]] = vocab[s[0]]
    return mat 
        
def get_bond_mat(smi) : 
    mol = rdkit.Chem.MolFromSmiles(smi) 
    mat = rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True, emptyVal=0) 
    return mat

def pad_mat(mat, max_len) :
   cur_size = mat.shape[0] 
   out = np.zeros((max_len, max_len))
   out[:cur_size, :cur_size] = mat 
   return out






###### Utils Function ######
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def get_mask( target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)

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

def get_ei(smi) : 
    mol = rdkit.Chem.MolFromSmiles(smi) 
    n_atoms = mol.GetNumAtoms() 
    ei = []
    for bond in mol.GetBonds() :
        b = bond.GetBeginAtomIdx() 
        e = bond.GetEndAtomIdx() 
        ei.append([b,e])
    for bond in mol.GetBonds() :
        b = bond.GetBeginAtomIdx() 
        e = bond.GetEndAtomIdx() 
        ei.append([e, b])
    return torch.tensor(ei).T

def get_nf(smi, vocab) : 
    mol = rdkit.Chem.MolFromSmiles(smi) 
    atom_list = [vocab[atom.GetSymbol()] for atom in mol.GetAtoms()]
    return torch.tensor(atom_list)

def get_gvocab(smi_list) :
    dic = {}
    for smi in smi_list :
        mol = rdkit.Chem.MolFromSmiles(smi) 
        for atom in mol.GetAtoms() : 
            symbol = atom.GetSymbol() 
            if symbol not in dic : 
                dic[symbol] = len(dic) 
    return dic 

def get_vocab(smi_list) :
    dic = {'<START>': 0, '<END>': 1, '<PAD>': 2}
    for smi in smi_list :
        for char in smi :
            if char not in dic :
                dic[char] = len(dic) 
    return dic 


def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens
def pad(smi, max_len) :
    return smi + [2] * (max_len - len(smi))

def encode(smi, vocab) :
    return [0] + [vocab[char] for char in smi] + [1]

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def get_valid(smi) : 
    return smi if get_mol(smi) else None 
def get_novel(smi) : 
    return smi if smi not in smi_list else None 