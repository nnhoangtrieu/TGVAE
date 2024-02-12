import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path
import numpy as np 

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