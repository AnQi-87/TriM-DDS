import csv
import os
import numpy as np
import networkx as nx
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdMolDescriptors
from utils_test import *


def loadSmilesAndSave(smis, path):
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)


def generate_3d_features(smile, max_atoms=100):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    coords = np.array(coords, dtype=np.float32)

    
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = atom_features(atom)  
        atom_feats.append(feat)
    atom_feats = np.array(atom_feats, dtype=np.float32)

    
    combined = np.concatenate([coords, atom_feats], axis=1)  
    return combined


def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))  
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


def creat_data(datafile, cellfile):
    file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)

    df_smiles = pd.read_csv('data/smiles.csv')
    compound_iso_smiles = list(df_smiles['smile'])
    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    smile_3d_feat = {}  
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

        
        feat_3d = generate_3d_features(smile)
        smile_3d_feat[smile] = feat_3d

    datasets = datafile
    save_img_dir = f"data/processed/{datasets}/images/"
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    
    save_3d_dir = f"data/processed/{datasets}/3d_feats/"
    if not os.path.exists(save_3d_dir):
        os.makedirs(save_3d_dir)

    smile_imageidx = {}
    for idx, smile in enumerate(compound_iso_smiles):
        loadSmilesAndSave(smile, f"{save_img_dir}/{idx}.png")
        
        np.save(f"{save_3d_dir}/{idx}.npy", smile_3d_feat[smile])
        smile_imageidx[smile] = idx

    if not os.path.isfile(f'data/processed/{datasets}_train.pt'):
        df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
        return drug1, drug2, cell, label, smile_graph, cell_features, smile_imageidx
    else:
        
        return None, None, None, None, smile_graph, cell_features, smile_imageidx


if __name__ == "__main__":
    cellfile = 'data/cell_features_954.csv'
    da = ['new_labels_0_10']
    for datafile in da:
        creat_data(datafile, cellfile)