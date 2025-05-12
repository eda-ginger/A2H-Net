########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
# https://github.com/JK-Liu7/AttentionMGT-DTA/tree/main

########################################################################################################################
########## Import
########################################################################################################################

import torch
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

import logging
logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')  

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_mol_edge_list_and_feat_mtx(mol_graph, pad=False):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_index = undirected_edge_list.T
    if pad:
        max_drug_nodes = 290
        actual_node_shape = n_features.shape
        num_virtual_nodes = max_drug_nodes - actual_node_shape[0]
        virtual_node_feat = torch.zeros(num_virtual_nodes, actual_node_shape[1])
        n_features = torch.cat((n_features, virtual_node_feat), dim=0)  # 290, feats

        # add self-loops
        edge_index_with_self_loop, _ = add_self_loops(edge_index, num_nodes=max_drug_nodes)
        edge_index = edge_index_with_self_loop

    return edge_index, n_features


def drug_to_graph(smi, pad=False, file=True):
    if file:
        mol = Chem.MolFromMol2File(smi)
    else:
        mol = Chem.MolFromSmiles(smi)
    if mol:
        edge_index, n_features = get_mol_edge_list_and_feat_mtx(mol, pad)
        return Data(x=n_features, edge_index=edge_index)


def protein_to_graph(protein, pfd, prot_inform, pad=False):
    seq, key = protein
    find_idx = prot_inform.index[prot_inform.eq(key).any(axis=1)]
    if not find_idx.empty:
        find_row = prot_inform.loc[find_idx, :].to_dict('records')[0]
        if seq == find_row['Seq']:
            for k, v in find_row.items():
                if k != 'Seq':
                    file = pfd / f"{v}.pt"
                    if file.is_file():
                        f = torch.load(file)
                        if pad:
                            max_nodes = 671 # maximum protein graph nodes
                            actual_node_shape = f.x.shape
                            num_virtual_nodes = max_nodes - actual_node_shape[0]
                            virtual_node_feat = torch.zeros(num_virtual_nodes, actual_node_shape[1])
                            f.x = torch.cat((f.x, virtual_node_feat), dim=0)  # 290, feats

                            # add self-loops
                            edge_index_with_self_loop, _ = add_self_loops(f.edge_index, num_nodes=max_nodes)
                            f.edge_index = edge_index_with_self_loop
                        return Data(x=f.x, edge_index=f.edge_index)

if __name__ == '__main__':
    dr = 'COc1cc2c(cc1Cl)C(c1ccc(Cl)c(Cl)c1)=NCC2'
    dg = drug_to_graph(dr)
    dg_pad = drug_to_graph(dr, pad=True)
    print(dg, dg_pad)
