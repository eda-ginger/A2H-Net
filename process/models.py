########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/hkmztrk/DeepDTA
# https://github.com/thinng/GraphDTA
# https://github.com/KailiWang1/DeepDTAF
# https://github.com/lennylv/CAPLA

########################################################################################################################
########## Import
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm

from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from utils.seq_to_vec import CHARISOSMILEN, CHARPROTLEN

########################################################################################################################
########## A2H-Net
########################################################################################################################

import numpy as np
import torch
import torch.nn as nn
from utils.gvp_utils import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean


class InducedFitNet(nn.Module):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 output_v_dim=3,
                 direction_only=True,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(InducedFitNet, self).__init__()
        
        self.direction_only = direction_only
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        # self.W_out = nn.Sequential(
        #     LayerNorm(node_h_dim),
        #     GVP(node_h_dim, (ns, 0)))
        
        
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, output_v_dim)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        v_out, _ = h_V # (num_nodes, ns)
        out_vecs = self.dense(v_out) # (num_nodes, output_v_dim)

        if self.direction_only:
            # 방향성만 볼때 (F.cosine_similarity 사용)
            return F.normalize(out_vecs, dim=-1) # unit vector로 정규화
        else:
            # 방향성과 크기 모두 볼때 (F.mse_loss 사용)
            return out_vecs

        # # original MQAModule
        # out = self.W_out(h_V)
        # if batch is None: out = out.mean(dim=0, keepdims=True)
        # else: out = scatter_mean(out, batch, dim=0)
        # return self.dense(out).squeeze(-1) + 0.5




########################################################################################################################

class A2HNet_GAT(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                 embed_dim=128, output_dim=128, dropout=0.2):
        super(A2HNet_GAT, self).__init__()

        # graph layers for drug
        self.drug_gat1 = GATConv(num_features_xd, num_features_xd, heads=5, dropout=dropout)
        self.drug_gat2 = GATConv(num_features_xd * 5, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=5)
        self.conv_xt2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_xt1 = nn.Linear(32, output_dim)
        
        # A2H graph branch
        self.a2h_gat1 = GATConv(24, 64, heads=5, dropout=dropout, edge_dim=5)
        self.a2h_gat2 = GATConv(64 * 5, 128, dropout=dropout, edge_dim=5)
        self.a2h_fc1 = nn.Linear(128, output_dim)
        
        # combined layers
        self.fc1 = nn.Linear(384, 1024)  # 128 + 128 + 128 = 384
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        drug, target, a2h = data 
        target = target.x

        # drug graph input feed-forward
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x = self.dropout(x)
        x = F.elu(self.drug_gat1(x, edge_index))
        x = self.dropout(x)
        x = self.drug_gat2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein sequence input feed-forward
        embedded_xt = self.embedding_xt(target)  # (batch_size, seq_len, embed_dim)
        embedded_xt = embedded_xt.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.adaptive_pool(conv_xt)  # (batch_size, 32, 1)
        conv_xt = conv_xt.squeeze(-1)  # (batch_size, 32)
        xt = self.fc_xt1(conv_xt)  # (batch_size, output_dim)
        
        # A2H graph input feed-forward
        a2h_x, a2h_edge_index, a2h_edge_attr, a2h_batch = a2h.x, a2h.edge_index, a2h.edge_attr, a2h.batch
        a2h_x = self.dropout(a2h_x)
        
        a2h_x = F.elu(self.a2h_gat1(a2h_x, a2h_edge_index, a2h_edge_attr))
        a2h_x = self.dropout(a2h_x)
        a2h_x = self.a2h_gat2(a2h_x, a2h_edge_index, a2h_edge_attr)
        a2h_x = self.relu(a2h_x)
        a2h_x = gmp(a2h_x, a2h_batch)
        a2h_x = self.a2h_fc1(a2h_x)
        a2h_x = self.relu(a2h_x)

        # concat all modalities
        xc = torch.cat((x, xt, a2h_x), 1)  # (batch_size, 384)
        
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class A2HNet_SEQ(torch.nn.Module):
    def __init__(self, n_filters=32, num_features_xd=78, n_output=1, num_features_xt=25,
                 embed_dim=128, output_dim=128, dropout=0.2):
        super(A2HNet_SEQ, self).__init__()

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(CHARISOSMILEN + 1, 128) # batch, 100, 128 -> batch, 128, 100
        self.conv_xd_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=4) # batch, 32, 97
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=4) # batch, 64, 94
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=4) # batch, 96, 91

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128 -> batch, 128, 1000
        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8) # batch, 32, 993
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8) # batch, 64, 986
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=8) # batch, 96, 979
        
        # A2H graph branch
        self.a2h_gat1 = GATConv(24, 64, heads=5, dropout=dropout, edge_dim=5)
        self.a2h_gat2 = GATConv(64 * 5, 96, dropout=dropout, edge_dim=5)
        self.a2h_fc1 = nn.Linear(96, 96)
        
        # activation and regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(n_filters * 9, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def conv_module(self, x, conv1, conv2, conv3):
        x = conv1(x)
        x = F.relu(x)
        x = conv2(x)
        x = F.relu(x)
        x = conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, data):
        drug, target, a2h = data 

        # drug
        xd = drug.x
        embedded_xd = self.embedding_xd(xd).permute(0, 2, 1)
        conv_xd = self.conv_module(embedded_xd, self.conv_xd_1, self.conv_xd_2, self.conv_xd_3)

        # protein
        xt = target.x
        embedded_xt = self.embedding_xt(xt).permute(0, 2, 1)
        conv_xt = self.conv_module(embedded_xt, self.conv_xt_1, self.conv_xt_2, self.conv_xt_3)
             
        # A2H graph input feed-forward
        a2h_x, a2h_edge_index, a2h_edge_attr, a2h_batch = a2h.x, a2h.edge_index, a2h.edge_attr, a2h.batch
        a2h_x = self.dropout(a2h_x)
        
        a2h_x = F.elu(self.a2h_gat1(a2h_x, a2h_edge_index, a2h_edge_attr))
        a2h_x = self.dropout(a2h_x)
        a2h_x = self.a2h_gat2(a2h_x, a2h_edge_index, a2h_edge_attr)
        a2h_x = self.relu(a2h_x)
        a2h_x = gmp(a2h_x, a2h_batch)
        a2h_x = self.a2h_fc1(a2h_x)
        a2h_x = self.relu(a2h_x)

        # concat all modalities
        xc = torch.cat((conv_xd, conv_xt, a2h_x), 1)  # (batch_size, 96 * 3)
        
        # add some dense layers
        out = self.classifier(xc)
        return out


########################################################################################################################
########## DeepDTA
########################################################################################################################

class DeepDTA(torch.nn.Module):
    def __init__(self, n_filters=32):
        super(DeepDTA, self).__init__()
        self.relu = nn.ReLU()
        self.n_filters = n_filters

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(CHARISOSMILEN + 1, 128) # batch, 100, 128 -> batch, 128, 100
        self.conv_xd_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=4) # batch, 32, 97
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=4) # batch, 64, 94
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=4) # batch, 96, 91

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128 -> batch, 128, 1000
        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8) # batch, 32, 993
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8) # batch, 64, 986
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=8) # batch, 96, 979

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(n_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def conv_module(self, x, conv1, conv2, conv3):
        x = conv1(x)
        x = F.relu(x)
        x = conv2(x)
        x = F.relu(x)
        x = conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, data):
        drug, target, _ = data
        xd, xt = drug.x, target.x
        
        # drug
        embedded_xd = self.embedding_xd(xd).permute(0, 2, 1)
        conv_xd = self.conv_module(embedded_xd, self.conv_xd_1, self.conv_xd_2, self.conv_xd_3)

        # protein
        embedded_xt = self.embedding_xt(xt).permute(0, 2, 1)
        conv_xt = self.conv_module(embedded_xt, self.conv_xt_1, self.conv_xt_2, self.conv_xt_3)
        
        # dense
        xc = torch.cat((conv_xd, conv_xt), 1)
        xc = self.classifier(xc)
        return xc

########################################################################################################################
########## GraphDTA
########################################################################################################################

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        drug, target, _ = data 
        target = target.x

        # graph input feed-forward
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        # target = data.target
        
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# GCN-CNN based model
class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        drug, target, _ = data
        
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        # target = data.target
        target = target.x
        
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        drug, target, _ = data 

        # get graph input
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        # # get protein input
        # target = data.target
        target = target.x

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        drug, target, _ = data
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        # target = data.target
        target = target.x

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = gap(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

########################################################################################################################
########## DeepDTAF & CAPLA
########################################################################################################################

PT_FEATURE_SIZE = 40

class Squeeze(nn.Module):   #Dimention Module
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class DilatedConv(nn.Module):     # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedConvBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  # Down Dimention
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)    # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)     # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)     # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)     # Dilated scale:8(2^3)
        self.d16 = DilatedConv(n, n, 3, 1, 16)   # Dilated scale:16(2^4)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)

        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output
    
    
class DilatedConvBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)   # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)   # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)   # Dilated scale:8(2^3)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DeepDTAF(nn.Module):

    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHARISOSMILEN + 1, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedConvBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        # (N, H=32, L)
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedConvBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        
        
        self.cat_dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU())
        

    def forward(self, data):
        smi, seq, pkt = data[0].x, data[1].x, data[2].x
        
        # assert seq.shape == (N,L,43)
        seq_embed = self.seq_embed(seq)  # (N,L,32)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,43)
        
        pkt_embed = self.seq_embed(pkt)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        
        output = self.classifier(cat)
        return output


class CAPLA(nn.Module):
    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128

        seq_oc = 128
        pkt_oc = 64
        smi_oc = 128
        td_oc = 32

        # SMILES, POCKET, PROTEIN Embedding
        self.smi_embed = nn.Embedding(CHARISOSMILEN + 1, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)

        # Global DilatedConv Module
        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedConvBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        # Pocket DilatedConv Module
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)
        
        td_conv = []
        ic = 1
        for oc in [16, 32, td_oc * 2]:
            td_conv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        td_conv.append(nn.AdaptiveMaxPool1d(1))
        td_conv.append(Squeeze())
        self.td_conv = nn.Sequential(*td_conv)

        td_onlyconv = []
        ic = 1
        for oc in [16, 32, td_oc]:
            td_onlyconv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        self.td_onlyconv = nn.Sequential(*td_onlyconv)

        # Ligand DilatedConv Module
        conv_smi = []
        ic = smi_embed_size

        # Cross-Attention Module
        self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2)  # 注意力机制
        self.tdpoc_attention_tdlig = EncoderLayer(32, 64, 0.1, 0.1, 1)
        
        self.adaptmaxpool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = Squeeze()

        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedConvBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)

        # Dropout
        self.cat_dropout = nn.Dropout(0.2)
        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + pkt_oc + smi_oc, 256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data):
        smi, seq, pkt = data[0].x, data[1].x, data[2].x

        # D(B_s,N,L)
        seq_embed = self.seq_embed(seq)
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_seq(seq_embed)

        pkt_embed = self.seq_embed(pkt)
        smi_embed = self.smi_embed(smi)
        smi_attention = smi_embed

        smi_embed = self.smi_attention_poc(smi_embed, pkt_embed)
        pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention)

        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)

        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)

        concat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)
        concat = self.cat_dropout(concat)
        output = self.classifier(concat)
        return output
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)

        # temp = x.cpu().numpy()
        # temp = temp.mean(axis=2)

        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    
if __name__ == "__main__":
    # 모델 인스턴스 생성
    # model = DeepDTAF()
    model = CAPLA()

    # 예제 입력 텐서 생성
    batch_size = 2
    seq_len = 100
    feature_dim = 40
    smi_len = 120

    seq_tensor = torch.randn(batch_size, seq_len, feature_dim)  # (N, L, 40)
    pkt_tensor = torch.randn(batch_size, seq_len, feature_dim)  # (N, L, 40)
    smi_tensor = torch.randint(0, 64, (batch_size, smi_len))     # (N, L)
    
    # 모델 실행
    output = model(seq_tensor, pkt_tensor, smi_tensor)

    # 결과 출력
    print("Output shape:", output.shape)
    print("Output values:", output)
    
    
    
    # import torch
    # import torch.nn as nn
    
    # lienar = torch.nn.Linear(40, 128)
    # x = torch.randn(256, 1000, 40)
    # x.shape
    # output = lienar(x)
    # print(output.shape)
