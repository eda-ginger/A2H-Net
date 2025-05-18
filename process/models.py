########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/hkmztrk/DeepDTA
# https://github.com/thinng/GraphDTA
# https://github.com/peizhenbai/DrugBAN
# https://github.com/595693085/DGraphDTA/tree/master
# https://github.com/JK-Liu7/AttentionMGT-DTA/tree/main
# https://github.com/zhaoqichang/AttentionDTA_TCBB/tree/main

########################################################################################################################
########## Import
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GINConv, global_add_pool as gap
from torch_geometric.nn import GCNConv, global_mean_pool as gep


########################################################################################################################
########## A2H-Net
########################################################################################################################

# class GnS(torch.nn.Module):
#     def __init__(self, n_output=1, num_features_xd=55, num_features_xt=25, n_filters=32, embed_dim=128,
#                  output_dim=128, dropout=0.2, joint='concat'):

#         super(GnS, self).__init__()

#         dim = 32
#         jdim = 256
#         self.joint = joint
#         if self.joint in ['add', 'multiple']:
#             jdim = 128

#         # GIN layers (drug)
#         nn1_xd = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
#         self.conv1_xd = GINConv(nn1_xd)
#         self.bn1_xd = torch.nn.BatchNorm1d(dim)

#         nn2_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv2_xd = GINConv(nn2_xd)
#         self.bn2_xd = torch.nn.BatchNorm1d(dim)

#         nn3_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv3_xd = GINConv(nn3_xd)
#         self.bn3_xd = torch.nn.BatchNorm1d(dim)

#         nn4_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv4_xd = GINConv(nn4_xd)
#         self.bn4_xd = torch.nn.BatchNorm1d(dim)

#         # 1D convolution on protein sequence
#         self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
#         self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)  # batch, 32, 121

#         if 'att' in self.joint:
#             nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 121))
#             self.conv5_xd = GINConv(nn5_xd) # batch, node, 121
#             self.bn5_xd = torch.nn.BatchNorm1d(121)

#             if self.joint == 'bi_att':
#                 self.jc = weight_norm(BANLayer(v_dim=121, q_dim=121, h_dim=jdim, h_out=2), name='h_mat', dim=None)
#             elif self.joint == 'cross_att':
#                 self.jc = CrossAttention(290, n_filters, embed_dim=121, out_dim=jdim, num_heads=8)
#             elif self.joint == 'co_att':
#                 self.jc = CoAttention(290, n_filters, embed_dim=121, out_dim=jdim, num_heads=8)
#             else:
#                 raise Exception('wrong att type')
#         else:
#             nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#             self.conv5_xd = GINConv(nn5_xd)
#             self.bn5_xd = torch.nn.BatchNorm1d(dim)

#             self.fc1_xd = Linear(dim, output_dim)
#             self.fc1_xt = nn.Linear(32 * 121, output_dim)

#             if self.joint in ['concat', 'add', 'multiple']:
#                 self.jc = Simple_Joint(self.joint)
#             elif self.joint == 'bi':
#                 self.jc = Bilinear_Joint(output_dim, jdim)
#             else:
#                 raise Exception(f'{self.joint} method not supported!!!')

#         # dense
#         self.classifier = nn.Sequential(
#             nn.Linear(jdim, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, n_output),
#         )

#     def forward(self, data):
#         drug, target, y = data
#         xd, xd_ei, xd_batch = drug.x, drug.edge_index, drug.batch
#         xt = target.x

#         # drug
#         xd = F.relu(self.conv1_xd(xd, xd_ei))
#         xd = self.bn1_xd(xd)
#         xd = F.relu(self.conv2_xd(xd, xd_ei))
#         xd = self.bn2_xd(xd)
#         xd = F.relu(self.conv3_xd(xd, xd_ei))
#         xd = self.bn3_xd(xd)
#         xd = F.relu(self.conv4_xd(xd, xd_ei))
#         xd = self.bn4_xd(xd)
#         xd = F.relu(self.conv5_xd(xd, xd_ei))
#         xd = self.bn5_xd(xd)

#         embedded_xt = self.embedding_xt(xt)
#         conv_xt = self.conv_xt_1(embedded_xt)

#         # joint
#         if 'att' in self.joint:
#             xd = xd.view(len(y), 290, 121)
#             xj = self.jc(xd, conv_xt)

#         else:
#             # flatten
#             xd = gap(xd, xd_batch)
#             xd = F.relu(self.fc1_xd(xd))
#             xd = F.dropout(xd, p=0.2, training=self.training)
#             xt = self.fc1_xt(conv_xt.view(-1, 32 * 121))
#             xj = self.jc(xd, xt)

#         # dense
#         out = self.classifier(xj).squeeze(1)
#         return out, y

########################################################################################################################
########## GraphDTA
########################################################################################################################

from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
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
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
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
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
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
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

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
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
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
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
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
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
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
