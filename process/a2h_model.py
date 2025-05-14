import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class A2HNet(nn.Module):
    def __init__(self, ligand_input_dim, seq_embed_dim, seq_len_example, 
                 a2h_timesteps, a2h_max_atoms, a2h_coord_dim,
                 hidden_dim=128, gnn_layers=3, cnn_out_channels=32, dropout_rate=0.2):
        super(A2HNet, self).__init__()

        # 1. Ligand GNN (GINConv based)
        self.ligand_convs = nn.ModuleList()
        self.ligand_bns = nn.ModuleList()
        self.ligand_convs.append(GINConv(nn.Sequential(nn.Linear(ligand_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        self.ligand_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(gnn_layers - 1):
            self.ligand_convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            self.ligand_bns.append(nn.BatchNorm1d(hidden_dim))
        self.ligand_fc = nn.Linear(hidden_dim, hidden_dim)

        # 2. Protein Sequence CNN (Simplified 1D CNN)
        # Assuming sequence input is (B, 1, L), where L is sequence length
        # If protein_seq_to_vec gives embeddings directly, this might change to an MLP or Transformer encoder
        # For now, using a simple CNN architecture inspired by DeepDTA/GnS
        # Actual sequence length can vary, so AdaptiveMaxPool1d or careful padding in collate_fn might be needed if not fixed.
        # Using a fixed example length for MLP input size calculation. This is a simplification.
        self.seq_len_example = seq_len_example # e.g., 1000 from GnS example
        self.protein_embed_dim = seq_embed_dim # Assuming input is (Batch, 1, SeqLen) - or (Batch, SeqLen) then unsqueeze for Conv1d
                                            # Or if it's (Batch, SeqLen, FeatureDim) then in_channels=FeatureDim
                                            # Based on (1, protein_length) from prepare.py, let's assume it's (B,1,L)
        self.protein_conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=8)
        self.protein_pool1 = nn.AdaptiveMaxPool1d(1) # Pool to a fixed size
        # Calculate flattened size after conv and pool for the example length
        # This is a common simplification. Real adaptive pooling output is (B, cnn_out_channels, 1)
        # So, after squeeze, it's (B, cnn_out_channels)
        self.protein_fc = nn.Linear(cnn_out_channels, hidden_dim)

        # 3. A2H Processor (LSTM)
        self.a2h_lstm = nn.LSTM(input_size=a2h_coord_dim, hidden_size=hidden_dim, batch_first=True)
        self.a2h_fc = nn.Linear(hidden_dim, hidden_dim)

        # 4. Combined Feature Processor and Regressor
        self.combined_fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2) # ligand_hid + protein_hid + a2h_hid
        self.combined_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1) # Predict affinity

        # Dropout and ReLU
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        ligand, protein_seq, a2h = data
        # ligand (batch, atom_num, 55)
        # protein_seq (batch, protein_length)
        # a2h (batch, timesteps, max_atom_num, 3)

        # 1. Process Ligand
        x_ligand, edge_index_ligand, batch_ligand = ligand_data.x, ligand_data.edge_index, ligand_data.batch
        for conv, bn in zip(self.ligand_convs, self.ligand_bns):
            x_ligand = self.relu(bn(conv(x_ligand, edge_index_ligand)))
        x_ligand = global_add_pool(x_ligand, batch_ligand)
        x_ligand = self.relu(self.ligand_fc(x_ligand))
        x_ligand = self.dropout(x_ligand)

        # 2. Process Protein Sequence
        x_protein = self.protein_conv1(protein_seq)
        x_protein = self.relu(x_protein)
        x_protein = self.protein_pool1(x_protein)
        x_protein = x_protein.squeeze(-1) # (B, cnn_out_channels)
        x_protein = self.relu(self.protein_fc(x_protein))
        x_protein = self.dropout(x_protein)

        # 3. Process A2H data
        batch_size = a2h_data.size(0)
        x_a2h = a2h_data.view(batch_size, -1) # Flatten
        x_a2h = self.relu(self.a2h_fc1(x_a2h))
        x_a2h = self.dropout(x_a2h)
        x_a2h = self.relu(self.a2h_fc2(x_a2h))
        x_a2h = self.dropout(x_a2h)

        # 4. Combine and Predict
        combined_features = torch.cat((x_ligand, x_protein, x_a2h), dim=1)
        
        x_combined = self.relu(self.combined_fc1(combined_features))
        x_combined = self.dropout(x_combined)
        x_combined = self.relu(self.combined_fc2(x_combined))
        x_combined = self.dropout(x_combined)
        
        output = self.output_fc(x_combined)
        return output

# Example instantiation (for checking dimensions, typically done in train.py):
if __name__ == '__main__':
    # These are example dimensions and should come from args in a real script
    LIGAND_INPUT_DIM = 55         # Node feature dim for ligands
    SEQ_EMBED_DIM = 1             # Channel dim for sequence (e.g., 1 if (B,1,L))
    SEQ_LEN_EXAMPLE = 1000        # Example sequence length for model definition 
    A2H_TIMESTEPS = 10
    A2H_MAX_ATOMS = 290 # Example from GnS paper for protein, check your actual a2h data
    A2H_COORD_DIM = 3
    
    HIDDEN_DIM = 128
    
    model = A2HNet(
        ligand_input_dim=LIGAND_INPUT_DIM,
        seq_embed_dim=SEQ_EMBED_DIM,
        seq_len_example=SEQ_LEN_EXAMPLE, 
        a2h_timesteps=A2H_TIMESTEPS,
        a2h_max_atoms=A2H_MAX_ATOMS,
        a2h_coord_dim=A2H_COORD_DIM,
        hidden_dim=HIDDEN_DIM
    )
    print(f"A2HNet model instantiated. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Dummy data for forward pass test
    # Ligand (PyG Batch)
    from torch_geometric.data import Data
    dummy_ligand1 = Data(x=torch.randn(10, LIGAND_INPUT_DIM), edge_index=torch.randint(0, 10, (2, 20)))
    dummy_ligand2 = Data(x=torch.randn(15, LIGAND_INPUT_DIM), edge_index=torch.randint(0, 15, (2, 30)))
    ligand_batch = Batch.from_data_list([dummy_ligand1, dummy_ligand2])
    
    # Protein Sequence (Batch, 1, SeqLen)
    protein_seq_batch = torch.randn(2, 1, SEQ_LEN_EXAMPLE)
    
    # A2H Data (Batch, Timesteps, MaxAtoms, Coords)
    a2h_batch_data = torch.randn(2, A2H_TIMESTEPS, A2H_MAX_ATOMS, A2H_COORD_DIM)
    
    dummy_model_input = (ligand_batch, protein_seq_batch, a2h_batch_data)
    
    try:
        output = model(dummy_model_input)
        print(f"Dummy forward pass successful. Output shape: {output.shape}") # Expected: (2,1)
    except Exception as e:
        print(f"Error during dummy forward pass: {e}") 