########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/hkmztrk/DeepDTA/tree/master

########################################################################################################################
########## Import
########################################################################################################################

import torch
import logging
import numpy as np
from torch_geometric.data import Data
logger = logging.getLogger(__name__)

########################################################################################################################
########## Define dictionaries
########################################################################################################################

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


########################################################################################################################
########## Function
########################################################################################################################


def integer_label_encoding(sequence, tp):
    """
    Integer encoding for string sequence.
    Args:
        sequence (str): Drug or Protein string sequence.
        max_length: Maximum encoding length of input string.
    """
    if tp == 'drug':
        max_length = 100
        charset = CHARISOSMISET
    elif tp == 'protein':
        max_length = 1200
        charset = CHARPROTSET

    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            if tp == 'protein':
                letter = letter.upper()
            letter = str(letter)
            encoding[idx] = charset[letter]
        except KeyError:
            logger.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return Data(x=torch.from_numpy(encoding).to(torch.long).unsqueeze(dim=0))


def protein_seq_to_vec(file):
    seq = read_fasta(file)
    return integer_label_encoding(seq, 'protein')



def read_fasta(file_path: str) -> str:
    """
    Read fasta file and Select chain_A(>=50 length) or Longest chain
    
    Args:
        file_path (str): .fasta file path
        
    Returns:
        str: chain sequence
    """
    sequences = {}
    current_id = None
    current_sequence = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_sequence)
                current_id = line[1:].split()[0]
                current_sequence = []
            else:
                current_sequence.append(line)
    
    if current_id:
        sequences[current_id] = ''.join(current_sequence)
    
    # search chain_A    
    chain_a = next((seq for id, seq in sequences.items() if id.endswith('_A')), None)
    longest_chain = max(sequences.items(), key=lambda x: len(x[1]))
    
    if chain_a and len(chain_a) >= 50:
        return chain_a
    else:
        longest_chain = max(sequences.items(), key=lambda x: len(x[1]))
        return longest_chain[1]
    

if __name__ == '__main__':
    dr = 'COc1cc2c(cc1Cl)C(c1ccc(Cl)c(Cl)c1)=NCC2'
    pr = 'MSWSPSLTTQTCGAWEMKERLGTGGFGNVIRWHNQETGEQIAIKQCRQELSPRNRERWCLEIQIMRRLTHPNVVAARDVPEGMQNLAPNDLPLLAMEYCQGGDLRKYLNQFENCCGLREGAILTLLSDIASALRYLHENRIIHRDLKPENIVLQQGEQRLIHKIIDLGYAKELDQGSLCTSFVGTLQYLAPELLEQQKYTVTVDYWSFGTLAFECITGFRPFLPNWQPVQWHSKVRQKSEVDIVVSEDLNGTVKFSSSLPYPNNLNSVLAERLEKWLQLMLMWHPRQRGTDPTYGPNGCFKALDDILNLKLVHILNMVTGTIHTYPVTEDESLQSLKARIQQDTGIPEEDQELLQEAGLALIPDKPATQCISDGKLNEGHTLDMDLVFLFDNSKITYETQISPRPQPESVSCILQEPKRNLAFFQLRKVWGQVWHSIQTLKEDCNRLQQGQRAAMMNLLRNNSCLSKMKNSMASMSQQLKAKLDFFKTSIQIDLEKYSEQTEFGITSDKLLLAWREMEQAVELCGRENEVKLLVERMMALQTDIVDLQRSPMGRKQGGTLDDLEEQARELYRRLREKPRDQRTEGDSQEMVRLLLQAIQSFEKKVRVIYTQLSKTVVCKQKALELLPKVEEVVSLMNEDEKTVVRLQEKRQKELWNLLKIACSKVRGPVSGSPDSMNASRLSQPGQLMSQPSTASNSLPEPAKKSEELVAEAHNLCTLLENAIQDTVREQDQSFTALDWSWLQTEEEEHSCLEQAS'

    drug_seq = integer_label_encoding(dr, 'drug').x
    prot_seq = integer_label_encoding(pr, 'protein').x
    
    print(drug_seq, drug_seq.shape, drug_seq.dtype)
    print(prot_seq, prot_seq.shape, prot_seq.dtype)

    
    from torch import nn
    # Wrong (GraphDTA) - Not use permute
    # e2 = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128
    # c2_1 = nn.Conv1d(in_channels=1000, out_channels=500, kernel_size=9)
    # c2_2 = nn.Conv1d(in_channels=500, out_channels=250, kernel_size=3)
    # c2_3 = nn.Conv1d(in_channels=250, out_channels=100, kernel_size=3)
    # c2_p = nn.AdaptiveMaxPool1d(1) # batch, 96, 1
    # fc2_xt = nn.Linear(100, 128)

    # TRUE    
    e2 = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128 (batch, seq_len, emb_dim)
    c2_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5) # batch, 256, 996
    c2_2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5) # batch, 128, 992
    c2_3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5) # batch, 64, 988
    c2_p = nn.AdaptiveMaxPool1d(1) # batch, 64, 1
    fc2_xt = nn.Linear(64, 128) # batch, 128    
   
    s2 = e2(prot_seq) # batch, 1000 -> batch, 1000, 128 (batch, seq_len, emb_dim)
    s2 = s2.permute(0, 2, 1) # batch, 128, 1000 (batch, emb_dim, seq_len)
    print(s2.shape)

    s2 = c2_1(s2)
    print(s2.shape)

    s2 = c2_2(s2)
    print(s2.shape)

    s2 = c2_3(s2)
    print(s2.shape)

    s2 = c2_p(s2)
    print(s2.shape)

    s2 = s2.view(-1, s2.shape[-2])
    print(s2.shape)

    s2 = fc2_xt(s2)
    print(s2.shape) # batch, 122 -> batch, 128
