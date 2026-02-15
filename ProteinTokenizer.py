end_token = "<|endofsequence|>"

aa_to_int = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X' : 21, end_token : 22
}

int_to_aa = {
    1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L',
    11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R',
    16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y',
    21: 'X', 22: end_token
}


# THe protein tokenizer - encodes amino acid symbols to numbers, and back
class ProteinTokenizer:
    def __init__(self):
        pass
    #converts a string where each letter represents an amino acid to a list of numbers that represent the amino acids
    def encode(self, sequence):
        try:
            return [aa_to_int[res]-1 for res in sequence]
        except Exception as e:
            print(e)
            return None
    # decodes a list of tokens to the letters that represent the corresponding amino acids
    def decode(self, sequence):
        try:
            return "".join([int_to_aa[res+1] for res in sequence])
        except Exception as e:
            print(e)
            return None