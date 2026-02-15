import torch
from torch.utils.data import Dataset, DataLoader

class UniRef50DataSet(Dataset):
    # Creates a dataset for UniRef50Data
    # data - the protein sequence data to create the dataloader from. A tensor with the shape (num_batches, 2 * context length), the first context_length items in each row are the input, and the rest are the target
    # context_length - the maximum sequence length for training
    def __init__(self, data, context_length):
        self.data = data
        self.input_sequences = data[:,:context_length].int()
        self.target_sequences = data[:,context_length:].to(torch.int64)

    def __len__(self):
        return self.input_sequences.shape[0]

    def __getitem__(self, idx):
        return self.input_sequences[idx, :], self.target_sequences[idx, :]

# Creates a dataloader for UniRef50Data
# data - the protein sequence data to create the dataloader from. A list of dictionaries, where the protein sequence is under the "sequence" key
# tokenizer - the tokenizer that coverts residue strings to numbers
# batch_size - the batch size
# context_length - the maximum sequence length for training
# shuffle - shuffle the order of the data if True
# drop_last - drop the last batch if its size is smaller than batch_size
# num_workers - the number of processes to use for data loading
def create_protein_sequence_dataloader(data, context_length, batch_size=4,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = UniRef50DataSet(data, context_length)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader