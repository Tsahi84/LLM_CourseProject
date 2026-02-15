import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

class ProteinStructureDataSet(Dataset):
    # Creates a dataset for protein structure
    # data - the protein sequence data to create the dataloader from. A list of dictionaries, where the protein sequence is under the "sequence" key, and the structure is under the "contact_map" key
    def __init__(self, data):
        self.data = data
        self.input_sequences = [entry["sequence"] for entry in data]
        self.target_maps = [entry["contact_map"] for entry in data]

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_maps[idx]

# makes all the data in the same length so it can be fitted into a tensor
def custom_collate_fn(batch, pad_token_id = 21, ignore_index=-100, allowed_max_length=None, device="cpu"):
    sequences = [item[0] for item in batch]
    contact_maps = [item[1] for item in batch]
    batch_max_length = max(len(item)+1 for item in sequences) #find the longest sequence in the batch
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)
    inputs_lst, targets_lst = [], []

    for seq, contact_map in zip(sequences, contact_maps):
        padded_seq = (seq + [pad_token_id] * (batch_max_length - len(seq))) # add the end token to the end of the sequence
        padded_contact_map = torch.ones((batch_max_length, batch_max_length), dtype=torch.long) * ignore_index # create a tensor where all values are the ignore index
        if contact_map.shape[0] > allowed_max_length:
            contact_map = contact_map[:allowed_max_length, :allowed_max_length] #trim the contact map if the sequence is longer than the allowed max length
        padded_contact_map[:contact_map.shape[0], :contact_map.shape[1]] = contact_map  #fill the larger tensor with the original contact map

        inputs = torch.tensor(padded_seq) # convert the padded sequence to a tensor
        targets = padded_contact_map

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device) #converts the inputs to a tensor and transfer it to the device
    targets_tensor = torch.stack(targets_lst).to(device) #converts the targets to a tensor and transfer it to the device
    return inputs_tensor, targets_tensor


# Creates a dataloader for UniRef50Data
# data - the protein sequence data to create the dataloader from. A list of dictionaries, where the protein sequence is under the "sequence" key
# context_length - the maximum sequence length for training
# device - the device to place the data on
# batch_size - the batch size
# shuffle - shuffle the order of the data if True
# drop_last - drop the last batch if its size is smaller than batch_size
# num_workers - the number of processes to use for data loading
def create_protein_structure_dataloader(data, context_length, device="cpu", batch_size=4,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = ProteinStructureDataSet(data)

    customized_collate_fn = partial(custom_collate_fn, device=device,  allowed_max_length=context_length)

    # Create dataloader
    dataloader = DataLoader(dataset, collate_fn= customized_collate_fn, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader