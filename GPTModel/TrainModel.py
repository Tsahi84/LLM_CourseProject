import torch
from .GenerateText import generate_and_print_sample

# Trains a model
# model - the model to train
# train_loader - the training dataloader
# val_loader - the validating dataloader
# optimizer - the optimizer to use in the training
# device - the device to perform the computations on (CPU/GPU)
# num_epochs - the number of epochs
# eval_freq - the frequency at which to measure the losses
# eval_iter - how many batches are included in the evaluation
# start_context - the input sequence to use for evaluating the model
# tokenizer - the tokenizer to use for tokenizing the text
# filename_prefix - the file name to use for saving model checkpoints
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, filename_prefix):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device) # Calculate the loss for the current batch
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                print(f"Ep {epoch + 1} (Step {global_step:06d}): ")

        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch + 1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # Print a sample text after each epoch
        if start_context is not None:
            generate_and_print_sample(model, tokenizer, device, start_context)
        file_name = "TrainedModels/" + filename_prefix + "_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), file_name)

    return train_losses, val_losses, track_tokens_seen

# Calculate the model loss for the training and validation sets
# model - the model to train
# train_loader - the training dataloader
# val_loader - the validating dataloader
# device - the device to perform the computations on (CPU/GPU)
# eval_iter - how many batches are included in the evaluation
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Set evaluation mode for the model
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = -1.0
        if val_loader is not None:
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train() # Set training mode for the model
    return train_loss, val_loss

# calculate the loss for a batch
# input_batch - the input batch, shape - (batch_size, num_tokens, embedding_dim)
# target_batch - the target batch, shape - (batch_size, num_tokens, embedding_dim)
# model - the model
# device - the device to perform the computations on (CPU/GPU)
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) # move the batches to the device
    logits = model(input_batch) # calculate the output logits of the model
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), target_batch.flatten()) # calculate the loss between the model output and the ground truth
    return loss

# Calculate the loss for a loader
# data_loader - the dataloader
# model - the model
# device - the device to perform the computations on (CPU/GPU)
# num_batches - how many batches are included in the evaluation, None for the entire data
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # Return the average loss per batch