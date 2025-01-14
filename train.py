import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import csv

import mymodel
import dataloader



# Initialize model
model = mymodel.LF_Unet()
model = model.to(torch.device('cuda:0'))
model.load_state_dict(torch.load('./checkpoints/u2net_full.pth'),strict=False)
#

def criterion(inputs,target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i],target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss



train_dataloader = dataloader.train_dataloader
val_dataloader = dataloader.val_dataloader

# Define directory to save checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def load_pretrained_model(model, checkpoint_dir='./checkpoints', filename='best_model_epoch_0.pth'):
    """ Load pretrained model parameters if they exist. """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path),strict=False)
        print(f'Loaded pretrained model from {checkpoint_path}')
    else:
        print(f'No pretrained model found at {checkpoint_path}, starting training from scratch.')

def save_model_parameters(model, checkpoint_dir, filename='model.pth'):
    """ Save only the model parameters (state_dict) to reduce file size. """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)

def save_to_file(epoch, step, train_loss, val_loss=None, filename='./checkpoints/training_log.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Step', 'Training Loss', 'Validation Loss'])
        writer.writerow([epoch, step, train_loss, val_loss])

def train(model, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001, device='cuda'):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load pretrained model if it exists
    load_pretrained_model(model, checkpoint_dir)

    # Define loss function and optimizer
    #criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_dataloader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Save training loss to file
            save_to_file(epoch + 1, i + 1, loss.item())

            # Print loss for every step
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        # Validation after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_dataloader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                cer = nn.BCEWithLogitsLoss()
                loss = cer(outputs,masks)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        # Save validation loss to file
        save_to_file(epoch + 1, None, running_loss / len(train_dataloader), val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {running_loss / len(train_dataloader):.9f}, Validation Loss: {val_loss:.9f}')

        # Save the best model
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            save_model_parameters(model, checkpoint_dir, filename=f'best_model_epoch_{epoch+1}.pth')

    print('Finished Training')

if __name__ == '__main__':
    train(model, train_dataloader, val_dataloader, num_epochs=70, learning_rate=0.00001)