"""pthelloworld.py - script that acts as a hello world to PyTorch

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from models import ptmodel

# load and prepare toy dataset (MNIST)
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# define dataloaders for the datasets
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# import a test model from the models subdirectory
model = ptmodel.BasicConv2DModel()

# send the model to the GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# define training function
def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss, current = loss.item(), batch * len(X)
    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# define testing function
def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# run the training a testing loop
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}, ')
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
