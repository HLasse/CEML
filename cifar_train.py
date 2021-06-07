import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

import wandb

from utils import get_mean_and_std, set_seeds

import numpy as np

## Based on https://github.com/kuangliu/pytorch-cifar and various pytorch tutorials


########################
# Calculate mean and std of the channels to find parameters
# for Normalize
# Only run once to get the parameters
########################


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      ])

# train = torchvision.datasets.CIFAR10(root='data', train=True,
#                                         download=True, transform=transform)

# means, stds = get_mean_and_std(train)
# means
# tensor([0.4914, 0.4822, 0.4465])

# stds
# tensor([0.2023, 0.1994, 0.2010])

########################
# Load datasets and transform
########################

print("[INFO] Loading and preparing data...")

img_means = (0.4914, 0.4822, 0.4465)
img_stds = (0.2023, 0.1994, 0.2010)

BATCH_SIZE = 128
device = "cuda:0" if torch.cuda.is_available() else "cpu"


transform_train = transforms.Compose(
    [
        # Images are 32x32, so desired output size is 32
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalizing with the previously calculated mean and stds
        transforms.Normalize(img_means, img_stds),
    ]
)

# Not augmenting the test set but performing normalization with the same values
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(img_means, img_stds),
    ]
)


trainset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform_train
)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

########################
# Define model
# Pretty basic CNN with 3 fully connected layers in the end
########################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            # kernel size = 3
            nn.MaxPool2d(3),
            # True = inplace
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3), nn.ReLU(True))

        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 128),
            nn.ReLU(True),
            nn.Linear(128, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Flatten dimensions to feed to fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    """Training loop"""
    # Set model to training mode
    model.train()

    # Start training loop over the data loader
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU
        data, target = data.to(device), target.to(device)
        # Set gradients to 0
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Compute loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()


def test(args, model, device, test_loader, criterion, classes):
    """
    Evaluation loop. Log performance to Weight and Biases
    """
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load test dataset to GPU
            data, target = data.to(device), target.to(device)
            # Forward pass to make predictions
            output = model(data)
            # Compute loss
            test_loss += criterion(output, target).item()
            # Get the index of the most likely prediction
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Log example image to WandB
            example_images.append(
                wandb.Image(
                    data[0],
                    caption="Pred {} Truth: {}".format(
                        classes[pred[0].item()], classes[target[0]]
                    ),
                )
            )
        # Log example images, accuracy, and loss to WandB
        wandb.log(
            {
                "Examples": example_images,
                "Test Accuracy": 100.0 * correct / len(test_loader.dataset),
                "Test Loss": test_loss,
            }
        )


######################
## Prepare to train
##
#####################

# https://colab.research.google.com/drive/1XDtq-KT0GkX06a_g1MevuLFOMk4TxjKZ#scrollTo=axH9Vd7igjO3


def main():
    set_seeds(42)

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, criterion, epoch)
        test(config, model, device, test_loader, criterion, classes)
        scheduler.step()

    # Save model checkpoint to file and wandb
    torch.save(model.state_dict(), "model.h5")
    wandb.save("model.h5")


if __name__ == "__main__":
    wandb.init(project="ceml", entity="hlasse")

    config = wandb.config
    config.lr = 0.01  # default 0.01
    config.epochs = 100
    config.momentum = 0.9  # default 0
    config.weight_decay = 5e-4  # default 0, try 5e-4

    main()
