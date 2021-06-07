# Utility functions
import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset. Used to compute values for
    normalisation of the channels of the input images
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class EvalPerformance:
    def __init__(self, model, device, test_loader, classes):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.classes = classes

        self.preds = []
        self.correct_pred = []
        self.targets = []
        self.pred_class = []
        self.true_class = []
        self.output = []

    def eval_performance(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                # Load test dataset to GPU
                data, target = data.to(self.device), target.to(self.device)
                # Forward pass to make predictions
                output = self.model(data)
                # Get the index of the most likely prediction
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum().item()

                self.preds.append(pred)
                self.correct_pred.append(correct)
                self.targets.append(target.view_as(pred))
                self.pred_class.append(self.classes[pred[0].item()])
                self.true_class.append(self.classes[target[0]])
                self.output.append(output)
