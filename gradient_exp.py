from typing import List
import os

import torch
import torchvision
import torchvision.transforms as transforms

from captum.attr._core.saliency import Saliency
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr import visualization as viz

import matplotlib.pyplot as plt

import numpy as np

from sklearn.neighbors import NearestNeighbors

from utils import UnNormalize, EvalPerformance
from cifar_train import Net, img_means, img_stds, classes


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def top_n_confidence_wrong(correct: List[int], output: List[torch.tensor], n=10):
    """
    Calculate top n most confident wrong outputs
    Parameters:
        - Correct: List of binary correct/wrong prediction
        - output: List of softmax output from model
    """

    incorrect_indices = [idx for idx, x in enumerate(correct) if x == 0]

    incorrect_outs = [output[idx] for idx in incorrect_indices]
    incorrect_outs = np.array([t.cpu().numpy().max() for t in incorrect_outs])

    # Getting index of top 10 most confident wrong predictions
    top_n_idx = np.argpartition(incorrect_outs, -n)[-n:]
    # mapping back to original indices
    top_n_idx = np.array(incorrect_indices)[top_n_idx]

    return top_n_idx


def find_nearest_neighbor(neighbor_model, image):
    """Input "trained" nearest neighbor model and outputs index
    of nearest neighbor (that is not itself)"""
    dist, idxs = neighbor_model.kneighbors(image.reshape(1, -1), 2)
    return idxs[0][1]


def flatten_images(dataset):
    """Flatten images to length 1"""
    images = []
    labs = np.zeros(len(dataset))

    for i in range(len(dataset)):
        images.append(dataset[i][0].numpy().reshape(-1))
        labs[i] = dataset[i][1]

    images = np.array(images)
    return images, labs


def remove_axis_ticks(axis):
    axis.xaxis.set_ticks_position("none")
    axis.yaxis.set_ticks_position("none")
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    axis.grid(b=False)


def show_pair(pair, labels, pred, savename=None):
    """Plot a pair of images side by side"""
    unorm = UnNormalize(img_means, img_stds)
    pair = [unorm(img).numpy() for img in pair]  # unnormalize and transform to numpy
    labels = [classes[l] for l in labels]  # get text label

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.transpose(pair[0], (1, 2, 0)))
    ax1.set_title(f"true: {labels[0]}, pred: {classes[pred]}")
    remove_axis_ticks(ax1)

    ax2.imshow(np.transpose(pair[1], (1, 2, 0)))
    ax2.set_title(f"nn: {labels[1]}")
    remove_axis_ticks(ax2)

    if savename is not None:
        plt.savefig(savename + ".png")

    return fig, (ax1, ax2)


def explainer(algorithm, input, target, **kwargs):
    """Applies Captum algorithm and returns calculated gradients"""
    net.zero_grad()
    # Get tensor attributions
    grads = algorithm.attribute(input, target=target, **kwargs)
    # reshape for vis
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return grads


def explain_pair(alg, pair: List[torch.tensor], labels: List[int], **kwargs):
    """
    Use a Captum explanation algorithm on a pair of images and plot side-by-side

    Parameters:
        alg: Captum algorithm, e.g. Saliency()
        pair: list of 2 images as torch tensors
        labels: the labels for each image
        **kwargs: additional arguments for Captum algorithm
    """

    def _prepare_explainer_input(img):
        input = img.unsqueeze(0)
        input.requires_grad = True
        input = input.cuda()
        return input

    inputs = [_prepare_explainer_input(img) for img in pair]
    grads = [explainer(alg, inp, lab, **kwargs) for inp, lab in zip(inputs, labels)]

    unorm = UnNormalize(img_means, img_stds)
    org_images = [unorm(img) for img in pair]
    org_images = [
        np.transpose(org_img.cpu().detach().numpy(), (1, 2, 0))
        for org_img in org_images
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    _ = viz.visualize_image_attr(
        grads[0],
        org_images[0],
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Predicted",
        plt_fig_axis=(fig, ax1),
        # use_pyplot to false to  avoid viz calling plt.show()
        use_pyplot=False,
    )
    _ = viz.visualize_image_attr(
        grads[1],
        org_images[1],
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Nearest neighbor",
        plt_fig_axis=(fig, ax2),
    )

    return fig, (ax1, ax2)


def create_pairs(indices, nearest_neighbors):
    """Create pairs of predicted image and their nearest neighbor"""
    pairs = []
    labels = []
    for test_idx, train_idx in zip(indices, nearest_neighbors):
        # 0 index to get tensor of image
        pairs.append([testset[test_idx][0], trainset[train_idx][0]])
        # 1 index to get label
        labels.append([testset[test_idx][1], trainset[train_idx][1]])
    return pairs, labels


def plot_original_and_explained_pair(
    pair: List[torch.tensor],
    labels: List[int],
    alg,
    pred: int,
    savename: str = None,
    method: str = "blended_heat_map",
    sign: str = "absolute_value",
):
    """
    Plot 2x2 grid of images. First row shows original images, second the gradient explanations.

    Args:
        pair (List[torch.tensor]): List of 2 images as torch tensors
        labels (List[int]): the true labels for the images
        alg ([type]): a Captum algorithm
        pred (int): the prediction for the original image
        savename (str, optional): If given, saves the image to disk. Defaults to None.
        method (str, optional): which visualization method to use (heat_map, blended_heat_map, original_image, masked_image, alpha_scaling)
        sign (str, optional): sign of attributions to visualiuze (positive, absolute_value, negative, all)
    """

    def _prepare_explainer_input(img):
        input = img.unsqueeze(0)
        input.requires_grad = True
        input = input.cuda()
        return input

    inputs = [_prepare_explainer_input(img) for img in pair]
    # Explaining the target
    grads_target = [explainer(alg, inp, lab) for inp, lab in zip(inputs, labels)]
    # Explaining the actual prediction
    grads_pred = [explainer(alg, inp, pred) for inp in inputs]

    unorm = UnNormalize(img_means, img_stds)
    org_images = [unorm(img) for img in pair]
    org_images = [
        np.transpose(org_img.cpu().detach().numpy(), (1, 2, 0))
        for org_img in org_images
    ]

    text_labels = [classes[l] for l in labels]  # get text label

    fig, axes = plt.subplots(2, 3)
    # plt.subplots_adjust(wspace=0.0001)
    ### Plot original images
    # Wrongly predicted
    _ = viz.visualize_image_attr(
        grads_target[0],
        org_images[0],
        method="original_image",
        title=f"true: {text_labels[0]}, pred: {classes[pred]}",
        plt_fig_axis=(fig, axes[0, 0]),
        use_pyplot=False,
    )

    # Nearest neighbor
    _ = viz.visualize_image_attr(
        grads_target[1],
        org_images[1],
        method="original_image",
        title=f"nn: {text_labels[1]}",
        plt_fig_axis=(fig, axes[1, 0]),
        use_pyplot=False,
    )

    ### Gradient explanations for predicted
    _ = viz.visualize_image_attr(
        grads_pred[0],
        org_images[0],
        method=method,
        sign=sign,  # org: "absolute_value"
        show_colorbar=True,
        title=f"Exp. wrt. {classes[pred]}",
        plt_fig_axis=(fig, axes[0, 1]),
        # use_pyplot to false to  avoid viz calling plt.show()
        use_pyplot=False,
    )
    _ = viz.visualize_image_attr(
        grads_pred[1],
        org_images[1],
        method=method,
        sign=sign,
        show_colorbar=True,
        title="",
        plt_fig_axis=(fig, axes[1, 1]),
        use_pyplot=True,
    )
    ### Gradient explanations for target
    _ = viz.visualize_image_attr(
        grads_target[0],
        org_images[0],
        method=method,
        sign=sign,  # org: "absolute_value"
        show_colorbar=True,
        title=f"Exp. wrt. {text_labels[0]}",
        plt_fig_axis=(fig, axes[0, 2]),
        # use_pyplot to false to  avoid viz calling plt.show()
        use_pyplot=False,
    )
    _ = viz.visualize_image_attr(
        grads_target[1],
        org_images[1],
        method=method,
        sign=sign,
        show_colorbar=True,
        title="",
        plt_fig_axis=(fig, axes[1, 2]),
        use_pyplot=False,
    )

    if savename is not None:
        plt.savefig(savename + ".png")

    plt.close()
    return fig, axes


if __name__ == "__main__":

    #######
    # Load model and images
    #######
    net = Net()
    net.load_state_dict(torch.load("model.h5"))

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(img_means, img_stds),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2
    )

    trainset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_test
    )

    ############
    # Calculate top 10 most confident predictions
    # Find nearest neighbors
    ############
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    perf = EvalPerformance(net, device, testloader, classes)
    perf.eval_performance()

    top_10_idx = top_n_confidence_wrong(perf.correct_pred, perf.output, n=10)

    neighbor = NearestNeighbors(n_neighbors=2)

    # Flatten channels and pixels to feed to nearest neighbor classifier
    train_flat, train_labs = flatten_images(trainset)
    test_flat, test_labs = flatten_images(testset)

    neighbor.fit(train_flat)

    nearest_neighbors = np.zeros(10)

    for i, img_idx in enumerate(top_10_idx):
        nearest_neighbors[i] = find_nearest_neighbor(neighbor, test_flat[img_idx])

    nearest_neighbors = nearest_neighbors.astype(int)

    ## Plot only nearest neighbors
    pairs, labels = create_pairs(top_10_idx, nearest_neighbors)
    create_folder("imgs/nn")
    for i in range(len(pairs)):
        show_pair(pairs[i], labels[i], perf.preds[top_10_idx[i]].item(), f"imgs/nn/{i}")

    ####
    # Explaining and plotting
    ####

    algos = {
        "saliency": Saliency(net),
        "gradcam": GuidedGradCam(net, net.layer3),
        "integrated_gradients": IntegratedGradients(net),
    }
    methods = ["heat_map", "masked_image", "alpha_scaling", "blended_heat_map"]
    signs = [
        "positive",
        "all",
        "absolute_value",
    ]  # all does not work with masked_image and alpha_scaling

    net.eval()
    for alg_name, alg in algos.items():
        create_folder("imgs")
        create_folder(f"imgs/{alg_name}")
        for sign in signs:
            create_folder(f"imgs/{alg_name}/{sign}")
            for method in methods:
                pairs, labels = create_pairs(top_10_idx, nearest_neighbors)

                for i in range(len(pairs)):
                    try:
                        fig, axes = plot_original_and_explained_pair(
                            pairs[i],
                            labels[i],
                            alg,
                            perf.preds[top_10_idx[i]].item(),
                            f"imgs/{alg_name}/{sign}/{method}_{i}",
                            method,
                            sign,
                        )
                    except AssertionError:
                        # If sign="all" not all methods work - skipping them
                        continue
