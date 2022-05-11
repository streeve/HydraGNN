##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import json, os
from functools import singledispatch
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from hydragnn import run_training
from hydragnn.preprocess import dataset_loading_and_splitting
from hydragnn.models import create_model_config
from hydragnn.train import train_validate_test
from hydragnn.utils import (
    setup_ddp,
    get_comm_size_and_rank,
    get_distributed_model,
    update_config,
    get_log_name_config,
    load_existing_model,
    iterate_tqdm,
    get_summary_writer,
    save_model,
)

from pi3nn.Optimizations import CL_boundary_optimizer


def run_uncertainty(
    config_file_mean,
    path,
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    retrain_mean=False,
    retrain_up_down=False,
):
    """
    Compute prediction intervals with PI3NN.
    """

    out_name = "uq_"
    mean_name = out_name + "mean"

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    world_size, world_rank = setup_ddp()

    #### LOAD THE ORIGINAL DATA LOADERS AND THE TRAINED MEAN MODEL
    mean_loaders = [train_loader, val_loader, test_loader]

    with open(config_file_mean, "r") as f:
        config_mean = json.load(f)
    config_mean = update_config(
        config_mean, mean_loaders[0], mean_loaders[1], mean_loaders[2]
    )
    config_mean["NeuralNetwork"]["Variables_of_interest"]["input_node_features"] = list(
        range(94)
    )  # one hot
    if retrain_mean:
        run_training(
            config_mean, train_loader, val_loader, test_loader, sampler_list, mean_name
        )
    # mean_model = load_model(config_mean, mean_loaders[0].dataset, mean_name+"_best", path+"/"+mean_name)

    config_file_up_down = os.path.join(path, mean_name, "config.json")
    with open(config_file_up_down, "r") as f:
        config = json.load(f)
    mean_model = load_model(
        config, mean_loaders[0].dataset, mean_name + "_best", path + "/" + mean_name
    )

    #### CREATE THE DATASET LOADERS
    up_loaders, down_loaders, up_sampler, down_sampler = create_loaders(
        mean_loaders, mean_model, config
    )

    #### LOAD OR TRAIN UP & DOWN MODELS
    up_name = out_name + "up"
    down_name = out_name + "down"
    if retrain_up_down:
        # config["NeuralNetwork"]["Architecture"]["hidden_dim"] = 10
        config["NeuralNetwork"]["Architecture"]["freeze_conv_layers"] = True
        config["NeuralNetwork"]["Architecture"]["set_large_bias"] = True
        config["NeuralNetwork"]["Training"]["num_epoch"] = 25
        model_up = train_model(up_loaders, sampler_list, up_name, config)
        # save_model(model_up, up_name, "logs/"+up_name)
        model_down = train_model(down_loaders, sampler_list, down_name, config)
        # save_model(model_down, down_name, "logs/"+down_name)
    else:
        model_up = load_model(
            config, up_loaders[0].dataset, up_name + "_best", path + "/" + up_name
        )
        model_down = load_model(
            config, down_loaders[0].dataset, down_name + "_best", path + "/" + down_name
        )

    #### COMPUTE ALL 3 PREDICTIONS ON TRAINING DATA
    (
        pred_mean_train,
        pred_up_train,
        pred_down_train,
        y_train,
        comp_train,
    ) = compute_predictions(mean_loaders[0], (mean_model, model_up, model_down), config)

    #### COMPUTE PREDICTION INTERVAL BOUNDS
    num_train_mean = len(mean_loaders[0].dataset)
    num_train_up = len(up_loaders[0].dataset)
    num_train_down = len(down_loaders[0].dataset)
    num_train = num_train_mean + num_train_up + num_train_down
    c_up_train, c_down_train = compute_pi(
        pred_mean_train,
        pred_up_train,
        pred_down_train,
        y_train,
        num_train_up,
        num_train_down,
    )

    ### COMPUTE PREDICTION INTERVAL COVERAGE PROBABILITY
    compute_picp(
        pred_mean_train,
        pred_up_train,
        pred_down_train,
        y_train,
        c_up_train[0],
        c_down_train[0],
    )

    fig1, ax1 = plt.subplots(1, 1)
    plot_uq_samples(
        ax1,
        pred_mean_train,
        pred_up_train,
        pred_down_train,
        y_train,
        c_up_train[0],
        c_down_train[0],
        comp_train,
    )
    fig1.tight_layout()
    plt.show()

    # Repeat for test set
    num_test_mean = len(mean_loaders[2].dataset)
    num_test_up = len(up_loaders[2].dataset)
    num_test_down = len(down_loaders[2].dataset)
    num_test = num_test_mean + num_test_up + num_test_down
    if num_test > 0:
        (
            pred_mean_test,
            pred_up_test,
            pred_down_test,
            y_test,
            comp_test,
        ) = compute_predictions(
            mean_loaders[2], (mean_model, model_up, model_down), config
        )
        c_up_test, c_down_test = compute_pi(
            pred_mean_test, pred_up_test, pred_down_test, y_test, num_test, num_test
        )
        compute_picp(
            pred_mean_test,
            pred_up_test,
            pred_down_test,
            y_test,
            c_up_test[0],
            c_down_test[0],
        )
        fig2, ax2 = plt.subplots(1, 1)
        plot_uq_samples(
            ax2,
            pred_mean_test,
            pred_up_test,
            pred_down_test,
            y_test,
            c_up_test[0],
            c_down_test[0],
            comp_test,
        )
        fig2.tight_layout()
        plt.show()

    fig3, ax3 = plt.subplots(1, 1)
    # plot_split_uq_intervals(ax3, pred_up_train, pred_down_train, c_up_train[0], c_down_train[0], comp_train)
    if num_test > 0:
        plot_uq_intervals(
            ax3,
            pred_up_test,
            pred_down_test,
            c_up_test[0],
            c_down_test[0],
            "test",
            "#998ec3",
        )
    plot_uq_intervals(
        ax3,
        pred_up_train,
        pred_down_train,
        c_up_train[0],
        c_down_train[0],
        "train",
        "#542788",
    )
    ax3.legend()
    fig3.tight_layout()
    plt.show()


def plot_uq_samples(ax, pred_mean, pred_up, pred_down, y, c_up, c_down, comp):
    """
    Plot mean, upper, and lower predictions.
    """

    """
    # Mean with error bars.
    bar = np.column_stack((c_down * pred_down, c_up * pred_up)).T
    markers, caps, bars = ax.errorbar(comp.squeeze(), pred_mean.squeeze(), yerr=bar, fmt='o', color="#005073", capsize=1.5, elinewidth=1, markersize=8)
    markers.set_alpha(0.5)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]

    # Mean only.
    ax.scatter(comp, y, edgecolor="#005073", marker="o", facecolor="none")

    # Fill between.
    ax.fill_between(comp, y1, y2, where=y2 >= y1, facecolor='cyan', alpha=0.5, interpolate=True)
    ax.fill_between(comp, y1, y2, where=y2 <= y1, facecolor='cyan', alpha=0.5, interpolate=True)
    """
    # Mean, upper, and lower.
    ax.scatter(y, pred_mean, edgecolor="#005073", marker="o", facecolor="none")
    ax.scatter(
        y,
        (pred_mean + c_up * pred_up),
        edgecolor="#a8e6cf",
        marker="o",
        facecolor="none",
    )
    ax.scatter(
        y,
        (pred_mean - c_down * pred_down),
        edgecolor="#ff8b94",
        marker="o",
        facecolor="none",
    )
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.10, 1.25])
    ax.set_xlabel("Atomic fraction (Fe)")
    ax.set_ylabel("Predicted enthalpy (normalized)")
    # ax.set_xlabel("DFT free energy")
    # ax.set_ylabel("GCNN free energy")


def plot_uq_intervals(ax, up, down, c_up, c_down, label, c):
    arr = up.detach() * c_up + down.detach() * c_down
    nbins = int(np.sqrt(len(arr)))
    hist, bins = np.histogram(arr, bins=nbins, density=True)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align="center", width=width, label=label, color=c, alpha=0.75)
    # savefig("intervals.png")
    # np.savetxt(label+".txt", np.column_stack((center, hist)))
    ax.set_xlabel("Prediction interval width")
    ax.set_ylabel("Probability density")


def plot_split_uq_intervals(ax, up, down, c_up, c_down, comp, split=0.8):
    train_up = up[comp < split]
    train_down = down[comp < split]
    # ax.scatter(comp, up.detach() * c_up + down.detach() * c_down, label="train")
    plot_uq_intervals(ax, train_up, train_down, c_up, c_down, "train")
    test_up = up[comp > split]
    test_down = down[comp > split]
    plot_uq_intervals(ax, test_up, test_down, c_up, c_down, "test")


def load_model(config, dataset, name, path):

    model = create_model_config(
        config=config["NeuralNetwork"]["Architecture"],
        verbosity=config["Verbosity"]["level"],
    )

    # model_name = model.__str__()
    output_name = name
    load_existing_model(model, output_name, path)
    return model


def compute_pi(pred_mean, pred_up, pred_down, y, num_samples_up, num_samples_down):
    boundaryOptimizer = CL_boundary_optimizer(
        y.numpy(),
        pred_mean,
        pred_up,
        pred_down,
        c_up0_ini=0.0,
        c_up1_ini=1000.0,
        c_down0_ini=0.0,
        c_down1_ini=1000.0,
        max_iter=1000,
    )

    num_outlier_list = [int(num_samples_up * (1 - x) / 2) for x in [0.90]]
    c_up = [
        boundaryOptimizer.optimize_up(outliers=x, verbose=0) for x in num_outlier_list
    ]
    num_outlier_list = [int(num_samples_down * (1 - x) / 2) for x in [0.90]]
    c_down = [
        boundaryOptimizer.optimize_down(outliers=x, verbose=0) for x in num_outlier_list
    ]
    print(c_up, c_down)
    return c_up, c_down


def compute_picp(pred_mean, pred_up, pred_down, y, c_up, c_down):
    """
    Compute prediction interval coverage probabilty - fraction of data within the bounds.
    """
    covered = 0.0

    num_samples = len(pred_mean)
    up = pred_mean + c_up * pred_up
    down = pred_mean - c_down * pred_down
    covered += ((down <= y) & (y <= up)).sum()
    print(
        "COVERED IN PI:",
        covered,
        "IN A TOTAL OF",
        num_samples,
        "PCIP:",
        covered / (num_samples),
    )


def compute_predictions(loader, models, config):
    """
    Compute predictions on the given dataset using mean/up/down trained models.
    """
    for m in models:
        m.eval()

    # device = next(models[0].parameters()).device
    verbosity = config["Verbosity"]["level"]

    pred = [None for i in range(len(models))]
    y = None
    comp = None
    for data in iterate_tqdm(loader, verbosity):
        # data = data.to(device)
        for i, m in enumerate(models):
            result = m(data)[0].detach()
            if pred[i] == None:
                pred[i] = result
            else:
                pred[i] = torch.cat((pred[i], result), 0)
        if y == None:
            y = data.y.detach()
            # if comp == None and hasattr(data, "comp"):
            #    comp = data.comp.detach()
        else:
            y = torch.cat((y, data.y.detach()), 0)
            # if hasattr(data, "comp"):
            #    comp = torch.cat((comp, data.comp), 0)

    return pred[0].detach(), pred[1].detach(), pred[2].detach(), y.detach(), comp


## NOTE: with MPI, the total dataset (before DDP splitting) should be used to create up and down, then re-split using DDP.
@torch.no_grad()
def create_loaders(loaders, model, config):
    """
    Create the up and down datasets by splitting on mean model predictions.
    """
    device = next(model.parameters()).device
    verbosity = config["Verbosity"]["level"]
    batch_size = config["NeuralNetwork"]["Training"]["batch_size"]

    model.eval()

    rank = 0
    if dist.is_initialized():
        _, rank = get_comm_size_and_rank()

    up_loaders = []
    down_loaders = []
    for l, loader in enumerate(loaders):
        up = []
        down = []
        nEq = 0
        for data in iterate_tqdm(loader, verbosity):
            data = data.to(device)
            diff = model(data)[0] - data.y

            data_copy = data.cpu().clone()
            data_copy.y = diff
            data_list = data_copy.to_data_list()
            size = diff.shape[0]
            for i in range(size):
                if data_copy.y[i] > 0:
                    up.append(data_list[i])
                elif data_copy.y[i] < 0:
                    data_copy.y[i] *= -1.0
                    down.append(data_list[i])
                else:
                    nEq += 1

        lengths = torch.tensor([len(up), len(down), nEq])
        lengths = lengths.to(device)
        dist.all_reduce(lengths)

        if rank == 0:
            print(
                "dataset:", l, "up", lengths[0], "down", lengths[1], "equal", lengths[2]
            )

        ## The code below creates the loaders on the LOCAL up and down samples, which is probably wrong.
        ## We should add a mechanism (either a gather-type communication, or a file write and read) to
        ## make the GLOBAL datasets visible in each GPU.
        up_sampler = []
        down_sampler = []
        if dist.is_initialized():
            up_sampler = torch.utils.data.distributed.DistributedSampler(up)
            down_sampler = torch.utils.data.distributed.DistributedSampler(down)

            up_loader = DataLoader(
                up, batch_size=batch_size, shuffle=False, sampler=up_sampler
            )
            down_loader = DataLoader(
                down, batch_size=batch_size, shuffle=False, sampler=down_sampler
            )
        else:
            up_loader = DataLoader(up, batch_size=batch_size, shuffle=True)
            down_loader = DataLoader(down, batch_size=batch_size, shuffle=True)

        up_loaders.append(up_loader)
        down_loaders.append(down_loader)

    return up_loaders, down_loaders, up_sampler, down_sampler


def train_model(loaders, sampler_list, output_name, config):
    """
    Train a model on the upper or lower dataset.
    """
    new_model = create_model_config(
        config=config["NeuralNetwork"]["Architecture"],
        verbosity=config["Verbosity"]["level"],
    )

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = get_summary_writer(output_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + output_name + "/config.json", "w") as f:
        json.dump(config, f)

    train_validate_test(
        new_model,
        optimizer,
        loaders[0],
        loaders[1],
        loaders[2],
        sampler_list,
        writer,
        scheduler,
        config["NeuralNetwork"],
        output_name,
        config["Verbosity"]["level"],
    )

    save_model(new_model, output_name, "logs/" + output_name)

    return new_model
