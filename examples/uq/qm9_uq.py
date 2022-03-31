import os, json
import torch
import torch_geometric

import hydragnn
from uq_pi3nn import run_uncertainty


def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    hydragnn.preprocess.update_predicted_values(
        var_config["type"],
        var_config["output_index"],
        data,
    )
    device = hydragnn.utils.get_device()
    return data.to(device)


# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(__file__), "../qm9/qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()

# Use built-in torch_geometric dataset.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
dataset = torch_geometric.datasets.QM9(
    root="dataset/qm9", pre_transform=qm9_pre_transform  # , pre_filter=qm9_pre_filter
)
split = lambda data: 7 in data.x
train, val, test = hydragnn.preprocess.split_dataset_biased(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"], split
)
(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

run_uncertainty(
    "./examples/qm9/qm9.json",
    "logs/uq_mean/config.json",
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    True,
    True,
)
