import os, json
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric

import hydragnn
from uq_pi3nn import run_uncertainty


def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    data.x = (data.x - 1.0) / 8.0  # max 9 min 1

    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    hydragnn.preprocess.update_predicted_values(
        var_config["type"],
        var_config["output_index"],
        data,
    )
    device = hydragnn.utils.get_device()
    return data.to(device)


def normalize(dataset):
    minmax = np.full((2, 1), np.inf)
    minmax[1, :] *= -1
    num_graph_features = 1
    for data in tqdm(dataset):
        # find maximum and minimum values for graph level features
        for ifeat in range(num_graph_features):
            minmax[0, ifeat] = min(data.y[ifeat], minmax[0, ifeat])
            minmax[1, ifeat] = max(data.y[ifeat], minmax[1, ifeat])
    for data in tqdm(dataset):
        for ifeat in range(num_graph_features):
            data.y[ifeat] = (data.y[ifeat] - minmax[0, ifeat]) / (
                minmax[1, ifeat] - minmax[0, ifeat]
            )

    return dataset


# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(__file__), "../qm9/qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
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
dataset = normalize(dataset)

"""from sklearn.preprocessing import OneHotEncoder
X = np.array([len(dataset), 5])
encoder = OneHotEncoder(handle_unknown='ignore')
for d, data in enumerate(dataset):
    print(encoder.fit_transform(data.x).toarray())
from sklearn import decomposition
pca = decomposition.PCA(n_components=5)
pca.fit(data.x)
pca_x = pca.transform(data.x)
"""

split = lambda data: 9 in data.x
train, val, test = hydragnn.preprocess.split_dataset_ignore(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"], split
)
# train, val, test = hydragnn.preprocess.split_dataset(
#    dataset, #config["NeuralNetwork"]["Training"]["perc_train"],
#    1.0, False
# )
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
    "logs",
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    True,
    True,
)
