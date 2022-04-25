import os, json
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.transforms import Distance
from hydragnn.preprocess import (
    RawDataLoader,
    SerializedDataLoader,
    split_dataset,
    split_dataset_biased,
    split_dataset_ignore,
    create_dataloaders,
    update_predicted_values,
    get_radius_graph_config,
)
from hydragnn.utils import setup_ddp, update_config
from uq_pi3nn import run_uncertainty

os.environ["SERIALIZED_DATA_PATH"] = "/gpfs/alpine/world-shared/csc457/reeve"

config_file = "./examples/materials_project/formation_energy.json"
with open(config_file, "r") as f:
    config = json.load(f)

setup_ddp()
"""
rloader = RawDataLoader(config["Dataset"])
dataset = rloader.load_raw_data()

compute_edges = get_radius_graph_config(config["NeuralNetwork"]["Architecture"])
feature_indices = config["NeuralNetwork"]["Variables_of_interest"][
    "input_node_features"
]
types = []
for data in dataset:
    update_predicted_values(
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_index"],
        data,
    )
    compute_edges(data)
    data.x = data.x[:, feature_indices]

    #unique = np.unique(data.x.numpy())
    #for t in unique:
    #    if t not in types:
    #        types.append(int(t*93+1+0.1))

#print(types)
#print(len(types))
    # 89 unique types, 94 max element
    data.x = torch.nn.functional.one_hot(data.x.to(torch.int64), num_classes=94).to(torch.float32)
"""

sloader = SerializedDataLoader(config)
dataset = sloader.load_serialized_data(
    "serialized_dataset/MaterialsProject.pkl", config
)
for data in dataset:
    data.x = data.x*93
    #print(data.x)
    data.x = torch.nn.functional.one_hot(data.x.to(torch.int64), num_classes=94).to(torch.float32).squeeze(dim=1)
    #print(data.x)

# Rhombohedral
# split = lambda data: "R" in data.spacegroup
# Ranges of space groups
#split = lambda data: data.spacegroup_no < 195 and data.spacegroup_no > 142
# (data.spacegroup_no in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228])
# int(data.spacegroup_no) < 3 #or (data.spacegroup_no > 142 and data.spacegroup_no < 195)
# Systems with lithium (normalized values)
# split = lambda data: np.any(np.isclose(data.x, (3-1)/(94.0-1), atol=1e-4))
# 4 component & higher
# split = lambda data: len(np.unique(data.x)) > 4

# Any of the 3 example test sets
split_all = lambda data: len(torch.nonzero(torch.sum(data.x, dim=0))) > 4 or 1 in data.x[:,8] or (data.spacegroup_no < 195 and data.spacegroup_no > 142)
ignore = lambda data: 1 in data.x[:,8] or (data.spacegroup_no < 195 and data.spacegroup_no > 142)
bias = lambda data: len(torch.nonzero(torch.sum(data.x, dim=0))) > 4

#train, val, test = split_dataset_biased_ignore(
#    dataset, config["NeuralNetwork"]["Training"]["perc_train"], bias, ignore,
#)

train, val, test = split_dataset_ignore(
    dataset,
    config["NeuralNetwork"]["Training"]["perc_train"],
#    # 1.0,
    split_all)
# train, val, test = split_dataset(
#    dataset,
#    config["NeuralNetwork"]["Training"]["perc_train"], #1.0,
#    False,
# )

train_loader, val_loader, test_loader, sampler_list = create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

run_uncertainty(
    config_file,  # _file?
    "logs/",
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    False,
    True,
)
