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
for data in dataset:
    update_predicted_values(
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_index"],
        data,
    )
    compute_edges(data)
    data.x = data.x[:, feature_indices]
"""
sloader = SerializedDataLoader(config)
dataset = sloader.load_serialized_data(
    "serialized_dataset/MaterialsProject.pkl", config
)
for d in dataset:
    d.x = torch.nn.functional.one_hot(d.x.to(torch.int64)).to(torch.float32)

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
split = lambda data: (data.spacegroup_no < 195 and data.spacegroup_no > 142) or len(np.unique(data.x.cpu())) > 4 or np.any(np.isclose(data.x.cpu(), (9-1)/(94.0-1), atol=1e-4))

#train, val, test = split_dataset_biased(
#    dataset, config["NeuralNetwork"]["Training"]["perc_train"], split
#)

train, val, test = split_dataset_ignore(
    dataset,
    config["NeuralNetwork"]["Training"]["perc_train"],
#    # 1.0,
    split)
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
    True,
    True,
)
