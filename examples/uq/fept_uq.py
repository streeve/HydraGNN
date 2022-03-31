import os, json
from torch_geometric.transforms import Distance
from hydragnn.preprocess import (
    RawDataLoader,
    split_dataset_biased,
    create_dataloaders,
    update_predicted_values,
    get_radius_graph_config,
)
from hydragnn.utils import setup_ddp, update_config
from uq_pi3nn import run_uncertainty

try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

config_file = "./examples/lsms/fept_single.json"
with open(config_file, "r") as f:
    config = json.load(f)

setup_ddp()
loader = RawDataLoader(config["Dataset"])
dataset = loader.load_raw_data()

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
    # data.x = data.x[:, feature_indices]#.squeeze()

split = lambda data: data.comp > 0.8
train, val, test = split_dataset_biased(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"], split
)
train_loader, val_loader, test_loader, sampler_list = create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)
config = update_config(config, train_loader, val_loader, test_loader)

run_uncertainty(
    config,
    "logs/uq_mean/config.json",
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    True,
    True,
)
