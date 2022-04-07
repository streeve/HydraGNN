import os, json
from torch_geometric.transforms import Distance
from hydragnn.preprocess import (
    RawDataLoader,
    split_dataset,
    split_dataset_biased,
    split_dataset_ignore,
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
    data.x = data.x[:, feature_indices]

split = lambda data: data.comp > 0.8
# train, val, test = split_dataset_biased(
#    dataset, 1.0, #config["NeuralNetwork"]["Training"]["perc_train"],
#    split)
# train, val, test = split_dataset_ignore(
#    dataset, config["NeuralNetwork"]["Training"]["perc_train"],
#    #1.0,
#    split)
train, val, test = split_dataset(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"], True
)
# 1.0, False)

train_loader, val_loader, test_loader, sampler_list = create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

run_uncertainty(
    config_file,  # _file?
    "logs/",  # fept_uq/3_",
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    True,
    True,
)
