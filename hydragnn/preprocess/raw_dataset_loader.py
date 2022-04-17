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

import os
import numpy as np
import pickle
import pathlib
from glob import glob

import csv

import torch
from torch_geometric.data import Data
from torch import tensor

from ase.io.cfg import read_cfg
from ase.io.cif import read_cif
from ase.io import read
from ase.spacegroup import get_spacegroup

from hydragnn.utils import iterate_tqdm

# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


def import_property_csv_file(csv_file):
    dict_from_csv = {}

    with open(csv_file, mode="r") as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]: rows[1] for rows in reader}

    return dict_from_csv


class RawDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config):
        """
        config:
          shows the dataset path the target variables information, e.g, location and dimension, in data file
        ###########
        dataset_list:
          list of datasets read from self.path_dictionary
        serial_data_name_list:
          list of pkl file names
        node_feature_dim:
          list of dimensions of node features
        node_feature_col:
          list of column location/index (start location if dim>1) of node features
        graph_feature_dim:
          list of dimensions of graph features
        graph_feature_col: list,
          list of column location/index (start location if dim>1) of graph features
        """
        self.dataset_list = []
        self.serial_data_name_list = []
        self.node_feature_name = config["node_features"]["name"]
        self.node_feature_dim = config["node_features"]["dim"]
        self.node_feature_col = config["node_features"]["column_index"]
        self.graph_feature_name = config["graph_features"]["name"]
        self.graph_feature_dim = config["graph_features"]["dim"]
        self.graph_feature_col = config["graph_features"]["column_index"]
        self.raw_dataset_name = config["name"]
        self.data_format = config["format"]
        self.path_dictionary = config["path"]
        self.types = []

    def load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        if not os.path.exists(serialized_dir):
            os.mkdir(serialized_dir)

        for dataset_type, raw_data_path in self.path_dictionary.items():
            all_files = glob(raw_data_path + "/*.cif")
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)

            assert len(all_files) > 0, "No data files provided in {}!".format(
                raw_data_path
            )

            dictionary_property = None
            if self.data_format == "CIF":
                property_path = (
                    raw_data_path + "/../properties-reference/" + "formationenergy.csv"
                )
                dictionary_property = import_property_csv_file(property_path)
            dataset = [None] * len(dictionary_property.keys())

            count = 0
            for name in iterate_tqdm(all_files, 2):
                # if the directory contains file, iterate over them
                data = self.__transform_input_to_data_object_base(
                    filepath=os.path.join(raw_data_path, name),
                    dictionary_property=dictionary_property,
                )
                if data is not None:
                    dataset[count] = data
                    count += 1

                # if the directory contains subdirectories, explore their content
                """
                elif os.path.isdir(os.path.join(raw_data_path, name)):
                    dir_name = os.path.join(raw_data_path, name)
                    for subname in os.listdir(dir_name):
                        if os.path.isfile(os.path.join(dir_name, subname)):
                            data_object = self.__transform_input_to_data_object_base(
                                filepath=os.path.join(dir_name, subname),
                                dictionary_property=dictionary_property,
                            )
                            if not isinstance(data_object, type(None)):
                                dataset.append(data_object)
                """
            # scaled features by number of nodes
            dataset = self.__scale_features_by_num_nodes(dataset)

            if dataset_type == "total":
                serial_data_name = self.raw_dataset_name + ".pkl"
            else:
                # append for train; test; validation
                serial_data_name = self.raw_dataset_name + "_" + dataset_type + ".pkl"

            self.dataset_list.append(dataset)
            self.serial_data_name_list.append(serial_data_name)

        self.__normalize_dataset()

        for serial_data_name, dataset_normalized in zip(
            self.serial_data_name_list, self.dataset_list
        ):
            with open(os.path.join(serialized_dir, serial_data_name), "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(dataset_normalized, f)

        return dataset

    def __transform_input_to_data_object_base(self, filepath, dictionary_property):
        if self.data_format == "CFG" or self.data_format == "CIF":
            data_object = self.__transform_input_file_to_data_object_base(
                filepath=filepath, dictionary_property=dictionary_property
            )
        elif self.data_format == "LSMS" or self.data_format == "unit_test":
            data_object = self.__transform_LSMS_input_to_data_object_base(
                filepath=filepath
            )

        return data_object

    def __transform_input_file_to_data_object_base(
        self, filepath, dictionary_property=None
    ):
        """Transforms lines of strings read from the raw data CIF file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        file_path_splitting = os.path.split(filepath)
        filename_without_extension = os.path.splitext(file_path_splitting[1])[0]

        if filename_without_extension in dictionary_property.keys():
            data_object = self.__transform_CIF_file_to_data_object(
                filepath, dictionary_property, filename_without_extension
            )
            return data_object

    def __transform_CFG_file_to_data_object(self, filepath):

        # FIXME:
        #  this still assumes bulk modulus is specific to the CFG format.
        #  To deal with multiple files across formats, one should generalize this function
        #  by moving the reading of the .bulk file in a standalone routine.
        #  Morevoer, this approach assumes tha there is only one global feature to look at,
        #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

        ase_object = read_cfg(filepath)

        data_object = Data()

        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
        c_peratom = np.expand_dims(ase_object.arrays["c_peratom"], axis=1)
        fx = np.expand_dims(ase_object.arrays["fx"], axis=1)
        fy = np.expand_dims(ase_object.arrays["fy"], axis=1)
        fz = np.expand_dims(ase_object.arrays["fz"], axis=1)
        node_feature_matrix = np.concatenate(
            (proton_numbers, masses, c_peratom, fx, fy, fz), axis=1
        )
        data_object.x = tensor(node_feature_matrix).float()

        filename_without_extension = os.path.splitext(filepath)[0]

        if os.path.exists(os.path.join(filename_without_extension + ".bulk")):
            filename_bulk = os.path.join(filename_without_extension + ".bulk")
            f = open(filename_bulk, "r", encoding="utf-8")
            lines = f.readlines()
            graph_feat = lines[0].split(None, 2)
            g_feature = []
            # collect graph features
            for item in range(len(self.graph_feature_dim)):
                for icomp in range(self.graph_feature_dim[item]):
                    it_comp = self.graph_feature_col[item] + icomp
                    g_feature.append(float(graph_feat[it_comp].strip()))
            data_object.y = tensor(g_feature)

        return data_object

    def __transform_CIF_file_to_data_object(
        self, filepath, dictionary_property, filename_without_extension
    ):

        # FIXME:
        #  this still assumes bulk modulus is specific to the CIG format.

        # I do not succeed in making ase.io.cif.read_cif work, so I use ase.io.read
        ase_object = read(filepath)

        data_object = Data()
        sg = get_spacegroup(ase_object)
        data_object.spacegroup = sg.symbol
        data_object.spacegroup_no = sg.no
        data_object.supercell_size = tensor(ase_object.cell.array, dtype=torch.float32)
        data_object.pos = tensor(ase_object.arrays["positions"], dtype=torch.float32)
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        data_object.x = tensor(proton_numbers, dtype=torch.float32)
        data_object.y = tensor([float(dictionary_property[filename_without_extension])])
        return data_object

    def __transform_LSMS_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        if name == ".DS_Store":
            return None

        data_object = Data()

        graph_feat = np.loadtxt(lines[0:1])
        g_feature = []
        # collect graph features
        for item in range(len(self.graph_feature_dim)):
            for icomp in range(self.graph_feature_dim[item]):
                it_comp = self.graph_feature_col[item] + icomp
                g_feature.append(graph_feat[it_comp])
        data_object.y = tensor(g_feature, dtype=torch.float32)

        nodes = np.loadtxt(lines[1:])
        node_features = nodes[:, self.node_feature_col]

        data_object.pos = tensor(nodes[:, 2:5], dtype=torch.float32)
        data_object.x = tensor(node_features, dtype=torch.float32)

        elements, counts = np.unique(nodes[:, 0], return_counts=True)
        data_object.comp = tensor(counts[0] / np.shape(nodes)[0], dtype=torch.float32)
        # this is a hack.
        if len(elements) == 1 and elements[0] == 26.0:
            data_object.comp = tensor(0.0, dtype=torch.float32)

        return data_object

    def __charge_density_update_for_LSMS(self, data_object: Data):
        """Calculate charge density for LSMS format
        Parameters
        ----------
        data_object: Data
            Data object representing structure of a graph sample.

        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        num_of_protons = data_object.x[:, 0]
        charge_density = data_object.x[:, 1]
        charge_density -= num_of_protons
        data_object.x[:, 1] = charge_density
        return data_object

    def __scale_features_by_num_nodes(self, dataset):
        """Calculate [**]_scaled_num_nodes"""
        scaled_graph_feature_index = [
            i
            for i in range(len(self.graph_feature_name))
            if "_scaled_num_nodes" in self.graph_feature_name[i]
        ]
        scaled_node_feature_index = [
            i
            for i in range(len(self.node_feature_name))
            if "_scaled_num_nodes" in self.node_feature_name[i]
        ]

        for idx, data_object in enumerate(dataset):
            dataset[idx].y[scaled_graph_feature_index] = (
                dataset[idx].y[scaled_graph_feature_index] / data_object.num_nodes
            )
            dataset[idx].x[:, scaled_node_feature_index] = (
                dataset[idx].x[:, scaled_node_feature_index] / data_object.num_nodes
            )

        return dataset

    def __normalize_dataset(self):

        """Performs the normalization on Data objects and returns the normalized dataset."""
        num_node_features = self.dataset_list[0][0].x.shape[1]
        num_graph_features = len(self.dataset_list[0][0].y)

        self.minmax_graph_feature = np.full((2, num_graph_features), np.inf)
        # [0,...]:minimum values; [1,...]: maximum values
        self.minmax_node_feature = np.full((2, num_node_features), np.inf)
        self.minmax_graph_feature[1, :] *= -1
        self.minmax_node_feature[1, :] *= -1
        for dataset in self.dataset_list:
            for data in dataset:
                # find maximum and minimum values for graph level features
                for ifeat in range(num_graph_features):
                    self.minmax_graph_feature[0, ifeat] = min(
                        data.y[ifeat], self.minmax_graph_feature[0, ifeat]
                    )
                    self.minmax_graph_feature[1, ifeat] = max(
                        data.y[ifeat], self.minmax_graph_feature[1, ifeat]
                    )
                # find maximum and minimum values for node level features
                for ifeat in range(num_node_features):
                    self.minmax_node_feature[0, ifeat] = np.minimum(
                        np.amin(data.x[:, ifeat].numpy()),
                        self.minmax_node_feature[0, ifeat],
                    )
                    self.minmax_node_feature[1, ifeat] = np.maximum(
                        np.amax(data.x[:, ifeat].numpy()),
                        self.minmax_node_feature[1, ifeat],
                    )
        for dataset in self.dataset_list:
            for data in dataset:
                for ifeat in range(num_graph_features):
                    data.y[ifeat] = tensor_divide(
                        (data.y[ifeat] - self.minmax_graph_feature[0, ifeat]),
                        (
                            self.minmax_graph_feature[1, ifeat]
                            - self.minmax_graph_feature[0, ifeat]
                        ),
                    )
                for ifeat in range(num_node_features):
                    data.x[:, ifeat] = tensor_divide(
                        (data.x[:, ifeat] - self.minmax_node_feature[0, ifeat]),
                        (
                            self.minmax_node_feature[1, ifeat]
                            - self.minmax_node_feature[0, ifeat]
                        ),
                    )
