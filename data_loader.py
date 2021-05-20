import pickle
import os
import torch
import time
import networkx as nx
import numpy as np
from torch_geometric.data import Data, DataLoader
import copy


class DataIterator(object):
    def __init__(self, data_list):
        self.data_list = data_list

    def get_snapshot(self):
        return self.data_list[self.curr_timestamp]

    def get_prev_snapshot(self):
        if self.curr_timestamp == 0:
            return None
        return self.data_list[self.curr_timestamp - 1]  # todo check on code if I use prev or next

    def __next__(self):
        if self.curr_timestamp < len(self.data_list):
            snapshot = self.get_snapshot()
            self.curr_timestamp = self.curr_timestamp + 1
            return snapshot
        else:
            self.curr_timestamp = 0
            raise StopIteration

    def __iter__(self):
        self.curr_timestamp = 0
        return self


def create_loader(data_list):
    my_loader = DataLoader(data_list, num_workers=0, pin_memory=True)
    return my_loader


def get_pickled_data_list(data_name, timestamps, cuda_device_id):
    t = time.time()
    data_list = pickle.load(open(f"./dataset/{data_name}/pkl/all_timestamps/"
                                 f"data_list_{timestamps}_cuda_{cuda_device_id}.pkl", "rb"))
    print(f"data list loaded: {time.time() - t}")
    return data_list


class DatasetLoader(object):
    def __init__(self, data_name, cuda_device, num_classes=None, timestamps=None, Nj=None,):

        self.cuda_device = cuda_device
        self.device = torch.device(f'cuda:{self.cuda_device}') if torch.cuda.is_available() else torch.device('cpu')

        self.data_name = data_name
        self.timestamps = timestamps
        self.num_classes = num_classes
        self._data_info()

        self.labels = list()
        self.features_mx = list()
        self.graphs_adj = list()
        self._load_data()

        self.test_indices = None
        self.val_indices = None
        self.train_indices = None
        self.train_test_val_split()

        self.train = None
        self.test = None
        self.validation = None
        self.train_labels = None
        self.test_labels = None
        self.val_labels = None
        self.train_test_val_list()

        self.Nj = Nj
        self.unbalanced_classes()

        self.dataset = self.get_dataset()

        self.my_data = copy.deepcopy(self.dataset)

    def _data_info(self):

        if self.data_name == 'DBLP':
            self.timestamps = 21 if self.timestamps is None else self.timestamps
            # or 15 with 0
            self.num_classes = 14 if self.num_classes is None else self.num_classes

        elif self.data_name == 'IMDB':
            self.timestamps = 10 if self.timestamps is None else self.timestamps
            self.num_classes = 11 if self.num_classes is None else self.num_classes

        else:  # tmall
            self.timestamps = 9 if self.timestamps is None else self.timestamps
            self.num_classes = 2 if self.num_classes is None else self.num_classes

    def _load_data(self):

        for i in range(self.timestamps):
            with open(os.path.join('dataset', self.data_name, 'input', 'graph_' + str(i) + '.pkl'), 'rb') as f:
                g = pickle.load(f)
            with open(os.path.join('dataset', self.data_name, 'input', 'one_hot_labels_' + str(i) + '.pkl'), 'rb') as f:
                l = pickle.load(f)
            with open(os.path.join('dataset', self.data_name, 'input', 'mx_' + str(i) + '.pkl'), 'rb') as f:
                mx = pickle.load(f)

            self.labels.append(l)

            input_features = torch.tensor(np.vstack([mx[node] for node in range(len(mx))]), device=self.device)

            adj = nx.adjacency_matrix(g).tocoo()

            input_adj = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long).to(self.device)

            self.graphs_adj.append(input_adj)
            self.features_mx.append(input_features)

    def train_test_val_split(self):

        if self.data_name == 'Tmall':
            bipartite_products = 200
            person_data = np.delete(np.arange(len(self.labels[0])),
                                    np.arange(bipartite_products))  # the first 200 inices are products
            val_test_inds = np.random.choice(person_data, round(len(person_data) * 0.4), replace=False)
            self.test_indices, self.val_indices = np.array_split(val_test_inds, 2)
            self.train_indices = np.setdiff1d(np.arange(len(self.labels[0])), val_test_inds)

        elif self.data_name == 'DBLP':
            ind_set = set()
            for timestamp in range(self.timestamps):  # times inds
                for n in range(len(self.labels[timestamp])):  # vertices
                    if self.labels[timestamp][n] == -1 or all(v == 0 for v in self.labels[timestamp][n]):
                        continue
                    else:
                        ind_set.add(n)

            val_test_inds = np.random.choice(list(ind_set), round(len(ind_set) * 0.4), replace=False)
            self.test_indices, self.val_indices = np.array_split(val_test_inds, 2)
            self.train_indices = np.setdiff1d(np.array(list(ind_set)), val_test_inds)
        else:
            val_test_inds = np.random.choice(len(self.labels[0]), round(len(self.labels[0]) * 0.4), replace=False)
            self.test_indices, self.val_indices = np.array_split(val_test_inds, 2)
            self.train_indices = np.delete(np.arange(len(self.labels[0])), val_test_inds)

    def train_test_val_list(self):

        self.train = [torch.tensor([k for k in self.train_indices if self.labels[j][k] != -1],
                                   dtype=torch.long).to(self.device) for j in range(len(self.labels))]
        self.test = [torch.tensor([k for k in self.test_indices if self.labels[j][k] != -1],
                                  dtype=torch.long).to(self.device) for j in range(len(self.labels))]
        self.validation = [torch.tensor([k for k in self.val_indices if self.labels[j][k] != -1],
                                        dtype=torch.long).to(self.device) for j in range(len(self.labels))]

        self.test_labels = [torch.tensor([self.labels[j][k] for k in self.test_indices if self.labels[j][k] != -1],
                                         dtype=torch.double).to(self.device) for j in range(self.timestamps)]
        self.train_labels = [torch.tensor([self.labels[j][k] for k in self.train_indices if self.labels[j][k] != -1],
                                          dtype=torch.double).to(self.device) for j in range(self.timestamps)]
        self.val_labels = [torch.tensor([self.labels[j][k] for k in self.val_indices if self.labels[j][k] != -1],
                                        dtype=torch.double).to(self.device) for j in range(self.timestamps)]

        path = "./dataset/" + self.data_name + "/pkl"
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(self.test_indices, open("./dataset/" + self.data_name + "/pkl/test_inds.pkl", "wb"))
        pickle.dump(self.val_indices, open("./dataset/" + self.data_name + "/pkl/val_indices.pkl", "wb"))
        pickle.dump(self.train_indices, open("./dataset/" + self.data_name + "/pkl/train_inds.pkl", "wb"))

        pickle.dump(self.train, open("./dataset/" + self.data_name + "/pkl/train.pkl", "wb"))
        pickle.dump(self.test, open("./dataset/" + self.data_name + "/pkl/test.pkl", "wb"))
        pickle.dump(self.validation, open("./dataset/" + self.data_name + "/pkl/validation.pkl", "wb"))

    def unbalanced_classes(self):
        # weights normalized by the train
        if self.Nj is not None:
            self.Nj = torch.as_tensor([sum([self.train_labels[u][t][j] for u in range(self.timestamps)
                                            for t in range(len(self.train_labels[u]))])
                                       for j in range(self.num_classes)]).to(device=self.device, dtype=torch.float)

    def set_data_to_new_device(self, data_list):
        for i, data in enumerate(data_list):
            data.x = data.x.to(self.device)
            data.edge_index = data.edge_index.to(device=self.device)
            data.Nj_s = data.Nj_s.to(self.device) if data.Nj_s is not None else None
            data.training_inds = data.training_inds.to(self.device)
            data.val_inds = data.val_inds.to(self.device)
            data.test_inds = data.test_inds.to(self.device)
            data.training_labels = data.training_labels.to(self.device)
            data.test_labels = data.test_labels.to(self.device)
            data.val_labels = data.val_labels.to(self.device)
        my_loader = create_loader(data_list)
        return my_loader

    def get_dataset(self):

        data_list = []
        for i in range(self.timestamps):
            data = Data(x=self.features_mx[i], edge_index=self.graphs_adj[i])
            data.Nj_s = self.Nj
            data.training_inds = self.train[i]
            data.val_inds = self.validation[i]
            data.test_inds = self.test[i]
            data.training_labels = self.train_labels[i]
            data.test_labels = self.test_labels[i]
            data.val_labels = self.val_labels[i]
            data_list.append(data)

        print("Data_list created")

        pkl_path = os.path.join(os.getcwd(), 'dataset', self.data_name, "pkl", "all_timestamps")
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)

        pickle.dump(data_list, open(f"./dataset/{self.data_name}/"
                                    f"pkl/all_timestamps/data_list_{self.timestamps}_cuda_{self.cuda_device}.pkl",
                                    "wb"))

        my_loader = create_loader(data_list)
        return my_loader


if __name__ == '__main__':
    data_name = 'DBLP'
    timestamps = 2
    cuda_device = 0
    loader = DatasetLoader(data_name='DBLP', timestamps=2, Nj='1/Njs')
