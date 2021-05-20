import numpy as np
from torch.optim import Adam, SGD
import time
from psgd import *
from models import *
from loggers import CSVLogger, PrintLogger, EmptyLogger
import utils
import nni
import pickle
import os



class ModelRunner:
    def __init__(self, params, loader, cuda_device, data_logger=None, epochs_logger=None):

        self._params = params

        self.prev_training_inds = None
        self.prev_val_inds = None
        self.prev_test_inds = None

        self._epoch_logger = EmptyLogger() if epochs_logger is None else epochs_logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger

        self._device = torch.device(f'cuda:{cuda_device}') if torch.cuda.is_available() else torch.device('cpu')

        self._mse_loss = self.weighted_mse_loss
        self._temporal_loss = self.weighted_mse_loss

        self.net = None
        self.opt = None
        self.loader = loader
        self.num_features = loader.dataset[0].num_features #loader[0].num_features

        # if SSP
        self.preconditioner = None
        self.eps = params['eps']
        self.update_freq = params['update_freq']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.lamda = params['lamda']

        self.best_loss = None
        # self.best_model = None
        self.best_epoch = None

        self.is_nni = params["is_nni"]

    def architecture(self):

        if self._params["model_name"] == 'GCN':
            self.net = GCN(num_of_features=self.num_features,
                           hid_size=self._params["hid_size"],
                           num_of_classes=self._params["number_of_classes"],
                           activation=F.relu,
                           dropout=self._params["dropout"]).to(self._device)


        elif self._params["model_name"] == 'GAT':
            self.net = GatNet(num_features=self.num_features,
                              num_classes=self._params["number_of_classes"],
                              hid_layer=self._params["hid_size"],
                              dropout=self._params["dropout"],
                              activation=F.elu,
                              heads=[8, 8]
                              ).to(self._device)

        elif self._params["model_name"] == 'SSP':
            self.net = SSPNet(num_of_features=self.num_features,
                              hid_size=self._params["hid_size"],
                              num_of_classes=self._params["number_of_classes"],
                              activation=F.relu,
                              dropout=self._params["dropout"]).to(self._device)

            self.preconditioner = KFAC(self.net, self.eps,
                                       sua=False,
                                       pi=False,
                                       update_freq=self.update_freq,
                                       alpha=self.alpha if self.alpha is not None else 1.,
                                       constraint_norm=False)

        elif self._params["model_name"] == 'GCNII':
            self.net = GCNII(num_of_features=self.num_features,
                             num_of_classes=self._params["number_of_classes"],
                             hid_size=self._params["hid_size"],
                             num_layers=self._params["num_layers"],
                             alpha=self._params["alpha"],
                             theta=self._params["theta"],
                             shared_weights=True,
                             dropout=self._params["dropout"]).to(self._device)

        if self._params['optimizer'] == 'SGD':
            self.opt = SGD(self.net.parameters(),
                           lr=self._params['lr'],
                           weight_decay=self._params['weight_decay'])

        elif self._params['optimizer'] == 'Adam':
            if self._params["model_name"] == 'GCNII':

                self.opt = torch.optim.Adam([dict(params=self.net.convs.parameters(), weight_decay=0.01),
                                             dict(params=self.net.lins.parameters(), weight_decay=5e-4)], lr=self._params['lr'])
            else:
                self.opt = Adam(self.net.parameters(),
                                lr=self._params['lr'],
                                weight_decay=self._params['weight_decay'])

    def weighted_mse_loss(self, pred, target, weights=None):
        if weights is None:
            return ((pred - target) ** 2).sum(dim=1).sum().to(self._device)
        elif self._params['loss_weights_type'] == 'sqrt(N/Nj)':
            loss = (torch.sqrt((weights).sum() / weights) * (pred - target) ** 2).sum(dim=1).sum().to(self._device)
            return loss
        elif self._params['loss_weights_type'] == '1/Njs':
            w = weights ** (-1)
            loss = (w * (pred - target) ** 2).sum(dim=1).sum().to(self._device)
            return loss

    def train(self, epoch):

        self.net.train()
        z_vals, outputs = [], []

        tempo_loss, loss_train = 0., 0.

        self.opt.zero_grad()

        f1_score_macro, f1_score_micro = [], []  # for all the timestamps only at the last epoch
        z_appearances, out_appearances = 0., 0.

        for i, data in enumerate(self.loader):

            z, output = self.net(data)
            output = output[data.training_inds]

            Nj_s = None if self._params['loss_weights_type'] == None else data.Nj_s

            loss_train += self._mse_loss(output, data.training_labels, Nj_s)

            z_vals.append(z)  # output of 1 GCN layer
            outputs.append(output)  # final predictions

            out_appearances += len(data.training_inds)

            if i > 0:
                z_inds = utils.intersection_inds(data.training_inds, self.prev_training_inds)
                z_appearances += len(z_inds)
                loss = self._temporal_loss(z_vals[i][z_inds], z_vals[i - 1][z_inds])
                tempo_loss += self._params["temporal_pen"] * loss

            self.prev_training_inds = data.training_inds

            f1_mac, f1_mic, _, _ = utils.f1_score_func(output, data.training_labels)
            f1_score_macro.append(f1_mac)
            f1_score_micro.append(f1_mic)  # todo

        tempo_loss /= z_appearances
        loss_train /= out_appearances
        total_loss = loss_train + tempo_loss

        total_loss.backward()

        if self.preconditioner:
            lam = (float(epoch) / float(self._params["epochs"])) ** self.gamma if self.gamma is not None else 0.
            self.preconditioner.step(lam=lam)

        self.opt.step()

        train_res_dict = {"loss": loss_train,
                          "f1_score_macro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_macro), 4),
                          "f1_score_micro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_micro), 4),
                          "tempo_loss": tempo_loss}

        return train_res_dict

    def validation(self, epoch):

        self.net.eval()
        with torch.no_grad():

            z_vals, outputs = [], []

            tempo_loss = 0.
            loss = 0.

            # self.prev_valid_inds = []
            f1_score_macro, f1_score_micro = [], []  # for all the timestamps only at the last epoch

            z_appearances = 0.
            out_appearances = 0.

            for i, data in enumerate(self.loader):

                z, output = self.net(data)
                output = output[data.val_inds]

                loss += self._mse_loss(output, data.val_labels)

                z_vals.append(z)  # After 1 GCN layer
                outputs.append(output)  # Final predictions

                out_appearances += len(data.val_inds)

                if i > 0:
                    z_inds = utils.intersection_inds(data.val_inds, self.prev_val_inds)
                    z_appearances += len(z_inds)
                    loss_valid = self._temporal_loss(z_vals[i][z_inds], z_vals[i - 1][z_inds])
                    tempo_loss += self._params["temporal_pen"] * loss_valid

                self.prev_val_inds = data.val_inds

                # if epoch == self._params['epochs'] - 1:

                f1_mac, f1_mic, _, _ = utils.f1_score_func(output, data.val_labels)
                f1_score_macro.append(f1_mac)
                f1_score_micro.append(f1_mic)

            tempo_loss /= z_appearances
            loss /= out_appearances
            total_loss_valid = loss + tempo_loss

            valid_res_dict = {"loss": loss,
                              "f1_score_macro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_macro),
                                                                                                4),
                              "f1_score_micro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_micro),
                                                                                                4),
                              "tempo_loss": tempo_loss}

            if total_loss_valid < self.best_loss:
                self.best_loss = total_loss_valid
                self.best_epoch = epoch

        return valid_res_dict


    def test(self):
        with torch.no_grad():
            self.net.eval()

            z_vals, outputs = [], []

            tempo_loss = 0.
            loss = 0.

            # prev_test_inds = []
            f1_score_macro, f1_score_micro = [], []  # for all the timestamps only at the last epoch
            real, pred = [], []

            z_appearances = 0.
            out_appearances = 0.

            for i, data in enumerate(self.loader):

                z, output = self.net(data)

                grid_outputs_folder = f"lr_{self._params['lr']}_do_{self._params['dropout']}_hs_{self._params['hid_size']}_tp_{self._params['temporal_pen']}_wd_{self._params['weight_decay']}_optimizer_{self._params['optimizer']}"

                t = time.time()
                products_path = os.path.join(os.getcwd(), 'dataset', self._params["data_name"],
                                             "gcn_outputs", grid_outputs_folder)
                if not os.path.exists(products_path):
                    os.makedirs(products_path)

                with open(os.path.join("./dataset", self._params["data_name"], "gcn_outputs", grid_outputs_folder, "gcn_out_" + str(i) + ".pkl"), "wb") as f:
                    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

                if self.is_nni is False:
                    print(f"pickles dump time:{time.time() - t}")
                output = output[data.test_inds]

                loss += self._mse_loss(output, data.test_labels)

                z_vals.append(z)  # After 1 GCN layer
                outputs.append(output)  # Final predictions

                out_appearances += len(data.test_inds)

                if i > 0:
                    z_inds = utils.intersection_inds(data.test_inds, self.prev_test_inds)
                    z_appearances += len(z_inds)
                    loss_test = self._temporal_loss(z_vals[i][z_inds], z_vals[i - 1][z_inds])
                    tempo_loss += self._params["temporal_pen"] * loss_test

                self.prev_test_inds = data.test_inds

                f1_mac, f1_mic, list_real, list_pred = utils.f1_score_func(output, data.test_labels)
                f1_score_macro.append(f1_mac)
                f1_score_micro.append(f1_mic)
                real.extend(list_real)
                pred.extend(list_pred)

            tempo_loss /= z_appearances
            loss /= out_appearances
            total_loss_test = loss + tempo_loss

            test_res_dict = {"loss": loss,
                             "f1_score_macro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_macro),
                                                                                               4),
                             "f1_score_micro": None if len(f1_score_micro) == 0 else np.around(np.mean(f1_score_micro),
                                                                                               4),
                             "tempo_loss": tempo_loss}

            if self.is_nni is False:
                print(f"best model obtained after {self.best_epoch} epochs")
        return test_res_dict, self.best_epoch
