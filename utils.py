import numpy as np
from sklearn.metrics import f1_score
import torch
import time
import os
import matplotlib.pyplot as plt

from loggers import CSVLogger, PrintLogger, EmptyLogger


def intersection_inds(tensor1, tesnor2):
    combined = torch.cat((tensor1, tesnor2))
    uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection


def f1_score_func(output, labels_all):
    labels = labels_all[labels_all.sum(dim=1) != 0]  # removes labels rows if all 0

    output = output[
        labels_all.sum(dim=1) != 0]  # removes output rows if all 0, we dont cate about them (DBLP will use it)

    real = labels.argmax(1)
    pred = output.argmax(1)

    f1_macro = f1_score(real.cpu().numpy(), pred.cpu().numpy(), average='macro')
    f1_micro = f1_score(real.cpu().numpy(), pred.cpu().numpy(), average='micro')

    return f1_macro, f1_micro, real, pred


def confusion_matrix(list_real, list_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes))  # classes X classes

    for i in range(len(list_pred)):
        matrix[list_real[i], list_pred[i]] += 1

    row_sums = matrix.sum(axis=1, dtype='float')

    new_matrix = np.zeros((num_classes, num_classes))  # classes X classes

    for i, (row, row_sum) in enumerate(zip(matrix, row_sums)):
        if row_sum == 0:
            new_matrix[i, :] = 0
        else:
            new_matrix[i, :] = row / row_sum

    new_matrix = np.around(new_matrix, 3)
    b = np.asarray(new_matrix)

    diag_sum = np.trace(b)
    diag_elements = np.diagonal(b)

    print('Diagonal (sum): ', np.trace(b))
    print('Diagonal (elements): ', np.diagonal(b))

    return diag_sum, diag_elements


def set_logs(data_name):
    grid_outputs_folder = time.strftime("%Y%m%d_%H%M%S")

    res_path = os.path.join(os.getcwd(), "dataset", data_name, "grid", grid_outputs_folder)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    grid_logger = CSVLogger("results_%s" % 'grid' + time.strftime("%Y%m%d_%H%M%S"), path=res_path)
    grid_logger_avg = CSVLogger("results_%s" % 'grid_it_avg' + time.strftime("%Y%m%d_%H%M%S"), path=res_path)

    grid_logger.info("data_name", "weights", "iteration", "total_it", "lr", "do", "wd", "hid_size", "temp_pen",
                     "epochs",
                     "best_epoch", "train_loss", "train_temp_loss", "total_train_loss", "train_f1_macro",
                     "train_f1_micro", "val_loss", "val_temp_loss", "total_val_loss", "val_f1_macro", "val_f1_micro",
                     "test_loss", "test_temp_loss", "total_test_loss", "test_f1_macro", "test_f1_micro", "diag_sum",
                     "diag_elements")

    grid_logger_avg.info("data_name", "weights", "iterations", "lr", "do", "hid_size", "wd", "temp_pen", "epochs",
                         "train_loss", "train_temp_loss", "total_train_loss", "train_f1_macro_mean",
                         "train_f1_macro_std", "train_f1_micro_mean", "train_f1_micro_std",
                         "val_loss", "val_temp_loss", "total_val_loss", "val_f1_macro_mean", "val_f1_macro_std",
                         "val_f1_micro_mean", "val_f1_micro_std",
                         "test_loss", "test_temp_loss", "total_test_loss", "test_f1_macro_mean", "test_f1_macro_std",
                         "test_f1_micro_mean", "test_f1_micro_std", "diag_sum", "diag_elements")

    return grid_logger, grid_logger_avg


def print_log_data(train_results, valid_results, test_results=None, epoch=None):
    if test_results is None:
        PrintLogger().debug('Epoch: {:04d} '.format(epoch + 1) +
                            'loss_train: {:.4f} '.format(train_results['loss']) +
                            'temp_loss_train: {:.4f} '.format(train_results['tempo_loss']) +
                            'f1_macro_train: {} '.format(train_results['f1_score_macro']) +
                            'f1_micro_train {} '.format(train_results['f1_score_micro']) +
                            'loss_valid: {:.4f} '.format(valid_results['loss']) +
                            'temp_loss_valid: {:.4f} '.format(valid_results['tempo_loss']) +
                            'f1_macro_valid: {} '.format(valid_results['f1_score_macro']) +
                            'f1_micro_valid {} '.format(valid_results['f1_score_micro']))
    else:
        PrintLogger().debug('loss_train: {:.4f} '.format(train_results['loss'].item()) +
                            'temp_loss_train: {:.4f} '.format(train_results['tempo_loss'].item()) +
                            'f1_macro_train: {} '.format(train_results['f1_score_macro']) +
                            'f1_macro_train {} '.format(train_results['f1_score_micro']) +
                            'loss_valid: {:.4f} '.format(valid_results['loss'].item()) +
                            'temp_loss_valid: {:.4f} '.format(valid_results['tempo_loss'].item()) +
                            'f1_macro_valid: {} '.format(valid_results['f1_score_macro']) +
                            'f1_macro_valid {} '.format(valid_results['f1_score_micro']) +
                            'reg_loss_test: {:.4f} '.format(test_results['loss'].item()) +
                            'temp_loss_test: {:.4f} '.format(test_results['tempo_loss'].item()) +
                            'f1_macro_test: {} '.format(test_results['f1_score_macro']) +
                            'f1_micro_test {} '.format(test_results['f1_score_micro']))


class Results():
    def __init__(self, grid_logger, grid_logger_avg, params):
        self.curr_train_res = None
        self.curr_test_res = None
        self.curr_val_res = None

        self.train_losses = []
        self.train_tempo_losses = []
        self.train_total_loss = []

        self.test_losses = []
        self.test_tempo_losses = []
        self.test_total_loss = []

        self.valid_losses = []
        self.valid_tempo_losses = []
        self.valid_total_loss = []

        self.train_f1_macro = []
        self.train_f1_micro = []

        self.test_f1_macro = []
        self.test_f1_micro = []

        self.valid_f1_macro = []
        self.valid_f1_micro = []

        self.diag_sum = []
        self.diag_elements = []

        self.grid_logger = grid_logger
        self.grid_logger_avg = grid_logger_avg

        self.params = params

    def insert_scores(self, train_results=None, valid_results=None, test_results=None):

        if train_results is not None:
            self.curr_train_res = train_results
            self.train_losses.append(train_results['loss'])
            self.train_tempo_losses.append(train_results['tempo_loss'])
            self.train_total_loss.append(train_results['loss'] + train_results['tempo_loss'])
            self.train_f1_macro.append(train_results['f1_score_macro'])
            self.train_f1_micro.append(train_results['f1_score_micro'])

        if valid_results is not None:
            self.curr_val_res = valid_results
            self.valid_losses.append(valid_results['loss'])
            self.valid_tempo_losses.append(valid_results['tempo_loss'])
            self.valid_total_loss.append(valid_results['loss'] + valid_results['tempo_loss'])
            self.valid_f1_macro.append(valid_results['f1_score_macro'])
            self.valid_f1_micro.append(valid_results['f1_score_micro'])

        if test_results is not None:
            self.curr_test_res = test_results
            self.test_losses.append(test_results["loss"])
            self.test_f1_macro.append(test_results["f1_score_macro"])
            self.test_f1_micro.append(test_results["f1_score_micro"])
            self.test_tempo_losses.append(test_results["tempo_loss"])
            self.test_total_loss.append(test_results["loss"] + test_results["tempo_loss"])


    def write_log(self, it, best_epoch):
        if self.curr_test_res is not None:
            self.grid_logger.info(self.params['data_name'],
                                  self.params['loss_weights_type'],
                                  it,
                                  self.params['iterations'],
                                  self.params['lr'],
                                  self.params["dropout"],
                                  self.params["weight_decay"],
                                  self.params["hid_size"],
                                  self.params["temporal_pen"],
                                  self.params["epochs"],
                                  best_epoch,

                                  np.around(self.curr_train_res['loss'].item(), 3),
                                  np.around(self.curr_train_res['tempo_loss'].item(), 3),
                                  np.around(
                                      self.curr_train_res['loss'].item() + self.curr_train_res['tempo_loss'].item(), 3),
                                  np.around(np.mean(self.curr_train_res['f1_score_macro']), 3),
                                  np.around(np.mean(self.curr_train_res['f1_score_micro']), 3),

                                  np.around(self.curr_val_res["loss"].item(), 3),
                                  np.around(self.curr_val_res['tempo_loss'].item(), 3),
                                  np.around(self.curr_val_res["loss"].item() + self.curr_val_res['tempo_loss'].item(),
                                            3),
                                  np.around(np.mean(self.curr_val_res['f1_score_macro']), 3),
                                  np.around(np.mean(self.curr_val_res['f1_score_micro']), 3),

                                  np.around(self.curr_test_res["loss"].item(), 3),
                                  np.around(self.curr_test_res['tempo_loss'].item(), 3),
                                  np.around(self.curr_test_res["loss"].item() + self.curr_test_res['tempo_loss'].item(),
                                            3),
                                  np.around(self.curr_test_res["f1_score_macro"], 3),
                                  np.around(self.curr_test_res["f1_score_micro"], 3),

                                  )
        else:
            self.grid_logger.info(self.params['data_name'],
                                  self.params['loss_weights_type'],
                                  it,
                                  self.params['iterations'],
                                  self.params['lr'],
                                  self.params["dropout"],
                                  self.params["weight_decay"],
                                  self.params["hid_size"],
                                  self.params["temporal_pen"],
                                  self.params["epochs"],
                                  best_epoch,

                                  np.around(self.curr_train_res['loss'].item(), 3),
                                  np.around(self.curr_train_res['tempo_loss'].item(), 3),
                                  np.around(
                                      self.curr_train_res['loss'].item() + self.curr_train_res['tempo_loss'].item(), 3),
                                  np.around(np.mean(self.curr_train_res['f1_score_macro']), 3),
                                  np.around(np.mean(self.curr_train_res['f1_score_micro']), 3),

                                  np.around(self.curr_val_res["loss"].item(), 3),
                                  np.around(self.curr_val_res['tempo_loss'].item(), 3),
                                  np.around(self.curr_val_res["loss"].item() + self.curr_val_res['tempo_loss'].item(),
                                            3),
                                  np.around(np.mean(self.curr_val_res['f1_score_macro']), 3),
                                  np.around(np.mean(self.curr_val_res['f1_score_micro']), 3),

                                  )

    def write_avg_log(self):
        self.grid_logger_avg.info(self.params['data_name'],
                                  self.params['loss_weights_type'],
                                  self.params['iterations'],
                                  self.params['lr'],
                                  self.params["dropout"],
                                  self.params["hid_size"],
                                  self.params["weight_decay"],
                                  self.params["temporal_pen"],
                                  self.params["epochs"],
                                  np.around(torch.mean(torch.stack(self.train_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.train_tempo_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.train_total_loss)).item(), 3),
                                  np.around(np.mean(self.train_f1_macro), 3),
                                  np.around(np.std(self.train_f1_macro), 3),
                                  np.around(np.mean(self.train_f1_micro), 3),
                                  np.around(np.std(self.train_f1_micro), 3),

                                  np.around(torch.mean(torch.stack(self.valid_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.valid_tempo_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.valid_total_loss)).item(), 3),

                                  np.around(np.mean(self.valid_f1_macro), 3),
                                  np.around(np.std(self.valid_f1_macro), 3),
                                  np.around(np.mean(self.valid_f1_micro), 3),
                                  np.around(np.std(self.valid_f1_micro), 3),

                                  np.around(torch.mean(torch.stack(self.test_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.test_tempo_losses)).item(), 3),
                                  np.around(torch.mean(torch.stack(self.test_total_loss)).item(), 3),


                                  np.around(np.mean(self.test_f1_macro), 3),
                                  np.around(np.std(self.test_f1_macro), 3),
                                  np.around(np.mean(self.test_f1_micro), 3),
                                  np.around(np.std(self.test_f1_micro), 3),

                                  )



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# def plot_early_stopping(train_loss, valid_loss):
#     # visualize the loss as the network trained
#     fig = plt.figure(figsize=(10, 8))
#     plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
#     plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
#
#     # find position of lowest validation loss
#     minposs = valid_loss.index(min(valid_loss)) + 1
#     plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
#
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.ylim(0, 0.5)  # consistent scale
#     plt.xlim(0, len(train_loss) + 1)  # consistent scale
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     # fig.savefig('loss_plot.png', bbox_inches='tight')