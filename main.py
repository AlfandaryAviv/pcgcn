import argparse
import time
import pickle
# import numpy as np
import nni
import torch
import data_loader
import utils
from trainer import ModelRunner
import logging
logger = logging.getLogger("NNI_logger")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['DBLP', 'IMDB', 'Tmall'], type=str, default='DBLP')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.03854704614193816)
    parser.add_argument('--weight_decay', type=float, default=0.002)
    parser.add_argument('--temporal_penalty', default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--model_name', choices=['GCN', 'SSP', 'GCNII','GAT'], default='GCN')
    parser.add_argument('--loss_weights', choices=[None, '1/Njs', 'sqrt(N/Nj)'], default=None)
    parser.add_argument('--cuda_device_id', type=int, default=3)
    parser.add_argument('--timestamps', type=int, default=10, help="21 for DBLP, 10 for IMDB, 9 for Tmall")
    parser.add_argument('--num_classes', type=int, default=14,
                        help="14 for DBLP (15 including 0), 11 for IMDB, 2 for Tmall")
    parser.add_argument('--hid_size', default=10)
    parser.add_argument('--is_NNI', default=False)
    parser.add_argument('--preconditioner', type=str, default=None)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=128)
    parser.add_argument('--shared_weights', type=bool, default=True)
    parser.add_argument('--hyperparam', type=str, default=None)
    args = parser.parse_args()

    try:
        t = time.time()
        loader_list = data_loader.get_pickled_data_list(args.data_name, args.timestamps, 0)
        loader = data_loader.set_data_to_new_device(loader_list, args.cuda_device_id)
        print(f"data list loaded: {time.time() - t}")

    except FileNotFoundError:
        dl = data_loader.DatasetLoader(data_name=args.data_name, timestamps=args.timestamps, Nj=args.loss_weights,cuda_device=args.cuda_device_id)
        loader = dl.my_data

    kwargs = dict(loader=loader,
                  data_name=args.data_name,
                  model_name=args.model_name,
                  optimizer=args.optimizer,
                  str_preconditioner=args.preconditioner,
                  iterations=args.trials,
                  loss_weights_type=args.loss_weights,
                  epochs=args.epochs,
                  hid_size=args.hid_size,
                  temporal_pen=args.temporal_penalty,
                  timestamps=args.timestamps,
                  number_of_classes=args.num_classes,
                  cuda_device=args.cuda_device_id,
                  lr=args.lr,
                  weight_decay=args.weight_decay,
                  dropout=args.dropout,
                  early_stopping_patience=args.early_stopping,
                  is_nni=args.is_NNI,
                  theta=args.theta,
                  num_layers=args.num_layers,
                  #ssp
                  lamda=args.lamda,
                  # momentum=args.momentum,
                  eps=args.eps,
                  update_freq=args.update_freq,
                  gamma=args.gamma,
                  alpha=args.alpha,
                  hyperparam=args.hyperparam)

    grid_logger, grid_logger_avg = utils.set_logs(args.data_name)

    run_trial(kwargs, grid_logger, grid_logger_avg, cuda_id=args.cuda_device_id)


def run_trial(params, grid_logger=None, grid_logger_avg=None, cuda_id=0):
    if not params["is_nni"]:
        results_logger = utils.Results(grid_logger, grid_logger_avg, params)

    for it in range(params['iterations']):
        print("Starting Trial")
        print(params)

        dataset = params['loader']
        model = ModelRunner(params, dataset, cuda_device=cuda_id)
        model.architecture()

        early_stopping = utils.EarlyStopping(patience=params['early_stopping_patience'], verbose=True)

        for epoch in range(int(params['epochs'])):

            train_results = model.train(epoch)                # train
            valid_results = model.validation(epoch)           # validation

            if params["is_nni"]:
                if epoch % 1 == 0:
                    nni.report_intermediate_result(train_results["f1_score_macro"])

            if not params['is_nni']:
                results_logger.insert_scores(train_results=train_results, valid_results=valid_results)
                utils.print_log_data(train_results=train_results, valid_results=valid_results, epoch=epoch)

            if epoch == int(params['epochs'])-1:
                test_results, best_epoch = model.test()        # test
                if params["is_nni"]:
                    nni.report_final_result(valid_results["f1_score_macro"])

                else:
                    results_logger.insert_scores(test_results=test_results)
                    utils.print_log_data(train_results=train_results, valid_results=valid_results, epoch=epoch)
                    results_logger.write_log(it, best_epoch)
                    results_logger.write_avg_log()

            valid_loss = valid_results['loss'] + valid_results['tempo_loss']
            early_stopping(valid_loss, model.net)
            if early_stopping.early_stop:
                print(f"Early stopping, epoch:{epoch}")
                break

        print("done")

def main_nni(data_name,cuda_device):

    if data_name == 'DBLP':
        timestamps = 21
        num_classes = 14
    elif data_name == 'IMDB':
        timestamps = 10
        num_classes = 11
    else:
        timestamps = 9
        num_classes = 2

    dl = data_loader.DatasetLoader(data_name=data_name, timestamps=timestamps, Nj="loss")
    loader = dl.my_data


    try:
        params = nni.get_next_parameter()
        logger.debug(params)

        params.update({"loader": loader, "data_name": data_name, "timestamps": timestamps,
                       "number_of_classes": num_classes, "is_nni": True, "early_stopping_patience": 20})

        if params["loss_weights_type"] == "None":
            params["loss_weights_type"] = None

        keys = ["str_preconditioner", "theta", "num_layers", "lamda", "eps", "update_freq", "gamma", "alpha", "hyperparam"]
        for key in keys:
            if params.get(key) is None:
                params.update({key: None})

        run_trial(params)

    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == '__main__':
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # is_nni = True
    is_nni = False

    if is_nni:
        data_name = "IMDB"
        cuda_id = 0
        main_nni(data_name, cuda_id)
    else:
        main()