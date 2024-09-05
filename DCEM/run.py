import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import dcem
import sys
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DCEM_run(
        model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
        tune_times=500, feature_T="", feature_A="", feature_V="",
        model_save_dir="", res_save_dir="", log_dir="",
        gpu_ids=[0], num_workers=4, verbose_level=1, mode='', is_distill=False
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:  # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_distill = is_distill  # use or not use distill, train use, test not use
    args.mode = mode  # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        # print(result)
        model_results.append(result)
    if args.is_distill:
        csv_file = res_save_dir / f"{dataset_name}.csv"
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=model_results[0].keys())
            writer.writeheader()
            for row in model_results:
                writer.writerow(row)
        logger.info(f"Results saved to {csv_file}.")


def T_SNE(data, label):
    all_color = [(99 / 255, 178 / 255, 238 / 255), (118 / 255, 218 / 255, 145 / 255),
                     (248 / 255, 203 / 255, 127 / 255)]

    label = np.array(label.cpu()).astype(int)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    data = result
    fig = plt.figure()
    for i in range(len(data)):
        if label[i] == 0:
            continue
        plt.scatter(data[i][0], data[i][1], color=all_color[label[i] + 1])
    fig.patch.set_alpha(0.)
    plt.show()


def _run(args, num_workers=3, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)
    if args.is_distill:
        print("training for DCEM")
        model = []
        model_dmd = getattr(dcem, 'DCEM')(args)
        model_dmd = model_dmd.cuda()
        model = [model_dmd]
    else:
        print("testing phase for DCEM")
        model = getattr(dcem, 'DCEM')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)

    if args.mode == 'test':
        model.load_state_dict(torch.load(r'D:\数据集\CMU\my-pth\实验结果\最好结果\CMU_MOSI\unaligned\2\8-dmd.pth'))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # T_SNE(all_pred, true)
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load('pt/best-dmd.pth'))
        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results