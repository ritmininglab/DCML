# -*- coding: utf-8 -*-
#
import argparse
import os
import logging
from collections import OrderedDict
import numpy as np
import itertools
import torch
import torch.nn as nn
import utils
import src.dcml as dcml
from torch.autograd import Variable
from tqdm import tqdm
from src.model import MetaLearner
from src.model import Net
from src.model import metrics
from src.data_loader import split_omniglot_characters
from src.data_loader import load_imagenet_images
from src.data_loader import OmniglotTask, FC100Task
from src.data_loader import ImageNetTask
# from src.data_loader_level import FC100Task
from src.data_loader import fetch_dataloaders
from evaluate import evaluate,evaluate_CT,evaluate_sym,evaluate_noise
import json
import pickle
import random
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    # default='data/FC100',
    # default='data/miniImageNet',
    # default='data/food101FS',
    default='data/CIFAR-100N',
    # default='data/miniweb',
    #default='data/CUB',
    #default='data/Omniglot',
    help="Directory containing the dataset")
parser.add_argument('--dataset',default='FC100')
parser.add_argument(
    '--model_dir',
   # default='experiments/cn_dccp',
    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default='best',
    help="Optional, name of the file in --model_dir containing weights to \
          reload before training")  # 'best' or 'last'
parser.add_argument('--warm_start',default=1)
parser.add_argument('--gamma',default=0.5)
parser.add_argument('--noise_type',default='real')
parser.add_argument('--proxy_ckpt_steps',default=1)
parser.add_argument('--proxy_ckpt_upper_bound',default=30000)
parser.add_argument('--save_summary_steps',default=500)
parser.add_argument('--update_curriculum_steps',default=1)
parser.add_argument('--num_curri_steps',default=100)
parser.add_argument('--class_start_percent',default=0.8)
parser.add_argument('--class_upper_bound',default=0.98)
parser.add_argument('--growing_factor',default=1.5)
parser.add_argument('--E-CL',default=True)
parser.add_argument('--eaxample_threshold_percent',default=0.8)
parser.add_argument('--threshold_e',default=0.8)
parser.add_argument('--class_select_num',default=30)
parser.add_argument('--remove_class_num_upperb',default=0)
parser.add_argument('--train_noise_class_step',default=38000)
parser.add_argument('--class_metrics',default='cp',help='cp, cp_asy,loss')
parser.add_argument('--use_CT',default=False,help='use channel tranform during test time')
parser.add_argument('--use_sample',default=True,help='use channel tranform during test time')
parser.add_argument('--test_sym',default=False,help='use to create asmmetric label noise')
parser.add_argument('--test_noise',default=False,help='use to create asmmetric label noise')
parser.add_argument('--test_noise_percent',default=0.4,help='use to create asmmetric label noise')
parser.add_argument('--test_noise_sample',default=False,help='use to create asmmetric label noise')
if __name__ == '__main__':

    args = parser.parse_args()
    print(args.model_dir)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    SEED = params.SEED
    meta_lr = params.meta_lr
    num_episodes = params.num_episodes

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # NOTE These params are only applicable to pre-specified model architecture.
    # Split meta-training and meta-testing characters
    if ('Omniglot' or 'omniglot' in args.data_dir) and params.dataset == 'Omniglot':
        params.in_channels = 1
        meta_train_classes, meta_val_classes, meta_test_classes = load_imagenet_images(
            args.data_dir, args.noise_type)
        task_type = OmniglotTask
    else:
        params.in_channels = 3
        meta_train_classes, meta_val_classes, meta_test_classes = load_imagenet_images(
            args.data_dir, args.noise_type)
        task_type = ImageNetTask
    # else:
    #     raise ValueError("I don't know your dataset")
    loss_fn = nn.NLLLoss()
    loss_fn_spl = dcml.SPLLoss(args=args, n_samples=params.num_query, n_classes=params.num_classes)
    if params.cuda:
        model = MetaLearner(params).cuda()
    else:
        model = MetaLearner(params)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=params.meta_lr)
    restore_path = os.path.join(args.model_dir,
                                args.restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, meta_optimizer)
    if args.use_CT:
        test_metrics, _, _ = evaluate_CT(model, loss_fn, meta_test_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test')
    elif args.test_noise_sample:
        test_metrics, _, _ = evaluate_noise(model, loss_fn_spl, meta_test_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test',args.test_noise_percent)                                  
    elif args.use_sample:
        test_metrics, _, _ = evaluate(model, loss_fn_spl, meta_test_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test')
    elif args.test_noise:
        test_metrics, _, _ = evaluate_noise(model, loss_fn, meta_test_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test',args.test_noise_percent)    
    else:
        test_metrics, _, _ = evaluate(model, loss_fn, meta_test_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test')

    test_loss = test_metrics['loss']
    test_acc = test_metrics['accuracy']
    print('test_acc=',test_acc,'test_loss=',test_loss)
    if args.test_sym:
        class_summary= evaluate_sym(model, loss_fn, meta_train_classes,
                                      params.task_lr, task_type, metrics, params,
                                      'test')
        print(class_summary)
        # create json object from dictionary
        json = json.dumps(class_summary)

        # open file for writing, "w"
        f = open("class_summary.json", "w")

        # write json object to file
        f.write(json)

        # close file
        f.close()

