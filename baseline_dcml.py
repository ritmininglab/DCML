# Base code is from https://github.com/cs230-stanford/cs230-code-examples
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
from evaluate import evaluate

import pickle
import random


def train_single_task(model, task_lr, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.

    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of
                     support set and query set
        params: (Params) hyperparameters
    """
    # extract params
    #loss_fn = nn.NLLLoss()
    num_train_updates = params.num_train_updates

    # set model to training mode
    model.train()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_sup2, Y_sup2 = dl_sup.__iter__().next()

    # move to GPU if available
    if params.cuda:
        X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()

    # compute model output and loss
    Y_sup_hat = model(X_sup)
    loss = loss_fn(Y_sup_hat, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    # optimizer.zero_grad()
    # loss.backward(create_graph=True)
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(1, num_train_updates):
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        zero_grad(adapted_params.values())
        # optimizer.zero_grad()
        # loss.backward(create_graph=True)
        grads = torch.autograd.grad(
            loss, adapted_params.values(), create_graph=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict


def train_and_evaluate(args, model,
                       meta_train_classes,
                       meta_test_classes,
                       meta_val_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,loss_fn_spl,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.
    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_train_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, meta_optimizer)

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_inner_tasks = params.num_inner_tasks
    task_lr = params.task_lr
    meta_lr = params.meta_lr

    # TODO validation accuracy
    best_val_acc = 0.0

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'loss': [],
        'id': []
    }
    ckpt_n = 0
    model_list=[]
    total_class_num=len(meta_train_classes)
    class_percent=float(args.class_start_percent)
    data_set_path=os.path.join(meta_train_classes[0].split('/')[0],meta_train_classes[0].split('/')[1],meta_train_classes[0].split('/')[2])#,meta_train_classes[0].split('/')[2]
    meta_train_classes_all=meta_train_classes
    threshold_e=float(args.threshold_e)
    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            if args.CMAML and episode % int(args.update_curriculum_steps_E) == 0 and episode > int(args.warm_start):
                loss_fn_spl.increase_threshold()
            for n_task in range(num_inner_tasks):
                task = task_type(meta_train_classes, num_classes, num_samples,
                                 num_query)
                dataloaders = fetch_dataloaders(['train', 'test', 'meta'],
                                                task)
                # Perform a gradient descent to meta-learner on the task

                if episode>int(args.warm_start):
                    a_dict = train_single_task(model, task_lr, loss_fn_spl,
                                               dataloaders, params)
                else:
                    a_dict = train_single_task(model, task_lr, loss_fn,
                                           dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)

            # Update the parameters of meta-learner
            # Compute losses with adapted parameters along with corresponding tasks
            # Updated the parameters of meta-learner using sum of the losses
            meta_loss = 0
            for n_task in range(num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['test']
                X_meta, Y_meta = dl_meta.__iter__().next()
                if params.cuda:
                    X_meta, Y_meta = X_meta.cuda(), Y_meta.cuda()

                a_dict = adapted_state_dicts[n_task]
                Y_meta_hat = model(X_meta, a_dict)
                #if args.CMAML:
                 #   loss_t = loss_fn(Y_meta_hat, Y_meta)
                if episode > int(args.warm_start):
                    loss_t = loss_fn_spl(Y_meta_hat, Y_meta)
                else:
                    loss_t = loss_fn(Y_meta_hat, Y_meta)
                # loss_nll = nn.NLLLoss()
                # loss_t_n = loss_nll(Y_meta_hat, Y_meta)
                # print('spl:',loss_t)
                # print('nll:', loss_t_n)
                meta_loss += loss_t
            meta_loss /= float(num_inner_tasks)
            # print(meta_loss.item())

            # Meta-update using meta_optimizer
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % args.save_summary_steps == 0:
                train_metrics, loss_summ, image_id = evaluate(model, loss_fn, meta_train_classes,
                                                              task_lr, task_type, metrics, params,
                                                              'train')
                test_metrics, _, _ = evaluate(model, loss_fn, meta_test_classes,
                                              task_lr, task_type, metrics, params,
                                              'test')
                val_metrics, _, _ = evaluate(model, loss_fn, meta_val_classes,
                                              task_lr, task_type, metrics, params,
                                              'val')

                train_loss = train_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                test_acc = test_metrics['accuracy']
                val_acc = val_metrics['accuracy']

                is_best = val_acc >= best_val_acc


                # Save weights
                utils.save_checkpoint({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': meta_optimizer.state_dict()
                },
                    is_best=is_best,
                    checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_val_acc = val_acc

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)
                plot_history['loss'].append(loss_summ)
                plot_history['id'].append(image_id)
                del loss_summ, image_id

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    val_acc='{:05.3f}'.format(val_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            if (episode + 1) % int(args.proxy_ckpt_steps) == 0 and episode<args.proxy_ckpt_upper_bound:
                ckpt_n=ckpt_n+1
                utils.save_checkpoint_proxy({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': meta_optimizer.state_dict()
                },
                    ckpt_n=ckpt_n,
                    checkpoint=model_dir)
                proxy_path=os.path.join('proxy'+str(ckpt_n)+'.pth.tar')
                model_list.append(proxy_path)
            if (episode + 1) % int(args.update_curriculum_steps) == 0 and episode>int(args.warm_start) and ckpt_n>2:
                curriculum_class_num=int(total_class_num*class_percent)
                #print(class_percent * total_class_num)

                if args.class_metrics=='cp':
                    class_mean_summary, class_var_summary = dcml.update_curriculum_cp(model_list, model, meta_optimizer,
                                                                                      loss_fn, meta_train_classes_all,
                                                                                      task_lr,
                                                                                      task_type, metrics, params,
                                                                                      'train', args)
                    class_mean_avg = {k: np.mean(v) for k, v in class_mean_summary.items()}
                    class_var_avg = {k: np.mean(v) for k, v in class_var_summary.items()}
                    class_mean_sorted = {k: v for k, v in sorted(class_mean_avg.items(), key=lambda item: item[1])}
                    class_var_sorted = {k: v for k, v in sorted(class_var_avg.items(), key=lambda item: item[1])}
                    del class_mean_summary, class_var_summary, class_mean_avg, class_var_avg
                    meta_clean_classes = [k for k in class_mean_sorted]
                    high_var_class_dict = dict(list(class_var_sorted.items())[args.class_select_num:])
                    high_mean_class_dict = dict(list(class_mean_sorted.items())[args.class_select_num:])
                    noisy_class = []  # high mean and high variance
                    for aclass in high_var_class_dict:
                        for bclass in high_mean_class_dict:
                            if aclass == bclass:
                                # meta_clean_classes.remove(aclass)
                                noisy_class.append(aclass)
                    #print('noisy_class', noisy_class)
                    if len(noisy_class) > args.remove_class_num_upperb:
                        remove_class = noisy_class[:args.remove_class_num_upperb]
                    else:
                        remove_class = noisy_class

                    for aclass in remove_class:
                        meta_clean_classes.remove(aclass)
                    meta_clean_classes = [os.path.join(data_set_path, k) for k in meta_clean_classes]
                    meta_train_classes = meta_clean_classes[0:curriculum_class_num]
                elif args.class_metrics=='cp_asy':
                    class_mean_summary, class_var_summary = dcml.update_curriculum_cp(model_list, model, meta_optimizer,
                                                                                      loss_fn, meta_train_classes_all,
                                                                                      task_lr,
                                                                                      task_type, metrics, params,
                                                                                      'train', args)
                    class_mean_avg = {k: np.mean(v) for k, v in class_mean_summary.items()}
                    class_var_avg = {k: np.mean(v) for k, v in class_var_summary.items()}
                    class_mean_sorted = {k: v for k, v in sorted(class_mean_avg.items(), key=lambda item: item[1])}
                    class_var_sorted = {k: v for k, v in sorted(class_var_avg.items(), key=lambda item: item[1])}
                    del class_mean_summary, class_var_summary, class_mean_avg, class_var_avg
                    meta_clean_classes = [k for k in class_mean_sorted]
                    # class_set=[k for k in class_mean_sorted]
                    # low_var_class_dict=dict(list(class_var_sorted.items())[:curriculum_class_num])
                    # high_mean_class_dict =dict(list(class_mean_sorted.items())[curriculum_class_num:])
                    low_var_class_dict = dict(list(class_var_sorted.items())[:args.class_select_num])
                    high_mean_class_dict = dict(list(class_mean_sorted.items())[args.class_select_num:])
                    similar_class = []  # high mean and low variance
                    for aclass in low_var_class_dict:
                        for bclass in high_mean_class_dict:
                            if aclass == bclass:
                                # meta_clean_classes.remove(aclass)
                                similar_class.append(aclass)
                    #print('similar_class', similar_class)
                    if len(similar_class) > args.remove_class_num_upperb:
                        remove_class = similar_class[:args.remove_class_num_upperb]
                    else:
                        remove_class = similar_class
                    for aclass in remove_class:
                        meta_clean_classes.remove(aclass)
                    meta_clean_classes = [os.path.join(data_set_path, k) for k in meta_clean_classes]
                    meta_train_classes = meta_clean_classes[0:curriculum_class_num]
                elif args.class_metrics=='loss':
                    class_loss = dcml.update_curriculum_loss(model, loss_fn, meta_train_classes_all, task_lr,
                                                             task_type, metrics, params, 'train', args)
                    class_loss_sorted = {k: v for k, v in sorted(class_loss.items(), key=lambda item: item[1])}
                    del class_loss
                    meta_train_classes = [os.path.join(data_set_path, k) for k in class_loss_sorted][
                                         0:curriculum_class_num]

                print('curriculum class number:',len(meta_train_classes))
                if class_percent<args.class_upper_bound:
                    class_percent = class_percent * args.growing_factor
                if episode>args.train_noise_class_step:
                    class_percent=1
                    meta_train_classes=meta_train_classes_all

            t.update()
            file_path = os.path.join(args.model_dir, 'dcml_seed' + str(params.SEED))
            f = open(file_path, 'wb')
            pickle.dump(plot_history, f)

    utils.plot_training_results(args.model_dir, plot_history)


def main(args):
    # if __name__ == '__main__':
    # Load the parameters from json file
    # args = parser.parse_args()
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

    # Define the model and optimizer
    if params.cuda:
        model = MetaLearner(params).cuda()
    else:
        model = MetaLearner(params)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    loss_fn_spl = dcml.SPLLoss(args=args,n_samples=params.num_query, n_classes=params.num_classes)
    model_metrics = metrics

    # Train the model
    logging.info("Starting training for {} episode(s)".format(num_episodes))
    train_and_evaluate(args, model, meta_train_classes, meta_test_classes,meta_val_classes, task_type,
                       meta_optimizer, loss_fn,loss_fn_spl, model_metrics, params,
                       args.model_dir, args.restore_file)