# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import logging
import copy
import torch.nn.functional as F
import torch
import numpy as np
from collections import OrderedDict
from src.data_loader import fetch_dataloaders
import torch.nn as nn
import src.dcml as dcml
def simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.max(x, zero_tensor)
    x_neg = torch.min(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg
def evaluate(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    loss_summ=[]
    image_id=[]
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()


        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
            X_que, Y_que = X_que.cuda(), Y_que.cuda()

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)


        query_example_loss = nn.functional.nll_loss(Y_que_hat, Y_que, reduction="none")
        query_image_id = dl_que.dataset.filenames
        loss_summ.append(query_example_loss.data.cpu().numpy())
        image_id.append(query_image_id)
        del query_example_loss, query_image_id
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()



        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return metrics_mean, loss_summ, image_id


def evaluate_CT(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split):
    """
    Evaluate the model on `num_steps` batches.

    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    loss_summ = []
    image_id = []
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        X_sup = simple_transform(X_sup, 1.3)
        X_que = simple_transform(X_que, 1.3)

        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
            X_que, Y_que = X_que.cuda(), Y_que.cuda()

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)

        query_example_loss = nn.functional.nll_loss(Y_que_hat, Y_que, reduction="none")
        query_image_id = dl_que.dataset.filenames
        loss_summ.append(query_example_loss.data.cpu().numpy())
        image_id.append(query_image_id)
        del query_example_loss, query_image_id
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()



        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return metrics_mean, loss_summ, image_id


def evaluate_sym(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split):
    """
    Evaluate the model on `num_steps` batches.

    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    loss_summ = []
    image_id = []
    class_summary = {class_id.split('/')[-1]: []
                     for class_id in meta_classes}
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
            X_que, Y_que = X_que.cuda(), Y_que.cuda()

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)

        query_example_loss = nn.functional.nll_loss(Y_que_hat, Y_que, reduction="none")
        query_image_id = dl_que.dataset.filenames
        loss_summ.append(query_example_loss.data.cpu().numpy())
        image_id.append(query_image_id)
        del query_example_loss, query_image_id
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()
        ####sym test#####

        Y_que_hat = np.argmax(Y_que_hat, axis=1)
        for i in range(len(Y_que_hat)):
            if Y_que_hat[i] != Y_que[i]:
                true_class_name = dl_sup.dataset.filenames[Y_que[i]].split('/')[-2]
                # print("true class:", true_class_name)
                pred_class_name = dl_sup.dataset.filenames[Y_que_hat[i]].split('/')[-2]
                # print("predicted class:", pred_class_name)
                dcml.add_value(class_summary,true_class_name,pred_class_name)



    return class_summary
    
def evaluate_noise(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,percent):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    loss_summ=[]
    image_id=[]
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        dcml.meta_label_noise(Y_sup, num_classes, dl_sup, meta_classes, percent)
        X_que, Y_que = dl_que.__iter__().next()
        dcml.meta_label_noise(Y_que, num_classes, dl_que, meta_classes, percent)


        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
            X_que, Y_que = X_que.cuda(), Y_que.cuda()

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)


        query_example_loss = nn.functional.nll_loss(Y_que_hat, Y_que, reduction="none")
        query_image_id = dl_que.dataset.filenames
        loss_summ.append(query_example_loss.data.cpu().numpy())
        image_id.append(query_image_id)
        del query_example_loss, query_image_id
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()



        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return metrics_mean, loss_summ, image_id
if __name__ == '__main__':
    # TODO Evaluate trained model.
    pass