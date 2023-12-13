import logging
import copy

import torch
import numpy as np
from collections import OrderedDict
from src.data_loader import fetch_dataloaders
import torch.nn as nn
import os
import utils
import torch.nn.functional as nnf
import random


def set_random_seed(seed, deterministic=False):
    """Set random seed just for debug.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def train_single_task(model, task_lr, loss_fn, dataloaders, params, tr_classes, args, criterion):
#     """
#     Train the model on a single few-shot task.
#     We train the model with single or multiple gradient update.
#
#     Args:
#         model: (MetaLearner) a meta-learner to be adapted for a new task
#         loss_fn: a loss function
#         dataloaders: (dict) a dict of DataLoader objects that fetches both of
#                      support set and query set
#         params: (Params) hyperparameters
#     """
#     # set model to training mode
#     model.train()
#
#     # support set for a single few-shot task
#     dl_sup = dataloaders['train']
#     idx, X_sup, Y_sup = dl_sup.__iter__().next()
#
#     # move to GPU if available
#     if params.cuda:
#         X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()
#
#     # compute model output and loss
#     Y_sup_hat = model(X_sup)
#     # loss = criterion(Y_sup_hat, Y_sup)
#
#     loss = loss_fn(Y_sup_hat, Y_sup)
#
#     # clear previous gradients, compute gradients of all variables wrt loss
#     def zero_grad(params):
#         for p in params:
#             if p.grad is not None:
#                 p.grad.zero_()
#
#     # NOTE if we want approx-MAML, change create_graph=True to False
#     zero_grad(model.parameters())
#     grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#
#     # performs updates using calculated gradients
#     # we manually compute adpated parameters since optimizer.step() operates in-place
#     adapted_state_dict = model.cloned_state_dict()
#     adapted_params = OrderedDict()
#     for (key, val), grad in zip(model.named_parameters(), grads):
#         adapted_params[key] = val - task_lr * grad
#         adapted_state_dict[key] = adapted_params[key]
#
#     for _ in range(1, params.num_train_updates):
#         Y_sup_hat = model(X_sup, adapted_state_dict)
#         loss = loss_fn(Y_sup_hat, Y_sup)
#
#         zero_grad(adapted_params.values())
#         # optimizer.zero_grad()
#         # loss.backward(create_graph=True)
#         grads = torch.autograd.grad(
#             loss, adapted_params.values(), create_graph=True)
#         for (key, val), grad in zip(adapted_params.items(), grads):
#             adapted_params[key] = val - task_lr * grad
#             adapted_state_dict[key] = adapted_params[key]
#     return adapted_state_dict
#
#

class SPLLoss(nn.NLLLoss):
    def __init__(self, args, n_samples=0,n_classes=0):
        super(SPLLoss, self).__init__()
        self.threshold = float(args.threshold_e)
        self.growing_factor=args.growing_factor
        self.v = torch.zeros(n_samples*n_classes).float().cuda()
        self.n_classes = n_classes

    def forward(self, input, target):
        loss_fn = nn.NLLLoss()
        loss = loss_fn(input,target)
        res = 0
        v_l=[]
        index_l=[]
        for i in range(self.n_classes):
            index = (target == i).nonzero(as_tuple=True)[0]
            index_l.append(index)
            if not len(index) == 0:
                super_loss_group = nn.functional.nll_loss(input[index], target[index], reduction="none")
                v_group = self.spl_loss(super_loss_group)
                num = torch.sum(v_group > 0)
                res += (super_loss_group * v_group).mean() * len(v_group) / num
                # res += (super_loss_group * v_group).mean()
                v_l.append(v_group)
        res /= self.n_classes
        # index_x=torch.zeros(5)
        # v_t = torch.stack(v_l)
        # index_l = torch.stack(index_l)
        # index_v = v_t*index_l.float()
        # # for i in range(len(index_l)):
        # #     for j in range(len(v_l[i])):
        # #         if v_l[i][j]==1:
        # #             index_x[i]=index_l[i][j].long()
        # loss1 = loss_fn(input[index_v.long()],target[index_v.long()])
        # super_loss = nn.functional.nll_loss(input, target, reduction="none")
        # v = self.spl_loss(super_loss)
        # #v[index] = v
        # # print(super_loss)
        # num = torch.sum(v > 0)
        # res1=(super_loss * v).mean() * len(v) / num
        return res

    def increase_threshold(self):
        self.threshold *= float(self.growing_factor)

    def spl_loss(self, super_loss):

        sort_loss = (list(t) for t in zip(*sorted(zip(super_loss))))
        t = []
        #print('th<0.8',self.threshold)
        for loss_value in sort_loss:
            t.append(loss_value)
        if float(self.threshold)>0.8:
            self.threshold=0.8
            #print('th>0.8',self.threshold)
            #print('th>0.8',self.threshold.dtype())
        threshold_t = t[0][-int(len(t[0]) * float(self.threshold))]
        #threshold_t = t[0][-int(len(t[0]) * 1)]
        v = super_loss < threshold_t #+ 0.01
        # v = super_loss < self.threshold
        return v.float()

def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary,
        it will associate multiple values with that
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        # dict_obj[key].append(value)
        dict_obj[key]+=[value]
    else:
        dict_obj[key] = [dict_obj[key], value]
def update_curriculum_loss(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args):
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
    num_steps = args.num_curri_steps
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

        for i in range(len(query_image_id)):
            add_value(class_summary,query_image_id[i].split('/')[-2],query_example_loss[i].detach().cpu().numpy())
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
    class_mean = {
        k: np.mean(v)
        for k,v in class_summary.items()
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return class_mean

def update_curriculum_loss_history(model_list,model, meta_optimizer, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args):
    class_summary = {class_id.split('/')[-1]: []
                     for class_id in meta_classes}
    for proxy_model in model_list:
        proxy_path = os.path.join(args.model_dir,proxy_model)
        logging.info("Restoring proxy parameters from {}".format(proxy_path))
        utils.load_checkpoint(proxy_path, model, meta_optimizer)
        class_loss=update_curriculum_loss(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args)
        for key in class_loss:
            add_value(class_summary, key, class_loss[key])
        class_mean = {
            k: np.mean(v)
            for k, v in class_summary.items()
        }
        class_var = {
            k: np.std(v)
            for k, v in class_summary.items()
        }
    return class_summary,class_mean,class_var


def class_pair_metric(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args):
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = args.num_curri_steps
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
    # class_pair_summary={[c1.split('/')[-1],c2.split('/')[-1]]:[]
    #                for c1 in meta_classes for j in meta_classes if c2!=c1}
    cp=[]
    for i in meta_classes:
        c1=i.split('/')[-1]
        for j in meta_classes:
            if j!=i:
                c2=j.split('/')[-1]
                cp.append((c1,c2))
    class_pair_summary={cp1:[]
                        for cp1 in cp}
    class_summary = {class_id.split('/')[-1]: []
                     for class_id in meta_classes}
    class_var_summary = {class_id.split('/')[-1]: []
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
        query_example_prob=nnf.softmax(model(X_que), dim=1).cpu().detach().numpy()
        query_class_pair,query_cp_prob=[],[]
        
        for i in range(0, len(Y_que), num_query):
            j_list=[]
            ty2 = dl_que.dataset.filenames[i].split('/')[-2]
            for j in range(0, params.num_classes * params.num_query, params.num_query):
                #j_list.append(j)
                #print(j_list)
                #print(len(dl_que.dataset.filenames))
                if i != j:  # incorrect prediction from class i to class j
                    #print(dl_que.dataset.filenames[j].split('/'))
                    #print(dl_que.dataset.filenames[j])
                    #print(dl_que.dataset.filenames[j].split('/')[-2])
                    ty_p2 = dl_que.dataset.filenames[j].split('/')[-2]
                    query_class_pair.append((ty2,ty_p2))
                    query_cp_prob.append(np.mean(query_example_prob[i:i + num_query], 0)[int(j / num_query)])
        # query_image_id = dl_que.dataset.filenames
        # loss_summ.append(query_example_loss.data.cpu().numpy())
        # image_id.append(query_image_id)
        
        for i in range(len(query_class_pair)):
            add_value(class_pair_summary, query_class_pair[i], query_cp_prob[i])


        # # extract data from torch Variable, move to cpu, convert to numpy arrays
        # Y_que_hat = Y_que_hat.data.cpu().numpy()
        # Y_que = Y_que.data.cpu().numpy()
        #
        # # compute all metrics on this batch
        # summary_batch = {
        #     metric: metrics[metric](Y_que_hat, Y_que)
        #     for metric in metrics
        # }
        # summary_batch['loss'] = loss.item()
        # summ.append(summary_batch)
    for k,v in class_pair_summary.items():
        if v:
            add_value(class_summary,k[0],v)

    class_summary_merge={}
    for k, v in class_summary.items():
        class_summary_merge[k]=[]
        for vi in v:
            class_summary_merge[k]=class_summary_merge[k]=vi
    del class_summary
    class_mean_summary = {
        k: np.mean(np.mean(v))
        for k, v in class_summary_merge.items()
    }
    class_var_summary = {
        k: np.std(np.array(v))
        for k, v in class_summary_merge.items()
    }
    # # compute mean of all metrics in summary
    # metrics_mean = {
    #     metric: np.mean([x[metric] for x in summ])
    #     for metric in summ[0]
    # }
    # metrics_string = " ; ".join(
    #     "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return class_mean_summary, class_var_summary

def update_curriculum_cp(model_list,model, meta_optimizer, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args):
    his_class_mean_summary = {class_id.split('/')[-1]: []
                     for class_id in meta_classes}
    his_class_var_summary = {class_id.split('/')[-1]: []
                              for class_id in meta_classes}
    for proxy_model in model_list:
        proxy_path = os.path.join(args.model_dir,proxy_model)
        logging.info("Restoring proxy parameters from {}".format(proxy_path))
        utils.load_checkpoint(proxy_path, model, meta_optimizer)
        class_mean_summ, class_var_summ=class_pair_metric(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split,args)
        for k,v in class_mean_summ.items():
            add_value(his_class_mean_summary, k, v)
        for k,v in class_var_summ.items():
            add_value(his_class_var_summary, k, v)
    return his_class_mean_summary,his_class_var_summary


def meta_label_noise(Y_meta, num_classes, dl_meta, tr_classes, percent):
    for i in range(len(Y_meta)):
        if random.randint(1, num_classes) < num_classes*percent:
            Y_meta[i] = np.random.randint(1, num_classes)
        # label noise
    return Y_meta

