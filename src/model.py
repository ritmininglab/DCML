import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

N_FILTERS = 64  # number of filters used in conv_block
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
EPS = 1e-8  # epsilon for numerical stability


class MetaLearner(nn.Module):
    """
    The class defines meta-learner for MAML algorithm.
    Training details will be written in train.py.
    TODO base-model invariant MetaLearner class
    """

    def __init__(self, params):
        super(MetaLearner, self).__init__()
        self.params = params
        self.meta_learner = Net(
            params.in_channels, params.num_classes, dataset=params.dataset)

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            param = torch.nn.Parameter(param)
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class Net(nn.Module):
    """
    The base CNN model for MAML for few-shot learning.
    The architecture is same as of the embedding in MatchingNet.
    """

    def __init__(self, in_channels, num_classes, dataset='Omniglot'):
        """
        self.net returns:
            [N, 64, 1, 1] for Omniglot (28x28)
            [N, 64, 5, 5] for miniImageNet (84x84)
        self.fc returns:
            [N, num_classes]
        
        Args:
            in_channels: number of input channels feeding into first conv_block
            num_classes: number of classes for the task
            dataset: for the measure of input units for self.fc, caused by 
                     difference of input size of 'Omniglot' and 'ImageNet'
        """
        super(Net, self).__init__()
        self.features = nn.Sequential(
            conv_block(0, in_channels, padding=1, pooling=True),
            conv_block(1, N_FILTERS, padding=1, pooling=True),
            conv_block(2, N_FILTERS, padding=1, pooling=True),
            conv_block(3, N_FILTERS, padding=1, pooling=True))
        if dataset == 'Omniglot':
            self.add_module('fc', nn.Linear(64, num_classes))
        elif dataset == 'ImageNet' or dataset ==  'food'or dataset ==  'miniweb':
            self.add_module('fc', nn.Linear(64 * 5 * 5, num_classes))
        elif dataset == 'FC100':
            self.add_module('fc', nn.Linear(64 * 5 * 5, num_classes)) #self.add_module('fc', nn.Linear(64 * 2 * 2, num_classes))
        elif dataset == 'CUB':
            self.add_module('fc', nn.Linear(64 * 2 * 2, num_classes))
        else:
            raise Exception("I don't know your dataset")

    def forward(self, X, params=None):
        """
        Args:
            X: [N, in_channels, W, H]
            params: a state_dict()
        Returns:
            out: [N, num_classes] unnormalized score for each class
        """
        if params == None:
            out = self.features(X)#FC 5,64,2,2
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            """
            The architecure of functionals is the same as `self`.
            """
            out = F.conv2d(
                X,
                params['meta_learner.features.0.conv0.weight'],
                params['meta_learner.features.0.conv0.bias'],
                padding=1)
            # NOTE we do not need to care about running_mean anv var since
            # momentum=1.
            out = F.batch_norm(
                out,
                params['meta_learner.features.0.bn0.running_mean'],
                params['meta_learner.features.0.bn0.running_var'],
                params['meta_learner.features.0.bn0.weight'],
                params['meta_learner.features.0.bn0.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.1.conv1.weight'],
                params['meta_learner.features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.1.bn1.running_mean'],
                params['meta_learner.features.1.bn1.running_var'],
                params['meta_learner.features.1.bn1.weight'],
                params['meta_learner.features.1.bn1.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.2.conv2.weight'],
                params['meta_learner.features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.2.bn2.running_mean'],
                params['meta_learner.features.2.bn2.running_var'],
                params['meta_learner.features.2.bn2.weight'],
                params['meta_learner.features.2.bn2.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.3.conv3.weight'],
                params['meta_learner.features.3.conv3.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.3.bn3.running_mean'],
                params['meta_learner.features.3.bn3.running_var'],
                params['meta_learner.features.3.bn3.weight'],
                params['meta_learner.features.3.bn3.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['meta_learner.fc.weight'],
                           params['meta_learner.fc.bias'])

        out = F.log_softmax(out, dim=1)
        return out


def conv_block(index,
               in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True)),
                ('pool'+str(index), nn.MaxPool2d(MP_SIZE))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True))
            ]))
    return conv


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# Maintain all metrics required in this dictionary.
# These are used in the training and evaluation loops.
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', ignore.weight.data)
        self.register_buffer('bias', ignore.bias.data)
        # self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        # self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class Conv4Net(MetaModule):
    """
    The class defines meta-learner for MAML algorithm.
    Training details will be written in train.py.
    TODO base-model invariant MetaLearner class
    """

    def __init__(self, params):
        super(Conv4Net, self).__init__()
        self.params = params
        self.meta_learner = Net(
            params.in_channels, params.num_classes, dataset=params.dataset)

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict