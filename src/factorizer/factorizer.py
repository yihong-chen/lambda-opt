import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from copy import deepcopy
from datetime import datetime

from factorizer.modules import MF
from utils.train import use_cuda, use_optimizer, get_grad_norm
from utils.data_loader import UserItemDataset


def setup_factorizer(opt):
    new_opt = deepcopy(opt)
    if opt['factorizer'] == 'mf':
        for k, v in opt.items():
            if k.startswith('mf_'):
                new_opt[k[3:]] = v
        return MFBPRFactorizer(new_opt)
    elif opt['factorizer'] == 'fm': # TODO
        for k, v in opt.items():
            if k.startswith('fm_'):
                new_opt[k[3:]] = v
        return FMBPRFactorizer(new_opt)


class BPR_Factorizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.clip = opt.get('grad_clip')
        self.use_cuda = opt.get('use_cuda')
        self.batch_size_test = opt.get('batch_size_test')

        # self.metron = MetronAtK(top_k=opt['metric_topk'])
        self.criterion = BCEWithLogitsLoss(size_average=False)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.l2_penalty = None

        self.param_grad = None
        self.optim_status = None

        self.prev_param = None
        self.param = None

        # is the factorizer for assumed update
        self.is_assumed = False

        self._train_step_idx = None
        self._train_episode_idx = None

        self.copy_count = 0
    # @profile
    def copy(self, new_factorizer):
        """Return a new copy of factorizer

        # Note: directly using deepcopy wont copy factorizer.scheduler correctly
                the gradient of self.model is not copied!
        """
        self.copy_count = self.copy_count + 1
        #print(self.copy_count)
        self.train_step_idx = deepcopy(new_factorizer.train_step_idx)
        self.param = deepcopy(new_factorizer.param)
        self.model.load_state_dict(deepcopy(new_factorizer.model.state_dict()))
        self.optimizer = use_optimizer(self.model, self.opt)
        self.optimizer.load_state_dict(deepcopy(new_factorizer.optimizer.state_dict()))

        self.scheduler = ExponentialLR(self.optimizer,
                                      gamma=self.opt['lr_exp_decay'],
                                       last_epoch=self.scheduler.last_epoch)

    @property
    def delta_param(self):
        """update of parameter, for SAC regularizer

        return:
            list of pytorch tensor
        """
        delta_param = list()
        for i, (prev_p, p) in enumerate(zip(self.prev_param, self.param)):
            delta_param[i] = p - prev_p
        return delta_param

    @property
    def train_step_idx(self):
        return self._train_step_idx

    @train_step_idx.setter
    def train_step_idx(self, new_step_idx):
        self._train_step_idx = new_step_idx

    @property
    def train_episode_idx(self):
        return self._train_episode_idx

    @train_episode_idx.setter
    def train_episode_idx(self, new_episode_idx):
        self._train_episode_idx = new_episode_idx

    def get_grad_norm(self):
        assert hasattr(self, 'model')
        return get_grad_norm(self.model)

    def set_assumed_flag(self, is_assumed):
        self.is_assumed = is_assumed

    # @profile
    def update(self, sampler, l2_lambda):
        if (self.train_step_idx > 0) and (
                self.train_step_idx % sampler.num_batches_train == 0):  # sampler.get_num_batch_per_epoch('train')
            self.scheduler.step()
            print('\tfactorizer lr decay ...')
        self.train_step_idx += 1

        self.model.train()
        self.optimizer.zero_grad()


class MFBPRFactorizer(BPR_Factorizer):
    def __init__(self, opt):
        super(MFBPRFactorizer, self).__init__(opt)
        self.model = MF(opt)
        self.opt = opt
        if self.use_cuda:
            use_cuda(True, opt['device_id'])
            self.model.cuda()

        self.optimizer = use_optimizer(self.model, opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=opt['lr_exp_decay'])

    def init_episode(self):
        opt = self.opt
        self.model = MF(opt)
        self._train_step_idx = 0
        if self.use_cuda:
            use_cuda(True, opt['device_id'])
            self.model.cuda()
        self.optimizer = use_optimizer(self.model, opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=opt['lr_exp_decay'])
        self.param = [p.data.clone() for p in self.model.parameters()]

    def update(self, sampler, l2_lambda):
        """update MF model paramters given (u, i, j)

        Args:
            l2_lambda: pytorch Tensor, dimension-wise lambda
        """
        super(MFBPRFactorizer, self).update(sampler, l2_lambda)
        u, i, j = sampler.get_sample('train')

        assert isinstance(u, torch.LongTensor)
        assert isinstance(i, torch.LongTensor)
        assert isinstance(j, torch.LongTensor)

        preference = torch.ones(u.size()[0])

        if self.use_cuda:
            u, i, j = u.cuda(), i.cuda(), j.cuda()
            preference = preference.cuda()
        prob_preference = self.model.forward_triple(u, i, j)
        non_reg_loss = self.criterion(prob_preference, preference) / (u.size()[0])
        l2_reg_loss = self.model.l2_penalty(l2_lambda, u, i, j) / (u.size()[0])
        loss = non_reg_loss + l2_reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        if (self.opt['optimizer'] == 'adam') and self.is_assumed:
            param_grad = []
            for p in self.model.parameters():  # Note: deepcopy model lose the gradients
                if p.grad is not None:
                    param_grad.append(p.grad.data.clone())
                else:
                    print('\tNo gradient!')
                    # param_grad.append(torch.zeros_like(p.data.cpu()))
                    param_grad.append(torch.zeros_like(p.data))
            self.param_grad = param_grad
            self.optim_status = {'step': [v['step']
                                          for _, v in self.optimizer.state.items() if len(v) > 0],
                                 'exp_avg': [v['exp_avg'].data.clone()
                                             for _, v in self.optimizer.state.items() if len(v) > 0],
                                 'exp_avg_sq': [v['exp_avg_sq'].data.clone()
                                                for _, v in self.optimizer.state.items() if len(v) > 0]}

        self.optimizer.step()

        # print('\tcurrent learing rate of MF optimizer ...')
        # for param_grp in self.optimizer.param_groups:
        #     print('\t{}'.format(param_grp['lr']))
        # print('-' * 80)
        if 'alter' in self.opt['regularizer']:
            # only adaptive methods need to cache current param
            self.param = [p.data.clone() for p in self.model.parameters()]
        self.l2_penalty = l2_reg_loss.item()
        # print('Loss {}, Non-reg Loss {}, L2 lambda {}'.format(loss.item(), non_reg_loss.item(), l2_reg_loss.item()))
        return non_reg_loss.item()