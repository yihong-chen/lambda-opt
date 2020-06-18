from regularizer.modules.lambda_network_mf import setup_lambda_network_mf
from utils.train import use_cuda, use_optimizer, get_grad_norm

import torch
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import higher

from copy import deepcopy


def setup_regularizer(opt):
    try:
        reg_name = opt.get('regularizer')
    except Exception:
        print('Please specify the regularizer type')

    assert reg_name in ['fixed', 'alter_mf'], NotImplementedError(
        'Invalid {} regularizer'.format(reg_name))

    if reg_name == 'fixed':
        regularizer = FixedRegularizer(opt)
    if reg_name == 'alter_mf':
        regularizer = MFAlterRegularizer(opt)
    if reg_name == 'alter_mf_higher':
        regularizer = MFARHigher(opt)
    return regularizer


def setup_opt(opt, prefix):
    new_opt = deepcopy(opt)
    for k, v in opt.items():
        if k.startswith(prefix + '_'):
            new_opt[k[len(prefix) + 1:]] = v
    return new_opt


class Regularizer(object):
    """API for regularizer"""

    def __init__(self, opt):
        self._opt = opt
        self._mode = None
        self._train_step_idx = None
        self._train_episode_idx = None
        self.status = None  # status for current episode
        self.valid_mf_loss = 0

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        self._mode = new_mode

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

    def init_episode(self):
        """Initialization for a new episode"""
        self._train_step_idx = 0

    def get_lambda(self, status):
        """Generate lambda given the state"""
        pass

    def checkpoint(self, path):
        pass


class FixedRegularizer(Regularizer):
    """Regularizer with fixed lambda"""

    def __init__(self, opt):
        super(FixedRegularizer, self).__init__(opt)
        self._cur_idx = 0
        self._cur_lambda = None
        self._best_lambda = None
        self._best_valid_metrics = {'loss': np.inf, 'auc': 0}
        self.use_cuda = opt.get('use_cuda')

        try:
            self.lambda_candidate = opt['fixed_lambda_candidate']
        except:
            raise KeyError('no fixed_lambda_candidate specified')

    def set_cur_lambda(self):
        """set current lambda by choose one in candidates"""
        if self._cur_idx >= len(self.lambda_candidate):
            return False

        latent_dim = self._opt.get('latent_dim')
        cur_lambda = self.lambda_candidate[self._cur_idx]
        self._cur_lambda = torch.FloatTensor([cur_lambda])
        self._cur_idx += 1
        return True

    def init_lambda(self):
        if self.use_cuda:
            self._cur_lambda = self._cur_lambda.cuda()
        return self._cur_lambda

    def get_lambda(self, status):
        return self._cur_lambda

    def track_metrics(self, metrics):
        if metrics['auc'] > self._best_valid_metrics['auc']:
            self._best_lambda = self._cur_lambda
            self._best_valid_metrics = metrics


class AlterRegularizer(Regularizer):
    def __init__(self, opt):
        super(AlterRegularizer, self).__init__(opt)
        self.use_cuda = opt.get('use_cuda')

        self.criterion = BCEWithLogitsLoss(size_average=False)

        self.lambda_network_opt = setup_opt(opt, prefix='lambda_network')
        self.clip = self.lambda_network_opt.get('grad_clip')
        self.valid_mf_loss = None

    def init_lambda_network(self):
        assert hasattr(self, 'lambda_network')
        if self.use_cuda:
            use_cuda(True, self._opt['device_id'])
            self.lambda_network.cuda()
        self.optimizer = use_optimizer(self.lambda_network, self.lambda_network_opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.lambda_network_opt['lr_exp_decay'])

    def get_grad_norm(self):
        return get_grad_norm(self.lambda_network)

    def checkpoint(self, path):
        if hasattr(self.lambda_network, 'user_lambda'):
            user_lambda = self.lambda_network.user_lambda.data
            if user_lambda.is_cuda:
                user_lambda = user_lambda.cpu()
            user_lambda = user_lambda.numpy()
        elif hasattr(self.lambda_network, 'user_lambda_fe'):
            user_lambda = self.lambda_network.user_lambda_fe.data
            if user_lambda.is_cuda:
                user_lambda = user_lambda.cpu()
            user_lambda = user_lambda.numpy()        
        else:
            user_lambda = None
        if hasattr(self.lambda_network, 'item_lambda'):
            item_lambda = self.lambda_network.item_lambda.data
            if item_lambda.is_cuda:
                item_lambda = item_lambda.cpu()
            item_lambda = item_lambda.numpy()
        elif hasattr(self.lambda_network, 'item_lambda_fe'):
            item_lambda = self.lambda_network.item_lambda_fe.data
            if item_lambda.is_cuda:
                item_lambda = item_lambda.cpu()
            item_lambda = item_lambda.numpy()
        else:
            item_lambda = None
        np.savez(path,
                 user=user_lambda,
                 item=item_lambda)


class MFAlterRegularizer(AlterRegularizer):
    def __init__(self, opt):
        super(MFAlterRegularizer, self).__init__(opt)
        self.lambda_network = setup_lambda_network_mf(self.lambda_network_opt)
        self.init_lambda_network()

    def init_episode(self):
        super(MFAlterRegularizer, self).init_episode()
        self.lambda_network = setup_lambda_network_mf(self.lambda_network_opt)
        self.init_lambda_network()

    def init_lambda(self):
        """Initialization using zero"""
        return self.lambda_network.init_lambda()

    def update_mf_lr(self, lr):
        # print('Update mf_lr {} for lambda network ...'.format(lr))
        self.lambda_network.mf_lr = lr

    # @profile
    def update(self, next_embs_non_reg, # from assumed non-regularized update
               emb_grad_non_reg, 
               curr_embs,
               curr_mf_optim_status,
               sampler):
        """update lambda_network given (u, i, j),
           requires next step non-regularized param and current param
        """
        valid_u, valid_i, valid_j = sampler.get_sample('valid')
        assert isinstance(valid_u, torch.LongTensor)
        assert isinstance(valid_i, torch.LongTensor)
        assert isinstance(valid_j, torch.LongTensor)

        self.lambda_network.train()
        self.optimizer.zero_grad()

        valid_preference = torch.ones(valid_u.size()[0])
        if self.use_cuda:
            valid_preference = valid_preference.cuda()
            valid_u, valid_i, valid_j = valid_u.cuda(), valid_i.cuda(), valid_j.cuda()
        valid_prob_preference = self.lambda_network(next_embs_non_reg,
                                                     emb_grad_non_reg,
                                                     curr_embs,
                                                     curr_mf_optim_status,
                                                     valid_u, valid_i, valid_j)
        valid_loss = self.criterion(valid_prob_preference, valid_preference)
        loss = valid_loss / valid_u.size()[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lambda_network.parameters(), self.clip)
        self.optimizer.step()
        self.valid_mf_loss = loss.item()
        return self.lambda_network.parse_lmbda(is_detach=True)

    def get_lambda(self, status):
        """Generate lambda given the state

        validation set gradient descent, fixed mf parameters, update only lambda

        Args:
            status: python dictionary, contains the following stuff
                   status['factorizer'], the matrix factorization model

        Return:
            pytorch tensor, dimension-wise lambda
        """
        sampler = status['sampler']
        factorizer = status['factorizer']
        self.update_mf_lr(lr=factorizer.scheduler.get_lr()[0])  # reset mf_lr for lambda network
        curr_embs = factorizer.param

        # Assumed upate without regularization
        factorizer.update(sampler,
                          l2_lambda=self.init_lambda())  # assumed param update without regularization, self.init_lambda return zero initialization
        if self._opt['mf_optimizer'] == 'sgd':
            next_embs_non_reg = factorizer.param
            emb_grad_non_reg = None
            curr_mf_optim_status = None
        elif self._opt['mf_optimizer'] == 'adam':
            next_embs_non_reg = None
            emb_grad_non_reg = factorizer.param_grad
            curr_mf_optim_status = factorizer.optim_status
        else:
            raise NotImplementedError('mf_optimizer not found!')

        if self.train_step_idx > 0 and self.train_step_idx % sampler.num_batches_valid == 0:
            self.scheduler.step()  # decay lr every epoch
            print('lambda network lr {}'.format(self.scheduler.get_lr()))

        self.train_step_idx += 1
        # Update lambda network
        next_lambda = self.update(next_embs_non_reg,
                                  emb_grad_non_reg,
                                  curr_embs,
                                  curr_mf_optim_status,
                                  sampler)  # get next lambda via update lambda network
        return next_lambda


class MFARHigher(AlterRegularizer):
    def __init__(self, opt):
        super(MFARHigher, self).__init__(opt)
        self.lambda_network = setup_lambda_network_mf(self.lambda_network_opt)
        self.init_lambda_network()

    def init_episode(self):
        super(MFARHigher, self).init_episode()
        self.lambda_network = setup_lambda_network_mf(self.lambda_network_opt)
        self.init_lambda_network()

    def init_lambda(self):
        return self.lambda_network.init_lambda()

    def update(self, m, m_optim, sampler_train, sampler_valid):
        ...

    def get_lambda(self, status):
        sampler = status['sampler']
        factorizer = status['factorizer']
        m = factorizer.model
        m_optimizer = factorizer.optimizer
        self.update_mf_lr(lr=factorizer.scheduler.get_lr()[0])
        self.train_step_idx += 1

        with higher.innerloop_ctx(m, m_optimizer) as (fmodel, diffopt):
            # look-ahead, this is very similar to factorizer update except that lambda is included in the computational graph
            u, i, j = sampler.get_sample('train')
            preference = torch.ones(u.size())[0]
            if self.use_cuda:
                u, i, j = u.cuda(), i.cuda(), j.cuda()
                preference = preference.cuda()
            prob_preference = fmodel.forward_triple(u, i, j)
            l_fit = self.criterion(prob_preference, preference) / (u.size()[0])
            lmbda = self.lambda_network.parse_lmbda(is_detach=False)
            l_reg = fmodel.l2_penalty(lmbda, u, i, j) / (u.size()[0])
            l = l_fit + l_reg
            diffopt.step(l)

            # compute the validation loss
            valid_u, valid_i, valid_j = sampler.get_sample('valid')
            valid_preference = torch.ones(valid_u.size()[0])
            if self.use_cuda:
                valid_preference = valid_preference.cuda()
                valid_u, valid_i, valid_j = valid_u.cuda(), valid_i.cuda(), valid_j.cuda()

            self.lambda_network.train()
            self.optimizer.zero_grad()
            valid_prob_preference = fmodel.forward_triple(valid_u, valid_i, valid_j) / valid_u.size()[0]
            l_val = self.criterion(valid_prob_preference, valid_preference)
            l_val.backward()
            torch.nn.utils.clip_grad_norm_(self.lambda_network.parameters(), self.clip)
            self.optimizer.step()
            self.valid_mf_loss = l_val.item()
            return self.lambda_network.parse_lmbda(is_detach=True)