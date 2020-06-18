import torch
from torch import nn
from torch.nn import Dropout

import math


def setup_lambda_network_mf(opt):
    try:
        pn_type = opt.get('type')
    except Exception:
        print('Please specify the lambda_network_type')

    assert pn_type in ['global',
                       'dimension-wise', 'double-dimension-wise',
                       'user-wise', 'item-wise',
                       'user+item',
                       'dimension+user', 'dimension+item',
                       'dimension+user+item'], NotImplementedError('Invalid {}  lambda_network_type'.format(pn_type))
    
    if pn_type == 'global':
        pn = GlobalLambdaNetwork(opt)
    if pn_type == 'dimension-wise':
        pn = DimWiseLambdaNetwork(opt)
    if pn_type == 'double-dimension-wise':
        pn = DoubleDimWiseLambdaNetwork(opt)
    if pn_type == 'user-wise':
        pn = UserWiseLambdaNetwork(opt)
    if pn_type == 'item-wise':
        pn = ItemWiseLambdaNetwork(opt)
    if pn_type == 'user+item':
        pn = UserItemLambdaNetwork(opt)
    if pn_type == 'dimension+user':
        pn = UserDimWiseLambdaNetwork(opt)
    if pn_type == 'dimension+item':
        pn = ItemDimWiseLambdaNetwork(opt)
    if pn_type == 'dimension+user+item':
        pn = UserItemDimWiseLambdaNetwork(opt)
    return pn


class LambdaNetwork(nn.Module):
    """Meta-class for lambda-Net"""
    def __init__(self, opt):
        self._opt = opt
        super(LambdaNetwork, self).__init__()
        try:
            self.latent_dim = opt['latent_dim']
        except:
            raise KeyError('Please specify latent dim!')

        try:
            self.num_users = opt['num_users']
        except:
            raise KeyError('Please specify num_users!')

        try:
            self.num_items = opt['num_items']
        except:
            raise KeyError('Please specify num_items!')

        try:
            self.mf_lr = opt['mf_lr']
            self.mf_lr = opt['mf_lr']
        except:
            raise KeyError('Please specify mf_lr!')

        self.dp_prob = opt['dp_prob']
        if opt['mf_optimizer'] == 'adam':
            self.mf_optim_hyper_param = {'lr': opt['mf_lr'],
                                     'eps': opt['mf_eps'],
                                     'beta1': opt['mf_betas'][0],
                                     'beta2': opt['mf_betas'][1]}

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        pass

    def parse_lmbda(self, is_detach=True):
        if is_detach:
            if self._opt['type'] in ['dimension-wise', 'global']:
                self.ori_lambda.data.clamp_(0)
                # return self.ori_lambda.data.clone() # clone or detach?
                return self.ori_lambda.data.detach()
            elif self._opt['type'] in [ 'double-dimension-wise',
                                    'user-wise',
                                    'item-wise',
                                    'user+item',
                                    'dimension+user',
                                    'dimension+item',
                                    'dimension+user+item']:
                self.user_lambda.data.clamp_(0)  # Non-negative guarantee
                self.item_lambda.data.clamp_(0)  # Non-negative guarantee
                # return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]
                return [self.user_lambda.data.detach(), self.item_lambda.data.detach()]
        else:
            if self._opt['type'] in ['dimension-wise', 'global']:
                self.ori_lambda.data.clamp_(0)
                # return self.ori_lambda.data.clone() # clone or detach?
                return self.ori_lambda
            elif self._opt['type'] in [ 'double-dimension-wise',
                                    'user-wise',
                                    'item-wise',
                                    'user+item',
                                    'dimension+user',
                                    'dimension+item',
                                    'dimension+user+item']:
                self.user_lambda.data.clamp_(0)  # Non-negative guarantee, don't record this into the computational graph
                self.item_lambda.data.clamp_(0)  # Non-negative guarantee
                # return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]
                return [self.user_lambda, self.item_lambda]

    def get_emb(self):
        if hasattr(self, 'lambda'):
            return self.ori_lambda.data.cpu()
        if hasattr(self, 'user_lambda') and hasattr(self, 'item_lambda'):
            return [self.user_lambda.data.cpu(), self.item_lambda.data.cpu()]
    
    def get_next_reg_emb_adam(self, emb_grad_no_reg, curr_emb, curr_mf_optim_status):
        assert emb_grad_no_reg is not None
        assert curr_mf_optim_status is not None
        lr = self.mf_optim_hyper_param['lr']
        eps = self.mf_optim_hyper_param['eps']
        beta1, beta2 = self.mf_optim_hyper_param['beta1'], self.mf_optim_hyper_param['beta2']

        user_emb_cur, item_emb_cur = curr_emb

        t, _ = curr_mf_optim_status['step']
        s, r = curr_mf_optim_status['exp_avg'], curr_mf_optim_status['exp_avg_sq']
        s_u, s_i = s
        r_u, r_i = r
        # gradients of assumed regularized update
        user_emb_grad_no_reg, item_emb_grad_no_reg = emb_grad_no_reg
        user_emb_grad = user_emb_grad_no_reg + 2 * self.user_lambda * user_emb_cur
        item_emb_grad = item_emb_grad_no_reg + 2 * self.item_lambda * item_emb_cur

        s_u = beta1 * s_u + (1 - beta1) * user_emb_grad
        s_i = beta1 * s_i + (1 - beta1) * item_emb_grad
        # r_u = beta2 * r_u + (1 - beta2) * user_emb_grad * user_emb_grad
        # r_i = beta2 * r_i + (1 - beta2) * item_emb_grad * item_emb_grad
        r_u = r_u.mul(beta2).addcmul(1 - beta2, user_emb_grad, user_emb_grad)
        r_i = r_i.mul(beta2).addcmul(1 - beta2, item_emb_grad, item_emb_grad)
        denom_u = r_u.add(eps).sqrt() # Note: avoid gradient near zero 0.5 x^(-1/2)
        denom_i = r_i.add(eps).sqrt()
        bias_correction1 = (1 - beta1 ** t)
        bias_correction2 = (1 - beta2 ** t)
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        user_emb = user_emb_cur.addcdiv(-step_size, s_u, denom_u)
        item_emb = item_emb_cur.addcdiv(-step_size, s_i, denom_i)
        return [user_emb, item_emb]

    def get_next_reg_emb_sgd(self, next_emb_no_reg, curr_emb):
        """Obtain both user-item embedding's next-step version regularized using current lambda

        args:
            next_emb_non_reg: packed user,item emb
        """
        # TODO: rename the function
        def get_next_reg_emb_sgd_single(next_emb_non_reg, curr_emb, l):
            return next_emb_non_reg - 2 * self.mf_lr * l * curr_emb

        next_user_emb_non_reg, next_item_emb_non_reg = next_emb_no_reg
        curr_user_emb, curr_item_emb = curr_emb

        next_user_emb = get_next_reg_emb_sgd_single(next_user_emb_non_reg, curr_user_emb, self.user_lambda)
        next_item_emb = get_next_reg_emb_sgd_single(next_item_emb_non_reg, curr_item_emb, self.item_lambda)
        return [next_user_emb, next_item_emb]
        
    def get_preference_prob(self, u, i, j, user_emb, item_emb):
        u_emb = user_emb[u]
        i_emb = item_emb[i]
        j_emb = item_emb[j]
        ui = torch.mul(u_emb, i_emb).sum(dim=1)
        uj = torch.mul(u_emb, j_emb).sum(dim=1)
        logit = ui - uj
        return logit

    def forward_ui_lambda(self, next_emb_no_reg,
                           emb_grad_no_reg,
                           curr_emb,
                           curr_mf_optim_status,
                           u, i, j):
        if self._opt['mf_optimizer'] == 'sgd':
            next_user_emb, next_item_emb = self.get_next_reg_emb_sgd(next_emb_no_reg, curr_emb)
        elif self._opt['mf_optimizer'] == 'adam':
            next_user_emb, next_item_emb = self.get_next_reg_emb_adam(emb_grad_no_reg,
                                                                      curr_emb,
                                                                      curr_mf_optim_status)
        next_user_emb = Dropout(p=self.dp_prob)(next_user_emb)
        next_item_emb = Dropout(p=self.dp_prob)(next_item_emb)
        prob = self.get_preference_prob(u, i, j, next_user_emb, next_item_emb)
        return prob


class GlobalLambdaNetwork(LambdaNetwork):
    """Regularize via a global adaptive lambda"""
    def __init__(self, opt):
        super(GlobalLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.ori_lambda = nn.Parameter(init * torch.ones(1))

    def init_lambda(self):
        return self.ori_lambda.data.clone()

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        self.user_lambda = self.ori_lambda
        self.item_lambda = self.ori_lambda
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)
        

class DimWiseLambdaNetwork(LambdaNetwork):
    """Dimension-wise lambda, User and Item share the same lambda"""
    def __init__(self, opt):
        super(DimWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.ori_lambda = nn.Parameter(init * torch.ones(self.latent_dim))

    def init_lambda(self):
        return self.ori_lambda.data.clone()

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        self.user_lambda = self.ori_lambda
        self.item_lambda = self.ori_lambda
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class DoubleDimWiseLambdaNetwork(LambdaNetwork):
    """Dimension-wise lambda respective to user and item"""
    def __init__(self, opt):
        super(DoubleDimWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.latent_dim))
        self.item_lambda = nn.Parameter(init * torch.ones(self.latent_dim))

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class UserWiseLambdaNetwork(LambdaNetwork):
    def __init__(self, opt):
        super(UserWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.num_users, 1))
        if opt['use_cuda']:
            self.item_lambda = init * torch.ones(self.num_items, 1).cuda()  # no reg for item matrix
        else:
            self.item_lambda = init * torch.ones(self.num_items, 1)

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class ItemWiseLambdaNetwork(LambdaNetwork):
    def __init__(self, opt):
        super(ItemWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        if opt['use_cuda']:
            self.user_lambda = (init * torch.ones(self.num_users, 1)).cuda() # no reg for user matrix
        else:
            self.user_lambda = init * torch.ones(self.num_users, 1)
        self.item_lambda = nn.Parameter(init * torch.ones(self.num_items, 1))

    def init_lambda(self):
        return [self.user_lambda.clone(), self.item_lambda.data.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class UserItemLambdaNetwork(LambdaNetwork):
    def __init__(self, opt):
        super(UserItemLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.num_users, 1))
        self.item_lambda = nn.Parameter(init * torch.ones(self.num_items, 1))

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)
        
        
class UserDimWiseLambdaNetwork(LambdaNetwork):
    """Lambda update in SGDA
    See the Notebook in exp for details"""

    def __init__(self, opt):
        super(UserDimWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.num_users, self.latent_dim))
        self.item_lambda = nn.Parameter(init * torch.ones(self.latent_dim))

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class ItemDimWiseLambdaNetwork(LambdaNetwork):
    """Lambda update in SGDA
    See the Notebook in exp for details"""

    def __init__(self, opt):
        super(ItemDimWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.latent_dim))
        self.item_lambda = nn.Parameter(init * torch.ones(self.num_items, self.latent_dim))

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]

    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)


class UserItemDimWiseLambdaNetwork(LambdaNetwork):
    """Lambda update in SGDA
    See the Notebook in exp for details"""

    def __init__(self, opt):
        super(UserItemDimWiseLambdaNetwork, self).__init__(opt)
        init = opt['lambda_network_init']
        self.user_lambda = nn.Parameter(init * torch.ones(self.num_users, self.latent_dim))
        self.item_lambda = nn.Parameter(init * torch.ones(self.num_items, self.latent_dim))

    def init_lambda(self):
        return [self.user_lambda.data.clone(), self.item_lambda.data.clone()]
 
    def forward(self, next_emb_no_reg,
                      emb_grad_no_reg,
                      curr_emb,
                      curr_mf_optim_status,
                      u, i, j):
        return self.forward_ui_lambda(next_emb_no_reg,
                                           emb_grad_no_reg,
                                           curr_emb,
                                           curr_mf_optim_status,
                                           u, i, j)