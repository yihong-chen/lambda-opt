import os
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from factorizer.factorizer import setup_factorizer
from regularizer.regularizer import setup_regularizer
from utils.data_loader import setup_sample_generator
from utils.mp_cuda_evaluate import evaluate_ui_uj_df


def setup_args(parser=None):
    """ Set up arguments for the Engine

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()
    data = parser.add_argument_group('Data')
    engine = parser.add_argument_group('Engine Arguments')
    factorize = parser.add_argument_group('Factorizer Arguments')
    matrix_factorize = parser.add_argument_group('MF Arguments')
    regularize = parser.add_argument_group('Regularizer Arguments')
    log = parser.add_argument_group('Tensorboard Arguments')

    engine.add_argument('--alias', default='experiment',
                        help='Name for the experiment')

    data.add_argument('--data-path', default='./data/ml-1m/ratings.dat')
    data.add_argument('--data-type', default='ml1m-mf', help='type of the dataset')
    data.add_argument('--filtered-data-path', default='./tmp/data/ml1m-mf-processed_ui_history.dat', 
                      help='path for cache the filtered data')
    data.add_argument('--reconstruct-data', default=True, help='re-filter the data')
    data.add_argument('--train-test-split', default='loo', help='train/test split method')
    data.add_argument('--random-split', default=False, help='random split or according to time')
    data.add_argument('--train_test-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--train-valid-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--test-latest-n', default=1)
    data.add_argument('--valid-latest-n', default=1)
    data.add_argument('--test-ratio', default=0.01)
    data.add_argument('--valid-ratio', default=0.01)
    data.add_argument('--num-negatives', default=1)
    data.add_argument('--batch-size-train', default=1)
    data.add_argument('--batch-size-valid', default=1)
    data.add_argument('--batch-size-test', default=1)
    data.add_argument('--multi-cpu-train', default=False)  # TODO
    data.add_argument('--multi-cpu-valid', default=False)
    data.add_argument('--multi-cpu-test', default=False)
    data.add_argument('--num-workers-train', default=1)   # TODO
    data.add_argument('--num-workers-valid', default=1)
    data.add_argument('--num-workers-test', default=1)
    data.add_argument('--device-ids-test', default=[0, 1, 2, 3], help='devices used for multi-processing evaluate')

    regularize.add_argument('--regularizer', default='fixed', help='type of regularizer, fixed or adaptive')
    regularize.add_argument('--penalty-param-path', default=None, help='save lambda for analysis')
    regularize.add_argument('--lambda-network-grad-clip', default=100)
    regularize.add_argument('--lambda-network-type', default='dimension-wise', help='granularity of lambda')
    regularize.add_argument('--lambda-network-lr', default=0.1, help='lr for lambda update')
    regularize.add_argument('--lambda-network-optimizer', default='sgd')
    regularize.add_argument('--lambda-network-dp-prob', default=0.5)
    regularize.add_argument('--lambda-network-multi-step', default=1)
    regularize.add_argument('--fixed-lambda-candidate',
                            default=[10, 1, 0, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
                            help='Grid search candidate for fixed')
    regularize.add_argument('--max-steps', default=1e8)
    regularize.add_argument('--lambda-update-interval', default=1,
                            help='Interval between two lambda updates')
    regularize.add_argument('--use-cuda', default=True)
    regularize.add_argument('--device-id', default=1, help='Training Devices')

    factorize.add_argument('--factorizer', default='mf', help='Type of the Factorization Model')
    factorize.add_argument('--metric-topk', default=10, help='Top K for HR and NDCG metric')
    factorize.add_argument('--latent-dim', default=8)

    type_opt = 'mf'
    matrix_factorize.add_argument('--{}-optimizer'.format(type_opt), default='sgd')
    matrix_factorize.add_argument('--{}-lr'.format(type_opt), default=1e-3)
    matrix_factorize.add_argument('--{}-grad-clip'.format(type_opt), default=1)

    log.add_argument('--log-interval', default=1)
    log.add_argument('--tensorboard', default='./tmp/runs')
    return parser


class Engine(object):
    """Engine wrapping the training & evaluation
       of adpative regularized maxtirx factorization
    """
    def __init__(self, opt):
        self._opt = opt
        self._sampler = setup_sample_generator(opt)

        self._opt['num_users'] = self._sampler.num_users
        self._opt['num_items'] = self._sampler.num_items
        self._opt['eval_res_path'] = self._opt['eval_res_path'].format(alias=self._opt['alias'],
                                                                       epoch_idx='{epoch_idx}')
        if self._opt['penalty_param_path'] is not None:
            self._opt['penalty_param_path'] = self._opt['penalty_param_path'].format(
                alias=self._opt['alias'],                                                                 epoch_idx='{epoch_idx}')
        # print('Set up factorizer ...')
        self._factorizer = setup_factorizer(opt)
        self._factorizer_assumed = setup_factorizer(opt)
        # print('Set up regularizer ...')
        self._regularizer = setup_regularizer(opt)
        self._writer = SummaryWriter(log_dir='{}/{}'.format(opt['tensorboard'], opt['alias']))
        self._writer.add_text('option', str(opt), 0)
        self._mode = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in ['complete', 'partial', None]  # training a complete trajectory or a partial trajctory
        self._mode = new_mode

    def train_fixed_reg(self):
        self.mode = 'complete'
        while self._regularizer.set_cur_lambda():
            valid_metrics, _ = self.train_an_episode(max_steps=self._opt['max_steps'])
            self._regularizer.track_metrics(valid_metrics)

    def train_alter_reg(self):
        self.mode = 'complete'
        self.train_an_episode(self._opt['max_steps'])

    def train_an_episode(self, max_steps, episode_idx=''):
        """Train a regularized matrix factorization model"""
        assert self.mode in ['partial', 'complete']

        print('-' * 80)
        print('[{} episode {} starts!]'.format(self.mode, episode_idx))

        log_interval = self._opt.get('log_interval')
        eval_interval = self._opt.get('eval_interval')
        lambda_update_interval = self._opt.get('lambda_update_interval')

        status = dict()
        print('Initializing ...')
        self._regularizer.init_episode()
        self._factorizer.init_episode()
        curr_lambda = self._regularizer.init_lambda()
        valid_mf_loss, train_mf_loss = np.inf, np.inf
        epoch_start = datetime.now()
        for step_idx in range(int(max_steps)):  # TODO: terminal condition for an episode
            # Prepare status for current step
            status['done'] = False
            status['sampler'] = self._sampler
            self._factorizer_assumed.copy(self._factorizer)
            status['factorizer'] = self._factorizer_assumed  # for assumed copy

            self._regularizer.observe(status)
            # Regularizer generate lambda for current mf update step
            # print('Regularizer generate lambda ...')
            if (step_idx % lambda_update_interval == 0) and (step_idx > 0):
                curr_lambda = self._regularizer.get_lambda(status=status)
                valid_mf_loss = self._regularizer.valid_mf_loss
            # Factorizer update
            # print('Factorizer update ...')
            train_mf_loss = self._factorizer.update(self._sampler, l2_lambda=curr_lambda)
            train_l2_penalty = self._factorizer.l2_penalty
            status['train_mf_loss'] = train_mf_loss

            # Logging & Evaluate on the Evaluate Set
            if self.mode == 'complete' and step_idx % log_interval == 0:
                epoch_idx = int(step_idx / self._sampler.num_batches_train)
                print('[Epoch {}|Step {}]'.format(epoch_idx, step_idx % self._sampler.num_batches_train))
                self._writer.add_scalar('train/step_wise/mf_loss', train_mf_loss, step_idx)
                self._writer.add_scalar('train/step_wise/l2_penalty', train_l2_penalty, step_idx)
                self._writer.add_scalar('valid/step_wise/mf_loss', valid_mf_loss, step_idx)
                # for analysis, comment it to accelerate training
                mf_grad_norm = self._factorizer.get_grad_norm()
                if hasattr(self._regularizer, 'get_grad_norm'):
                    reg_grad_norm = self._regularizer.get_grad_norm()
                else:
                    reg_grad_norm = 0
                self._writer.add_scalar('grad_norm/step_wise/mf', mf_grad_norm, step_idx)
                self._writer.add_scalar('grad_norm/step_wise/reg', reg_grad_norm, step_idx)

                if step_idx % self._sampler.num_batches_train == 0: 
                    # for analysis, histgram of lambda, comment it to accelerate training, 
                    # log histgram every epoch
                    if isinstance(curr_lambda, list):
                        user_lambda, item_lambda = curr_lambda[0].cpu().numpy(), curr_lambda[1].cpu().numpy()
                    else: 
                        user_lambda, item_lambda = curr_lambda.cpu().numpy(), curr_lambda.cpu().numpy()
                    self._writer.add_histogram('lambda/epoch_wise/user', user_lambda, epoch_idx)
                    self._writer.add_histogram('lambda/epoch_wise/item', item_lambda, epoch_idx)

                if (step_idx % self._sampler.num_batches_train == 0) and (epoch_idx % eval_interval == 0):
                    print('Evaluate on test ...')
                    start = datetime.now()
                    eval_res_path = self._opt['eval_res_path'].format(epoch_idx=epoch_idx)
                    eval_res_dir, _ = os.path.split(eval_res_path)
                    if not os.path.exists(eval_res_dir):
                        os.mkdir(eval_res_dir)
                    model = self._factorizer.model
                    model_name = self._opt['factorizer']

                    ui_uj_df = self._sampler.test_ui_uj_df
                    item_pool = self._sampler.item_pool
                    top_k = self._opt['metric_topk']
                    use_cuda = self._opt['use_cuda']
                    device_ids = self._opt['device_ids_test']
                    num_workers = self._opt['num_workers_test']
                    test_metrics = evaluate_ui_uj_df(model=model, model_name=model_name, card_feat=None, 
                                          ui_uj_df=ui_uj_df, item_pool=item_pool, 
                                          metron_top_k=top_k, eval_res_path=eval_res_path,  
                                          use_cuda=use_cuda, device_ids=device_ids, num_workers=num_workers)
                    # save lambda network's parameter
                    if self._opt['penalty_param_path'] is not None:
                        penalty_param_path = self._opt['penalty_param_path'].format(epoch_idx=epoch_idx)
                        penalty_param_dir, _ = os.path.split(penalty_param_path)
                        if not os.path.exists(penalty_param_dir):
                            os.mkdir(penalty_param_dir)
                        self._regularizer.checkpoint(penalty_param_path)
                    self._writer.add_scalar('test/epoch_wise/metron_auc', test_metrics['auc'], epoch_idx)
                    self._writer.add_scalar('test/epoch_wise/metron_hr', test_metrics['hr'], epoch_idx)
                    self._writer.add_scalar('test/epoch_wise/metron_ndcg', test_metrics['ndcg'], epoch_idx)   
                    end = datetime.now()
                    print('Evaluate Time {} minutes'.format((end - start).total_seconds() / 60))
                    epoch_end = datetime.now()
                    dur = (epoch_end - epoch_start).total_seconds() / 60
                    epoch_start = datetime.now()
                    print('[Epoch {:4d}] train MF loss: {:04.8f}, '
                          'valid loss: {:04.8f}, time {:04.8f} minutes'.format(epoch_idx,
                                                                               train_mf_loss,
                                                                               valid_mf_loss,
                                                                               dur))

    def train(self):
        if self._opt['regularizer'] == 'fixed':
            self.train_fixed_reg()
        elif 'alter' in self._opt['regularizer']:
            self.train_alter_reg()


if __name__ == '__main__':
    opt = setup_args()
    engine = Engine(opt).train()