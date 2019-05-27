import numpy as np
import pandas as pd

import os
import math
import random
from copy import deepcopy
from datetime import datetime

import torch
from torch.multiprocessing import Process, Queue
from torch.utils.data import DataLoader, Dataset


def setup_sample_generator(opt):
    """Choose different type of sampler for MF & FM"""
    if 'fm' in opt['data_type']:
        return FMSampleGenerator(opt)
    if 'mf' in opt['data_type']:
        return MFSampleGenerator(opt)


def split_loo(ui_history, latest_n=1, is_random=False):
    """leave one out train/test split

    Args:
        ui_history, pandas dataframe
    """
    print('Split train/test latest_n {}'.format(latest_n))
    if is_random:
        ui_history['idx'] = np.random.randint(len(ui_history), size=len(ui_history))
        ui_history['rank'] = ui_history.groupby(['userId'])['idx'].rank(method='first', ascending=False)
    else:
        ui_history['rank'] = ui_history.groupby('userId')['timestamp'].rank(method='first', ascending=False)
    test = ui_history[ui_history['rank'] <= latest_n]
    train = ui_history[ui_history['rank'] > latest_n]
    assert train['userId'].nunique() == test['userId'].nunique()
    if is_random:
        del train['idx']
        del test['idx']
    del train['rank']
    del test['rank']
    return train, test


def split_lro(ui_history, test_ratio=0.01, is_random=False):
    """Split train/test according to ratio"""
    print('Split train/test, test ratio {}'.format(test_ratio))
    if is_random:
        ui_history.loc[:, 'idx'] = np.random.randint(len(ui_history), size=len(ui_history))
        ui_history.loc[:, 'rank'] = ui_history.groupby(['userId'])['idx'].rank(method='first', ascending=False)
    else:
        ui_history.loc[:, 'rank'] = ui_history.groupby('userId')['timestamp'].rank(method='first', ascending=False)
    ui_history.loc[:, 'limit'] = ui_history.groupby(['userId'])['itemId'].transform(lambda x: int(len(x) * test_ratio))
    test = ui_history[ui_history['rank'] <= ui_history['limit']]
    train = ui_history[ui_history['rank'] > ui_history['limit']]
    if is_random:
        del train['idx']
        del test['idx']
    del train['rank']
    del test['rank']
    del train['limit']
    del test['limit']
    return train, test


def split_freq_wise(ui_history, freq_bd, latest_n_low_freq, test_ratio_high_freq, is_random=False):
    """user with low frequency would be split into train/test according to latest_n_low_freq
    user with high frequency would be split into train/test according to test_ratio_high_freq

    args:
        freq_bd: frequency boundary
    """
    print('Split train/test frequency-wise ...')
    user_freq = ui_history[['userId', 'itemId']].groupby('userId').count().reset_index().rename(
        columns={'itemId': 'freq'})
    low_freq_user = user_freq[user_freq['freq'] < freq_bd]
    lu_history = pd.merge(ui_history, low_freq_user, on=['userId'], how='right')
    print('\tNumber of users with freqeuncy lower than {}: {}'.format(freq_bd, len(low_freq_user)))
    print('\tTotal records of users with freqeuncy lower than {}: {}'.format(freq_bd, len(lu_history)))
    high_freq_user = user_freq[user_freq['freq'] >= freq_bd]
    hu_history = pd.merge(ui_history, high_freq_user, on=['userId'], how='right')
    print('\tNumber of users with freqeuncy higher or equal than {}: {}'.format(freq_bd, len(high_freq_user)))
    print('\tTotal records of users with freqeuncy higher or equal than {}: {}'.format(freq_bd, len(hu_history)))
    train_lu, test_lu = split_loo(lu_history, latest_n=latest_n_low_freq, is_random=is_random)
    train_hu, test_hu = split_lro(hu_history, test_ratio=test_ratio_high_freq, is_random=is_random)
    train = pd.concat([train_hu, train_lu])
    test = pd.concat([test_hu, test_lu])
    del train['freq']
    del test['freq']
    return train, test


def load_ui_data(data_type, data_path):
    """Load data as pd.DataFrame from path

    Note:
        It takes long(10 minutes) to load gowalla data

    return
        DataFrame with columns = ['uid', 'iid', 'timestamp']
    """
    print('Loading data {} from {} ...'.format(data_type, data_path))
    start = datetime.now()
    if data_type in ['ml1m-mf']:
        ui_data = pd.read_csv(data_path, sep='::', header=None,
                              names=['uid', 'iid', 'rating', 'timestamp'],
                              engine='python')
    elif data_type in ['ml10m-mf']:
        ui_data = pd.read_csv(data_path, sep='::', header=None,
                              names=['uid', 'iid', 'rating', 'timestamp'],
                              engine='python')
    elif data_type == 'amazon-food-mf':
        ui_data = pd.read_csv(data_path,
                       header=0, 
                       parse_dates=['Time'],
                       engine='python')[['ProductId', 'UserId', 'Score', 'Time']]
        ui_data.rename(columns={'UserId': 'uid', 'ProductId': 'iid', 'Time': 'timestamp', 'Score': 'rating'}, inplace=True)
        # print(ui_data.describe())
        # print(ui_data.head(20))
    else:
        raise NotImplementedError('Invalid data_type!')
    ui_data.sort_values(['uid', 'timestamp'], ascending=True, inplace=True)
    end = datetime.now()
    print('\tSuccessfully load {} from {}, {} mins'.format(data_type, data_path, (end - start).seconds / 60.0))
    return ui_data


def filter_ui_data(ui_data,
                   num_users_per_item_lb=10,
                   num_users_per_item_ub=int(1e8),
                   num_items_per_user_lb=10,
                   num_items_per_user_ub=int(1e8)):
    """Filter ui_data according to user freq limit and item freq limit

    args:
        ui_data: pd.DataFrame, columns = ['uid', 'iid', 'timestamp']
    """
    print('Filtering data ...')
    start = datetime.now()
    item_cnt = ui_data[['uid', 'iid']].groupby('iid').count().reset_index()
    item_cnt.rename(columns={'uid': 'count'}, inplace=True)
    item_cnt = item_cnt[item_cnt['count'] >= num_users_per_item_lb]
    item_cnt = item_cnt[item_cnt['count'] <= num_users_per_item_ub]
    ui_data = pd.merge(item_cnt, ui_data, on='iid', how='inner')
    print('\tmin(num_users_per_item): {}'.format(ui_data['count'].min()))
    print('\tmax(num_users_per_item): {}'.format(ui_data['count'].max()))
    del ui_data['count']

    user_cnt = ui_data[['uid', 'iid']].groupby('uid').count().reset_index()
    user_cnt.rename(columns={'iid': 'count'}, inplace=True)
    user_cnt = user_cnt[user_cnt['count'] >= num_items_per_user_lb]
    user_cnt = user_cnt[user_cnt['count'] <= num_items_per_user_ub]
    ui_data = pd.merge(user_cnt, ui_data, on='uid', how='inner')
    print('\tmin(num_items_per_user): {}'.format(ui_data['count'].min()))
    print('\tmax(num_items_per_user): {}'.format(ui_data['count'].max()))
    del ui_data['count']

    # Reindex
    user_id = ui_data[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ui_data = pd.merge(ui_data, user_id, on=['uid'], how='left')

    item_id = ui_data[['iid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ui_data = pd.merge(ui_data, item_id, on=['iid'], how='left')

    ui_data_cols = ['userId', 'itemId', 'timestamp']
    ui_data = ui_data[ui_data_cols]
    end = datetime.now()
    print('\tSuccessfully filter the data, {} mins'.format((end - start).seconds / 60.0))
    print('\tRange of userId is [{}, {}]'.format(ui_data.userId.min(), ui_data.userId.max()))
    print('\tRange of itemId is [{}, {}]'.format(ui_data.itemId.min(), ui_data.itemId.max()))
    print('\tNumber of interactions is {}'.format(len(ui_data)))
    print('\tSparsity is {}'.format(len(ui_data) * 1.0 / ((ui_data.userId.nunique()) * (ui_data.itemId.nunique()))))
    return ui_data


def sample_negative(ui_history, interact_history, item_pool):
    """Effective sampling strategy for sparse datasets
    args:
        interact_history: interact history up to the time of ui_history(ui_history included)
        ui_history: pd.DataFrame, columns=['userId', 'itemId', 'interacted_items',
                                           'negative', 'continue']
        item_pool: all the item space
    """
    ui_history['negative_items'] = np.random.choice(item_pool, len(ui_history), replace=True)
    res = pd.merge(interact_history[['userId', 'itemId']],
                   ui_history[['userId', 'negative_items']],
                   left_on=['userId', 'itemId'],
                   right_on=['userId', 'negative_items'],
                   how='inner')
    if len(res) > 0:
        res['continue'] = True
        ui_history = pd.merge(ui_history,
                              res[['userId', 'negative_items', 'continue']],
                              on=['userId', 'negative_items'],
                              how='left').fillna(False)
    else:
        ui_history['continue'] = False
    # condition signaling continue sampling
    cont = (ui_history['continue'] == True)
    while len(ui_history[cont]) > 0:
        print('\tNumber of re-sample: {}'.format(len(ui_history[cont])))

        del ui_history['continue']  # delete old continue label
        ui_history.loc[cont, 'negative_items'] = np.random.choice(item_pool,
                                                                  len(ui_history[cont]),
                                                                  replace=True)
        res = pd.merge(interact_history[['userId', 'itemId']],
                       ui_history.loc[cont, ['userId', 'negative_items']],
                       left_on=['userId', 'itemId'],
                       right_on=['userId', 'negative_items'],
                       how='inner')
        if len(res) > 0:
            res['continue'] = True
            ui_history = pd.merge(ui_history,
                                  res[['userId', 'negative_items', 'continue']],
                                  on=['userId', 'negative_items'],
                                  how='left').fillna(False)
        else:
            ui_history['continue'] = False
        cont = ui_history['continue'] == True
    # ui_history['negative_items'] = ui_history['negative_items'].apply(lambda x: [x])
    del ui_history['continue']
    print(ui_history.columns)
    return ui_history


class UserItemDataset(Dataset):
    def __init__(self, user_tensor, item_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class UserPosNegDataset(Dataset):
    """Wrapper, convert <user, postive item, negative item> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, pos_item, neg_item):
        self.user_tensor = user_tensor
        self.pos_item = pos_item
        self.neg_item = neg_item

    def __getitem__(self, index):
        return self.user_tensor[index], self.pos_item[index], self.neg_item[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    def __init__(self, opt):
        data_path = opt['data_path']
        data_type = opt['data_type']
        filtered_path = opt['filtered_data_path']  # preprocessed
        # cache_ui_path = opt['cache_ui_split_path']

        if os.path.exists(filtered_path) and not opt['reconstruct_data']:
            ui_history = pd.read_csv(filtered_path,
                                     parse_dates=['timestamp'],
                                     engine='python', )
            print('Load filtered data in {}'.format(filtered_path))
        else:
            ui_history = load_ui_data(data_type, data_path)
            ui_history = filter_ui_data(ui_history,
                                        num_users_per_item_lb=opt['item_freq_threshold_lb'],
                                        num_users_per_item_ub=opt['freq_threshold_ub'],
                                        num_items_per_user_lb=opt['user_freq_threshold_lb'],
                                        num_items_per_user_ub=opt['freq_threshold_ub'])
            if filtered_path is not None:
                ui_history.to_csv(filtered_path, header=True, index=False)
                print('Saving filtered data in {}'.format(filtered_path))

        assert 'userId' in ui_history.columns
        assert 'itemId' in ui_history.columns

        self.opt = opt
        self.ui_history = ui_history
        self.num_users = self.ui_history['userId'].nunique()
        self.num_items = self.ui_history['itemId'].nunique()
        self.user_pool = range(self.num_users)
        self.item_pool = range(self.num_items)

        self.batch_size_train = opt.get('batch_size_train')
        self.batch_size_valid = opt.get('batch_size_valid')
        self.batch_size_test = opt.get('batch_size_test')

        # train/valid/test split
        if opt['train_test_split'] == 'loo':
            self.train_ui_history, self.test_ui_history = split_loo(ui_history, latest_n=opt['test_latest_n'])
            self.train_ui_history, self.valid_ui_history = split_loo(self.train_ui_history,
                                                                     latest_n=opt['valid_latest_n'])
        elif opt['train_test_split'] == 'lro':
            self.train_ui_history, self.test_ui_history = split_lro(ui_history, test_ratio=opt['test_ratio'])
            self.train_ui_history, self.valid_ui_history = split_lro(self.train_ui_history,
                                                                     test_ratio=opt['valid_ratio'])
        elif opt['train_test_split'] == 'freq-wise':
            self.train_ui_history, self.test_ui_history = split_freq_wise(ui_history,
                                                                          freq_bd=opt['train_test_freq_bd'],
                                                                          latest_n_low_freq=opt['test_latest_n'],
                                                                          test_ratio_high_freq=opt['test_ratio'],
                                                                          is_random=opt['random_split'])
            self.train_ui_history, self.valid_ui_history = split_freq_wise(self.train_ui_history,
                                                                           freq_bd=opt['train_valid_freq_bd'],
                                                                           latest_n_low_freq=opt['valid_latest_n'],
                                                                           test_ratio_high_freq=opt['valid_ratio'],
                                                                           is_random=opt['random_split'])

        print('\tNum of train records: {}'.format(len(self.train_ui_history)))
        print('\tNum of valid records: {}'.format(len(self.valid_ui_history)))
        print('\tNum of test records: {}'.format(len(self.test_ui_history)))
        print('\tNum of unique items in train history {}'.format(self.train_ui_history['itemId'].nunique()))
        print('\tNum of unique items in valid history {}'.format(self.valid_ui_history['itemId'].nunique()))
        print('\tNum of unique items in test history {}'.format(self.test_ui_history['itemId'].nunique()))
        
        print( len(set(self.test_ui_history['itemId'].unique().tolist()) - set(self.train_ui_history['itemId'].unique().tolist())))
        print( len(set(self.valid_ui_history['itemId'].unique().tolist()) - set(self.train_ui_history['itemId'].unique().tolist())))
        print( len(set(self.test_ui_history['itemId'].unique().tolist()) - set(self.valid_ui_history['itemId'].unique().tolist())))
        
        # print('Caching train/valid/test user,item history ...')
        # self.train_ui_history.to_csv(cache_ui_path.format('train'), index=False, header=True)
        # self.valid_ui_history.to_csv(cache_ui_path.format('valid'), index=False, header=True)
        # self.test_ui_history.to_csv(cache_ui_path.format('test'), index=False, header=True)
        
        interact_history = self.train_ui_history
        # interact history up to the train period
        self.train_interact_history = interact_history
        # interacted_items up to the train period
        self.train_interact_status = interact_history.groupby('userId')['itemId'].apply(set).reset_index() \
            .rename(columns={'itemId': 'interacted_items'})

        interact_history = pd.concat([self.valid_ui_history, self.train_ui_history])
        self.valid_interact_history = interact_history
        self.valid_interact_status = interact_history.groupby('userId')['itemId'].apply(set).reset_index() \
            .rename(columns={'itemId': 'interacted_items'})
        self.valid_user_freq = interact_history.groupby('userId')['itemId'].count().reset_index() \
             .rename(columns={'itemId': 'freq'})
        self.valid_item_freq = interact_history.groupby('itemId')['userId'].count().reset_index() \
             .rename(columns={'userId': 'freq'})

        interact_history = pd.concat([self.test_ui_history, self.valid_ui_history, self.train_ui_history])
        self.test_interact_history = interact_history
        self.test_interact_status = interact_history.groupby('userId')['itemId'].apply(set).reset_index() \
            .rename(columns={'itemId': 'interacted_items'})
        self.test_item_freq = interact_history.groupby('itemId')['userId'].count().reset_index() \
             .rename(columns={'userId': 'freq'})
        self.valid_item_freq = pd.merge(self.test_item_freq[['itemId']], 
                                        self.valid_item_freq, 
                                        on=['itemId'],
                                        how='left').fillna(0)
        self._train_epoch = iter([])
        self._valid_epoch = iter([])

        self.num_batches_train = math.ceil(len(self.train_ui_history) / self.batch_size_train)
        self.num_batches_valid = math.ceil(len(self.valid_ui_history) / self.batch_size_valid)

    @property
    def train_epoch(self):
        """list of training batches"""
        return self._train_epoch

    @train_epoch.setter
    def train_epoch(self, new_epoch):
        self._train_epoch = new_epoch

    @property
    def valid_epoch(self):
        """list of validation batches"""
        return self._valid_epoch

    @valid_epoch.setter
    def valid_epoch(self, new_epoch):
        self._valid_epoch = new_epoch

    def set_epoch(self, type):
        raise NotImplementedError

    def get_epoch(self, type):
        """

        return:
            list, an epoch of batchified samples of type=['train', 'valid']
        """
        if type == 'train':
            return self.train_epoch

        if type == 'valid':
            return self.valid_epoch

    def get_sample(self, type):
        """get training sample or validation sample"""
        # start = datetime.now()
        epoch = self.get_epoch(type)

        try:
            sample = next(epoch)
        except StopIteration:
            print('Generate new epoch, sample new negative items!')
            self.set_epoch(type)
            epoch = self.get_epoch(type)
            sample = next(epoch)
            if self.opt['load_in_queue']:
                # continue to queue
                self.cont_queue(type)
        # end = datetime.now()
        # print('Get {} sample time {}'.format(type, (end - start).total_seconds()))
        return sample


class MFSampleGenerator(SampleGenerator):
    def __init__(self, opt):
        """
        args:
            ui_history: pd.DataFrame, history of user-item interactions,
                        which contains 3 columns = ['userId', 'itemId', 'timestamp']
        """
        super(MFSampleGenerator, self).__init__(opt)

        print('Initial test data for evaluation ...')
        start = datetime.now()
        self.test_ui_uj_df = self.set_uij('test',
                                          sample_or_not=False,
                                          ui_uj_format=True)
        end = datetime.now()
        print('\tInitial test data for evaluation, {} mins'.format((end - start).total_seconds() / 60))

    # @profile
    def set_uij(self, type, sample_or_not=True, ui_uj_format=False):
        """create (u,i,j) tuples from interaction history and negative samples

        args:
            num_negatives: 1 for BPR, 99 for evaluation on test
            sample_or_not: sample 1 negative items as BPR required,
                           or use the entire negative items
                           note: Slow!
            ui_uj_format: ((u,i), (u,j)) tensor or (u,i,j) batch
        """
        if type == 'train':
            ui_history = self.train_ui_history
            interact_history = self.train_interact_history
            interact_status = self.train_interact_status
        elif type == 'valid':
            ui_history = self.valid_ui_history
            interact_history = self.valid_interact_history
            interact_status = self.valid_interact_status
        elif type == 'test':
            ui_history = self.test_ui_history
            interact_history = self.test_interact_history
            interact_status = self.test_interact_status
        else:
            raise Exception('Invalid data type')

        # Generate negative samples
        start = datetime.now()
        ui_history = pd.merge(ui_history, interact_status, on=['userId'], how='left')
        if sample_or_not is True:
            ui_history = sample_negative(ui_history, interact_history, self.item_pool)
        else:
            # do not sample, should return all the negatives
            # resolve: return interacted_items and test_items
            ui = ui_history.groupby('userId')['itemId'].apply(list).reset_index() \
                .rename(columns={'itemId': 'pos_items'})
            ui_uj = pd.merge(ui, interact_status, on=['userId'])

        end = datetime.now()
        print('\tNegative sample time {}'.format((end - start).total_seconds() / 60))
        if ui_uj_format:
            # return pos_items, interacted_items(used to compute neg_items)
            ui_uj.loc[:, 'num_pos_items'] = ui_uj['pos_items'].apply(len)
            ui_uj.sort_values('num_pos_items', ascending=False, inplace=True)
            del ui_uj['num_pos_items']
            return ui_uj
        else:
            uij = ui_history[['userId', 'itemId', 'negative_items']]
            user_tensor = torch.LongTensor(uij.userId.values)
            pos_item_tensor = torch.LongTensor(uij.itemId.values)
            neg_item_tensor = torch.LongTensor(uij.negative_items.values)
            return user_tensor, pos_item_tensor, neg_item_tensor

    def set_epoch(self, type):
        """setup batches of type = [training, validation]"""
        print('\tSetting epoch {}'.format(type))
        start = datetime.now()
        if type == 'train':
            user_tensor, pos_item_tensor, neg_item_tensor = self.set_uij('train', sample_or_not=True,
                                                                         ui_uj_format=False)
            uij = UserPosNegDataset(user_tensor, pos_item_tensor, neg_item_tensor)
            uij_loader = DataLoader(uij,
                                    batch_size=self.batch_size_train,
                                    shuffle=True, pin_memory=False)
            self.train_epoch = iter(uij_loader)
            num_batches = len(self.train_epoch)
        elif type == 'valid':
            user_tensor, pos_item_tensor, neg_item_tensor = self.set_uij('valid', sample_or_not=True,
                                                                         ui_uj_format=False)
            uij = UserPosNegDataset(user_tensor, pos_item_tensor, neg_item_tensor)
            uij_loader = DataLoader(uij,
                                    batch_size=self.batch_size_valid,
                                    shuffle=True, pin_memory=False)
            self.valid_epoch = iter(uij_loader)
            num_batches = len(self.valid_epoch)
        end = datetime.now()
        print('\tFinish setting epoch {}, num_batches {}, time {} mins'.format(type,
                                                                               num_batches,
                                                                               (end - start).total_seconds() / 60))