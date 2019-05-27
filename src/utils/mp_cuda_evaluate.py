""" Multiprocessing + CUDA for evaluation over all users
"""
import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

import sys
from copy import deepcopy
import numpy as np
import pandas as pd
from math import ceil
from datetime import datetime
import time
import GPUtil
# import line_profiler

DUMP_ITEM_WISE = False # set it to true if you want to dump results for each user
mp.set_start_method('spawn', force=True)  # forkserver


def measure_user_wise_vectorized(top_k, pos_scores, neg_scores):
    """ vectorized version, address long-tail head user
    args:
        pos_scores: numpy array, shape (num_pos, )
        neg_scores: numpy array, shape (num_pos, num_neg) FM
                                 shape (num_pos, ) MF
    """
    num_pos = pos_scores.shape[0]

    pos_scores = pos_scores.reshape(num_pos, 1) # for np broadcast
    # (pos_scores > neg_scores) shape (num_pos, num_neg)
    auc = (pos_scores > neg_scores).mean()

    pos_ranks = (pos_scores < neg_scores).sum(axis=1) + 1
    pos_top_k = pos_ranks[pos_ranks <= top_k]
    hr = pos_top_k.shape[0] / num_pos
    ndcg = (np.log(2) / np.log(1 + pos_top_k)).sum() / num_pos
    return hr, ndcg, auc


def measure_ui_wise_vectorized(top_k, pos_scores, neg_scores):
    """ vectorized version, address long-tail head user
    user's all pos_item
    args:
        pos_scores: numpy array, shape (num_pos, )
        neg_scores: numpy array, shape (num_pos, num_neg) FM
                                 shape (num_pos, ) MF
    """
    num_pos = pos_scores.shape[0]
    pos_scores = pos_scores.reshape(num_pos, 1) # for np broadcast
    # (pos_scores > neg_scores) shape (num_pos, num_neg)
    acc = (pos_scores > neg_scores).sum(axis=1)

    pos_ranks = (pos_scores < neg_scores).sum(axis=1) + 1
    hr_item_wise = (pos_ranks <= top_k)
    ndcg_item_wise = (np.log(2) / np.log(1 + pos_ranks)) * hr_item_wise
    return acc, hr_item_wise, ndcg_item_wise


def score_ui_tensor_mf(model, ui_tensor, use_cuda, device_ids):
    """score ui_tensor in cpu, don't batch since memory is big enough
    """
    current = mp.current_process()
    worker_id = current._identity[0]  # get worker_id, no so secure
    device_id = device_ids[worker_id % len(device_ids)]
    user_tensor, item_tensor = ui_tensor
    if use_cuda:
        with torch.cuda.device(device_id):
            model.cuda()
            user_tensor = user_tensor.cuda()
            item_tensor = item_tensor.cuda()
    score_tensor = model(user_tensor, item_tensor)
    if score_tensor.is_cuda:
        user_tensor = user_tensor.cpu()
        item_tensor = item_tensor.cpu()
        score_tensor = score_tensor.cpu()
    users = user_tensor.data.view(-1).numpy()
    items = item_tensor.data.view(-1).numpy()
    scores = score_tensor.data.view(-1).numpy()
    return users, items, scores
    # return user_tensor.view(-1), item_tensor.view(-1), score_tensor.view(-1)


# @profile
def evaluate_user_wise_mf(model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k):
    start = datetime.now()

    uid, pos_items, interacted_items = ui_uj_grp[0], ui_uj_grp[1], ui_uj_grp[2]
    uid = np.array(uid)
    assert isinstance(pos_items, list)

    pos_u_tensor = torch.LongTensor(uid.repeat(len(pos_items)))
    pos_i_tensor = torch.LongTensor(pos_items)

    neg_items = []
    for i in item_pool:
        if not (i in interacted_items):
            neg_items.append(i)

    neg_u_tensor = torch.LongTensor(uid.repeat(len(neg_items)))
    neg_i_tensor = torch.LongTensor(neg_items)

    pos_user, pos_item, pos_score = score_ui_tensor_mf(model, [pos_u_tensor, pos_i_tensor],
                                                       use_cuda, device_ids)
    neg_user, neg_item, neg_score = score_ui_tensor_mf(model, [neg_u_tensor, neg_i_tensor],
                                                       use_cuda, device_ids)
    hit_ratio, ndcg, auc = measure_user_wise_vectorized(metron_top_k, pos_score, neg_score)
    end = datetime.now()
    dur = (end - start).total_seconds()
    num_items = len(pos_items)
    # print('[Evaulate] user {}, #(items) {}, time {} seconds, '
    #   'hr {}, ndcg {}, auc {}'.format(uid, num_items, dur, hit_ratio, ndcg, auc))
    return uid, hit_ratio, ndcg, auc, num_items, dur


def evaluate_ui_wise_mf(model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k):
    start = datetime.now()

    uid, pos_items, interacted_items = ui_uj_grp[0], ui_uj_grp[1], ui_uj_grp[2]
    uid = np.array(uid)
    assert isinstance(pos_items, list)

    pos_u_tensor = torch.LongTensor(uid.repeat(len(pos_items)))
    pos_i_tensor = torch.LongTensor(pos_items)

    neg_items = []
    for i in item_pool:
        if not (i in interacted_items):
            neg_items.append(i)

    neg_u_tensor = torch.LongTensor(uid.repeat(len(neg_items)))
    neg_i_tensor = torch.LongTensor(neg_items)

    pos_user, pos_item, pos_score = score_ui_tensor_mf(model, [pos_u_tensor, pos_i_tensor],
                                                       use_cuda, device_ids)
    neg_user, neg_item, neg_score = score_ui_tensor_mf(model, [neg_u_tensor, neg_i_tensor],
                                                       use_cuda, device_ids)
    acc, hr_itemwise, ndcg_itemwise = measure_ui_wise_vectorized(metron_top_k, pos_score, neg_score)
    end = datetime.now()
    dur = np.array((end - start).total_seconds()).repeat(len(pos_items))
    # print('[Evaulate] user {}, #(items) {}, time {} seconds, '
    #   'hr {}, ndcg {}, auc {}'.format(uid, num_items, dur, hit_ratio, ndcg, auc))
    return uid.repeat(len(pos_items)), np.array(pos_items), acc, hr_itemwise, ndcg_itemwise, dur


def wrapper_mf(x):
    model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k = x
    return evaluate_user_wise_mf(model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k)


def wrapper_mf_ui(x):
    model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k = x
    return evaluate_ui_wise_mf(model, ui_uj_grp, item_pool, use_cuda, device_ids, metron_top_k)


def evaluate_ui_uj_df(model, model_name,
                      card_feat, ui_uj_df, item_pool,
                      metron_top_k, eval_res_path,
                      use_cuda, device_ids, num_workers):
    """

    args:
        model: MF or FM or ...
        model_name: 'mf' or 'fm'
        card_feat: for fm models,list of cardinality for each feat
                   this is necessary for generating the feature embedding
                   the order should correspond to ui_uj_df
                   eg. 0: user, 1:item, 2:prev_item,
                      -1: interacted_items(only used for generating negative smaples)
        ui_uj_df: pandas DataFrame, userId, pos_items, interacted_items
        item_pool: all available items
        metron_top_k: Top K for NDCG and HR
        eval_res_path: path for saving the evaluate results
        num_workers: >1
    """
    start = datetime.now()
    model = deepcopy(model)
    model.eval()
    if next(model.parameters()).is_cuda:
        model.cpu()  # move it to cpu in order to distribute the model parallel
    # userId, pos_items, interacted_items
    # userId, pos_items, prev_items, interacted_items
    gpu_avai = False
    while gpu_avai is False:
        available_device_ids = GPUtil.getAvailable(order = 'first', limit = 4,
                                    maxLoad = 0.6, maxMemory = 0.6,
                                    excludeID=[], excludeUUID=[])
        gpu_avai = set(available_device_ids).issuperset(set(device_ids))
        if gpu_avai is False:
            print(available_device_ids, device_ids)
            print('\tWaiting for all device_ids available ...')
            time.sleep(60)
    print('\tGPUs are ready for evaluation!')
    num_users_per_worker = ceil(len(ui_uj_df) / num_workers)
    with mp.Pool(processes=num_workers) as p:
        if model_name == 'mf':
            ui_uj_df = ui_uj_df[['userId', 'pos_items', 'interacted_items']]
            ui_uj_array = ui_uj_df.sample(frac=1).values # np.array saves more space!!! avoid dataframe here
            jobs = [(model, group, item_pool,
                     use_cuda, device_ids, metron_top_k) for group in ui_uj_array]
            if DUMP_ITEM_WISE:
                res_list = p.map(wrapper_mf_ui, jobs, chunksize=num_users_per_worker)
            else:
                res_list = p.map(wrapper_mf, jobs, chunksize=num_users_per_worker)  # list of (uid, hr, ndcg, auc, acc) corresponding to each group
        elif model_name == 'fm':
            ui_uj_df = ui_uj_df[['userId', 'pos_items', 'prev_items', 'interacted_items']]
            ui_uj_array = ui_uj_df.sample(frac=1).values # shuffle to make it uniform for each worker
            jobs = [(model, group, card_feat, item_pool,
                     use_cuda, device_ids, metron_top_k) for group in ui_uj_array]
            if DUMP_ITEM_WISE:
                res_list = p.map(wrapper_fm_ui, jobs, chunksize=num_users_per_worker)
            else:
                res_list = p.map(wrapper_fm, jobs, chunksize=num_users_per_worker)
        else:
            raise NotImplementedError('Model {} not found!'.format(model_name))

        if DUMP_ITEM_WISE:
            uids = [x[0] for x in res_list]
            iids = [x[1] for x in res_list]
            accs = [x[2] for x in res_list]
            hrs = [x[3] for x in res_list]
            ndcgs = [x[4] for x in res_list]
            
            uids = np.concatenate(uids)
            iids = np.concatenate(iids)
            accs = np.concatenate(accs)
            hrs = np.concatenate(hrs)
            ndcgs = np.concatenate(ndcgs)
            res = pd.DataFrame({'userId': uids,
                        'itemId': iids,
                        'acc': accs,
                        'hr': hrs,
                        'ndcg': ndcgs})
        else:
            user = [x[0] for x in res_list]
            hr = [x[1] for x in res_list]
            ndcg = [x[2] for x in res_list]
            auc = [x[3] for x in res_list]
            num_items = [x[4] for x in res_list]
            run_time = [x[5] for x in res_list]
            res = pd.DataFrame({'userId': user,
                        'evaluate_time': run_time,
                        'num_test_items': num_items,
                        'hr': hr, 'ndcg': ndcg, 'auc': auc})
    print(res.head(10))
    res.sort_values('userId', inplace=True)
    res.to_csv(eval_res_path, index=False, header=True)
    end = datetime.now()
    print('Time {}'.format((end - start).total_seconds()))
    if DUMP_ITEM_WISE:
        hr = res.groupby('userId')['hr'].mean().reset_index()['hr'].mean()
        ndcg = res.groupby('userId')['ndcg'].mean().reset_index()['ndcg'].mean()
        num_correct = res.groupby('userId')['acc'].sum().reset_index().rename(columns={'acc': 'num_correct'})
        tmp = ui_uj_df[['userId', 'interacted_items', 'pos_items']]
        tmp['num_pos'] = tmp['pos_items'].apply(len)
        tmp['num_interacted'] = tmp['interacted_items'].apply(len)
        tmp['num_neg'] = len(item_pool) - tmp['num_interacted']
        tmp['num_tot'] = tmp['num_pos'] * tmp['num_neg']
        del tmp['interacted_items'], tmp['num_interacted']
        del tmp['pos_items'], tmp['num_pos']
        num_correct = pd.merge(num_correct, tmp, on='userId', how='inner')
        auc = (num_correct['num_correct'] / num_correct['num_tot']).mean()
        metrics = {'hr': hr,
                   'ndcg': ndcg,
                   'auc': auc}
    else:
        # aggregate over all users
        metrics = {'hr': res['hr'].mean(),
               'ndcg': res['ndcg'].mean(),
               'auc': res['auc'].mean()}
    return metrics
