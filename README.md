# lambda-opt
This repo provides our implementation for λOpt. λOpt is a fine-grained adaptive regularizer for recommender models. The key idea is to specify an individual regularization strength for each user and item. Check [our paper](https://arxiv.org/abs/1905.11596) for details about λOpt if you are interested!

> Yihong Chen, Bei Chen, Xiangnan He, Chen Gao, Yong Li, Jian-Guang Lou and Yue Wang(2019). λOpt: Learn to Regularize Recommender Models in Finer Levels. In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

# Usage

### 1. Put the data in `src/data/`
For example, movielens-1m, put `ratings.dat` in `src/data/ml-1m`
```
cd src
mkdir data
cd data 
mkdir ml-1m
```

### 2. Create folders for tmp files in `src/tmp/`
```
cd src
mkdir tmp
mkdir tmp/data
mkdir tmp/res
mkdir tmp/res/ml1m/
mkdir tmp/penalty/
mkdir tmp/penalty/ml1m/
```

### 3. Specify the hyper-parameters in `train_[dataset]_[regularization method].py`
For fixed lambda on movielens-1m, the hyper-parameters are in `train_ml1m_fixed.py`
For lambdaopt on movielens-1m, the hyper-parameters are in `train_ml1m_alter.py`

### 4. Train
```
cd src
python train_ml1m_alter.py
```

# Files

- `src/regularizer/`: lambdaopt
- `src/factorizer/`: matrix factorization model
- `src/utils/`: handy modules for data sampling, evaluation etc.
- `src/engine.py`: training engine
- `src/train_[dataset]_[regularization method].py`: entry point

# What's new?
λOpt features a parameterized regularizer, where very fine-grained regularization is employed over the user and item embeddings. 
While itself offers a novel technique to solve the bi-level optimization problem, there are also a bunch of cool work from the meta-learning community.
Hence, to put readers in a broader context, this repo also offers the implementation of λOpt using [Higher](https://github.com/facebookresearch/higher), a pytorch meta-learning repo. 
Higher provides useful tookits, like differentiable optimizers and automatic patch to make pytorch model functional, which facilitates convenient computation of higher-order gradients.

Comparison between the original λOpt and λOpt using higher is shown in the following table.

|                                          | λOpt Original       | λOpt Higher     |
|------------------------------------------|---------------------|-----------------|
| Gradient Path                            | Truncated 2nd-order | Fully 2nd-order |
| Memory Cost                              | 1X                  | ~1.2X           |
| Computational Cost                       | 1X                  | ~1.7X           |
| Require manual gradient composition?     | Yes                 | No              |
| Require duplicating model forward logic? | Yes                 | No              |
| Require manual differentiable optimizer? | Yes                 | No              |
| Concise code                             | No                  | Yes             |

Note that λOpt Higher takes more computational resources but offers a very concise and beautiful code:
```Python
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
```


How to run λOpt in higher?
- simply specify the `regularizer` as `alter_mf_higher`

# Requirement
- pytorch=1.2.0
- higher

# Citation
If you this repo, please kindly cite our paper.
```
@inproceedings{lambdaopt,
 author = {Chen, Yihong and Chen, Bei and He, Xiangnan and Gao, Chen and Li, Yong and Lou, Jian-Guang and Wang, Yue},
 title = {lambdaOpt: Learn to Regularize Recommender Models in Finer Levels},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 series = {KDD '19},
 year = {2019},
 isbn = {978-1-4503-6201-6},
 location = {Anchorage, AK, USA},
 pages = {978--986},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3292500.3330880},
 doi = {10.1145/3292500.3330880},
 acmid = {3330880},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {matrix factorization, regularization hyperparameter, top-k recommendation},
} 
```
# Contact
Any feedback is much appreciated! Drop us a line at yihong.chen@cs.ucl.ac.uk or simply open a new issue.

# TODO
- [ ] Test LambdaOpt in Higher


