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

# Requirement
- pytorch=0.4.0

# Citation
If you this repo, please kindly cite our paper.
```
@inproceedings{lambdaopt19,
  author = {Yihong Chen and
         Bei Chen and
         Xiangnan He and
         Chen Gao and
         Yong Li and
         Jian-Guang Lou and 
         Yue Wang},
  title = {λOpt: Learn to Regularize Recommender Models in Finer Levels},
  booktitle = {{KDD}},
  year  = {2019}
}
```
# Contact
Any feedback is much appreciated! Drop us a line at laceychen@outlook.com or simply open a new issue.

# TODO
- [ ] Upgrade to pytorch 1.0



