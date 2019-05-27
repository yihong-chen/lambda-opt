from engine import setup_args, Engine


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        alias='food',
        tensorboard='./tmp/runs/amazon-food/MF/alter/',
        regularizer='alter_mf',
        ##########
        ## data ##
        ##########
        reconstruct_data=True,
        data_type='amazon-food-mf',
        data_path='./data/amazon-food/Reviews.csv',
        filtered_data_path='./tmp/data/amazon-food-mf-filter_u20i20.dat',
        cache_ui_split_path='./tmp/data/amazon-food-mf-filter_u20i20/{}.csv',
        eval_res_path='./tmp/res/amazon-food/{alias}/{epoch_idx}.csv',
        penalty_param_path='./tmp/penalty/amazon-food/{alias}/{epoch_idx}.csv',
        item_freq_threshold_lb=20, # 10
        user_freq_threshold_lb=20, # 3
        freq_threshold_ub=int(1e9),
        metric_topk=100,
        ######################
        ## train/test split ##
        ######################
        train_test_split='lro',
        test_ratio=0.2,
        valid_ratio=0.25,
        ######################
        ##########################
        ## Devices & Efficiency ##
        ##########################
        use_cuda=True,
        # max_steps=1140,
        log_interval=1, # 816
        eval_interval=1, # 10 epochs between 2 evaluations
        multi_cpu_train=False,
        num_workers_train=1,
        multi_cpu_valid=False,
        num_workers_valid=1,
        multi_cpu_test=True,
        num_workers_test=6,
        device_ids_test=[3], # 2.5 minutes for single gpu
        device_id=1,
        batch_size_train=1024,
        batch_size_valid=1024,
        batch_size_test=1024,
        # batch_size_train=10000,
        # batch_size_valid=10000,
        # batch_size_test=10000,
        num_negatives=1,
        ###########
        ## Model ##
        ###########
        latent_dim=128,
        mf_lr=1e-3,
        mf_optimizer='adam', # 'sgd'
        mf_amsgrad=False,
        mf_eps=1e-8,
        mf_l2_regularization=0,
        mf_betas=(0.9, 0.999),
        mf_grad_clip=100, #0.1
        mf_lr_exp_decay=1,
        lambda_update_interval=1, # 1
        # lambda_network_grad_clip=100,
        lambda_network_grad_clip=1e-3,
        lambda_network_init=0,
        lambda_network_type='dimension+user+item',
        # lambda_network_init=1e-1,
        # lambda_network_type='user+item',
        # lambda_network_type='user-wise',
        # lambda_network_type='item-wise',
        # lambda_network_type='dimension-wise',
        # lambda_network_type='dimension+user',
        # lambda_network_type='dimension+item',
        # lambda_network_lr=1e-4,
        # lambda_network_lr=1e-3,
        lambda_network_lr=1e-2,
        # lambda_network_optimizer='sgd',
        lambda_network_optimizer='adam',
        # lambda_network_dp_prob=0.1,
        lambda_network_dp_prob=0,
        lambda_network_amsgrad=True,
        lambda_network_l2_regularization=0,
        lambda_network_betas=(0.9, 0.999),
        lambda_network_momentum=0,
        lambda_network_lr_exp_decay=1,
    )

    opt = parser.parse_args(args=[])
    opt = vars(opt)

    # rename alias
    opt['alias'] = opt['alias'] + 'Top{}_K{}_MFlr{}_{}_{}_PNlr{}_{}_{}_type.{}' \
                                  '_amsgrad{}_l2{}_ed{}.dp{}_li{}_init{}'.format(
                    opt['metric_topk'],
                    opt['latent_dim'],
                    opt['mf_lr'],
                    opt['mf_optimizer'],  
                    opt['mf_grad_clip'],
                    opt['lambda_network_lr'],
                    opt['lambda_network_optimizer'],
                    opt['lambda_network_grad_clip'],
                    opt['lambda_network_type'],
                    opt['lambda_network_amsgrad'],
                    opt['lambda_network_l2_regularization'],
                    opt['lambda_network_lr_exp_decay'],
                    opt['lambda_network_dp_prob'],
                    opt['lambda_update_interval'],
                    opt['lambda_network_init'])
    print(opt['alias'])

    engine = Engine(opt)
    engine.train()