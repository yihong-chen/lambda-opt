from engine import setup_args, Engine


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        alias='20-corev0.2',
        tensorboard='./tmp/runs/amazon-food/MF/fixed/',
        regularizer = 'fixed',
        ##########
        ## data ##
        ##########
        reconstruct_data=True,
        data_type='amazon-food-mf',
        data_path='./data/amazon-food/Reviews.csv',
        load_in_queue=False,
        filtered_data_path='./tmp/data/amazon-food-mf-filter_u20i20.dat',
        eval_res_path='./tmp/res/amazon-food/{alias}/{epoch_idx}.csv',
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
        ## train/test split ##
        ######################
        # train_test_split='freq-wise',  # freq-wise
        # train_test_freq_bd=5,
        # test_latest_n=1,
        # test_ratio=0.2,
        # train_valid_freq_bd=4,
        # valid_ratio=0.25, #1
        # valid_latest_n=1,
        ##########################
        ## Devices & Efficiency ##
        ##########################
        use_cuda=True,
        log_interval=1, # 816
        eval_interval=20, # 10 epochs between 2 evaluations
        multi_cpu_train=False,
        num_workers_train=1,
        multi_cpu_valid=False,
        num_workers_valid=1,
        multi_cpu_test=True,
        num_workers_test=8,
        device_ids_test=[0], # 2.5 minutes for single gpu
        device_id=0,
        batch_size_train=1024,
        batch_size_valid=1024,
        batch_size_test=1024,
        num_negatives=1,
        ###########
        ## Model ##
        ###########
        fixed_lambda_candidate=[1e-5],
        latent_dim=128,
        mf_lr=1e-3,
        mf_optimizer='adam',
        mf_amsgrad=False,
        mf_eps=1e-8,
        mf_l2_regularization=0,
        mf_betas=(0.9, 0.999),
        mf_grad_clip=100,#0.1
        mf_lr_exp_decay=1,
        lambda_update_interval=1,
    )

    opt = parser.parse_args(args=[])
    opt = vars(opt)

    # rename alias
    opt['alias'] = opt['alias'] + 'Top{}_{}_lambda{}_K{}_mflr_{}_mfoptim{}'.format(
                                                           opt['metric_topk'],
                                                           opt['regularizer'],
                                                           opt['fixed_lambda_candidate'][0],
                                                           opt['latent_dim'],
                                                           opt['mf_lr'],
                                                           opt['mf_optimizer'])

    engine = Engine(opt)
    engine.train()