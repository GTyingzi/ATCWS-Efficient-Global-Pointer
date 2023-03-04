import time

common = {
    "exp_name": "CCKS2019",
    "encoder": "BERT",
    "data_home": "./datasets",
    "bert_path": "hfl/chinese-bert-wwm",  # bert-base-cased， bert-base-chinese
    "attack_train": "False", # 是否添加对抗训练 True,False
    "run_type": "test",  # train,test
    "f1_2_save": 0.9,  # 存模型的最低f1值
    "logger": "wandb"  # wandb or default，default意味着只输出日志到控制台
}

# wandb的配置，只有在logger=wandb时生效。用于可视化训练过程
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

train_config = {
    "train_data": "train.json",
    "ent2id": "ent2id.json",
    "path_to_save_model": "./outputs",  # 在logger不是wandb时生效
    "hyper_parameters": {
        "lr": 5e-5,
        "batch_size": 16,
        "epochs": 20,
        "seed": 2,
        "max_seq_len": 128,
        "scheduler": "CAWR",
        "attack" : "fgm" # fgm,fgsm,pgd
    }
}

eval_config = {
    # "model_state_dir": "./wandb/CCKS2019/seed_2333/EffiGlobalPointer_token_fgm/files",  # 预测时注意填写模型路径（时间tag文件夹）
    "model_state_dir": "./wandb/EffiGlobalPointer_token/files",  # 预测时注意填写模型路径（时间tag文件夹）
    "run_id": "",
    "last_k_model": 1,  # 取倒数第几个model_state
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "batch_size": 16,
        "max_seq_len": 128,
    }

}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------------------------
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}
