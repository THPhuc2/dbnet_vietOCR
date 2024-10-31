import os
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from datetime import datetime

config = Cfg.load_config_from_name('vgg_transformer')
dataset_params = {
    'name':'hw',
    'data_root':'../../data/recog_data_3/',
    'train_annotation':'train.txt',
    'valid_annotation':'val.txt'
}

params = {
    'print_every':200,
    'valid_every':15*200,
    'iters':80000,
    'checkpoint':'./weights/transformerocr_ads_1.pth',    
    'export':'./weights/transformerocr_ads_last.pth',
    'metrics': 10000,
    'batch_size': 16
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

trainer = Trainer(config, pretrained=True)
trainer.config.save('config_final.yml')
trainer.train()
trainer.precision()