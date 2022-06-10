# -*- coding: utf-8 -*-
"""
Created on 2020_06_30

@author: wenting
"""
import os
import torch
from opt import modelEvalution
def test(cfg):
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution(cfg.model_step_pretrained_G, net,
                       result_folder, 0,
                       use_cuda=cfg.use_cuda,
                       dataset=cfg.dataset_name,
                       input_ch=cfg.input_nc,
                       config=cfg,
                       strict_mode=True)

if __name__ == '__main__':
    from config import config_test_HRF as cfg
    test(cfg);
