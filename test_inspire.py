# -*- coding: utf-8 -*-
"""
@author: wenting
"""
from opt import modelEvalution_inspire
import imp
import os
import torch
def test(cfg):
    cfg.dataset_name = 'INSPIRE'
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution_inspire(cfg.model_step_pretrained_G, net,
                       result_folder, 0,
                       use_cuda=cfg.use_cuda,
                       input_ch=cfg.input_nc,
                       config=cfg,
                       strict_mode=False)
if __name__ == '__main__':
    config = imp.load_source('config','./config/config_test_AV_DRIVE.py')
    test(config);
