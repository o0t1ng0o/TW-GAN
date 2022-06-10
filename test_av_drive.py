# -*- coding: utf-8 -*-
"""
@author: wenting
"""
import os
from opt import modelEvalution
import torch
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
    from config import config_test_AV_DRIVE as cfg
    test(cfg);
