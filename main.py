# -*- coding: utf-8 -*-
"""
Created on 2020_06_23

@author: wenting
"""
import os
import torch


from models.tw_gan import TW_GAN
from opt import Dataloader_DRIVE,Dataloader_HRF

import argparse
import imp
from Tools.utils import LogMaker


def train(opt, config_file):

    # load dataset
    train_data1, label_data, label_data_fake, label_centerness_maps, label_dilation_maps = \
                                                Dataloader_DRIVE(opt.trainset_path, opt.trainset_shuffle_path, \
                                                                     opt.trainset_centerness_path, dilation_list=opt.dilation_list, overlap=opt.overlap_vessel) \
                                                if opt.dataset_name=='DRIVE' else \
                                                 Dataloader_HRF(opt.trainset_path, opt.trainset_shuffle_path, \
                                                                              opt.trainset_centerness_path, dilation_list=opt.dilation_list, overlap=opt.overlap_vessel, k_fold_idx=opt.k_fold_idx, k_fold=opt.k_fold)

    # make log
    logger = LogMaker(opt, config_file)

    twgan = TW_GAN(opt)
    twgan.setup(opt,logger.log_folder)


    for i in range(opt.model_step, opt.max_step+1):

        twgan.set_input(i,
                        train_data1=train_data1,
                        label_data=label_data,
                        label_data_fake=label_data_fake,
                        label_data_centerness=label_centerness_maps,
                        label_data_dilation=label_dilation_maps)

        twgan.optimize_parameters()

        
        if i % opt.save_iter == 0 or (i >= 25000 and i % opt.print_iter == 0):
            twgan.save_model()

        losses = twgan.get_current_losses()
        logger.write(i, losses)
        # draw the predicted images in summary
        if i % opt.display_iter == 0:
            twgan.log(logger)
        
        if i % opt.print_iter == 0:
            logger.print(losses, i)
            if i >= (opt.max_step - 5000):
                twgan.test(logger.result_folder)
    logger.writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str, default='default')
    args = parser.parse_args()
    config = imp.load_source('config', args.config_file)

    train(config, args.config_file);
