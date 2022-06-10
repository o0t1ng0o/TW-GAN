import time, os
from tensorboardX import SummaryWriter
import shutil
import torch
class LogMaker():
    def __init__(self, opt, config_file):
        self.dilation_list = opt.dilation_list
        self.use_centerness = opt.use_centerness
        # summary
        now = int(time.time())
        timeArray = time.localtime(now)
        log_folder = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
        self.log_folder = os.path.join('log', log_folder)
        self.result_folder = os.path.join(self.log_folder, 'running_result')
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.writer = SummaryWriter(self.log_folder)
        # record the hyperparameters
        shutil.copy(config_file, os.path.join(self.log_folder, 'config.py'))
        shutil.copy('./models/tw_gan.py', os.path.join(self.log_folder,'tw_gan.py'))
        shutil.copy('./loss.py', os.path.join(self.log_folder,'loss.py'))
    def write(self, step, losses):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, step)


    def draw_prediction(self, pred, targs, centerness_preds, centerness_targs, step):
        target_artery = targs[0:1, 0, :, :]
        target_vein = targs[0:1, 1, :, :]
        target_all = targs[0:1, 2, :, :]

        pred_sigmoid = pred  # nn.Sigmoid()(pred)

        self.writer.add_image('artery', torch.cat([pred_sigmoid[0:1, 0, :, :], target_artery], dim=1), global_step=step)
        self.writer.add_image('vessel', torch.cat([pred_sigmoid[0:1, 1, :, :], target_all], dim=1), global_step=step)
        self.writer.add_image('vein', torch.cat([pred_sigmoid[0:1, 2, :, :], target_vein], dim=1), global_step=step)

        if self.use_centerness:
            center_artery = centerness_preds[0:1, 0, :, :]
            center_vein   = centerness_preds[0:1, 1, :, :]
            center_vessel = centerness_preds[0:1, 2, :, :]

            center_artery_targ = centerness_targs[0:1, 0, :, :]
            center_vein_targ   = centerness_targs[0:1, 1, :, :]
            center_vessel_targ = centerness_targs[0:1, 2, :, :]

            bs, ch, h, w = centerness_preds.shape
            

            self.writer.add_image('centerness_pred_dil=0', torch.cat([center_artery, center_vein, center_vessel], dim=1), global_step=step)
            self.writer.add_image('centerness_targ_dil=0', torch.cat([center_artery_targ, center_vein_targ, center_vessel_targ], dim=1), global_step=step)
            
            if ch//3 >= 2:
                self.writer.add_image('centerness_pred_dil=' + str(self.dilation_list[1]), torch.cat([centerness_preds[0:1, 3, :, :], centerness_preds[0:1, 4, :, :], centerness_preds[0:1, 5, :, :]], dim=1), global_step=step)
                self.writer.add_image('centerness_targ_dil=' + str(self.dilation_list[1]), torch.cat([centerness_targs[0:1, 3, :, :], centerness_targs[0:1, 4, :, :], centerness_targs[0:1, 5, :, :]], dim=1), global_step=step)
            
            if ch//3 >= 3:
                self.writer.add_image('centerness_pred_dil=' + str(self.dilation_list[2]), torch.cat([centerness_preds[0:1, 6, :, :], centerness_preds[0:1, 7, :, :], centerness_preds[0:1, 8, :, :]], dim=1), global_step=step)
                self.writer.add_image('centerness_targ_dil=' + str(self.dilation_list[2]), torch.cat([centerness_targs[0:1, 6, :, :], centerness_targs[0:1, 7, :, :], centerness_targs[0:1, 8, :, :]], dim=1), global_step=step)



    def print(self, losses, step):
        print_str = '{} step -'.format(step)
        for k, v in losses.items():
            print_str += ' {}:{:.4}'.format(k,v)
        print(print_str)
