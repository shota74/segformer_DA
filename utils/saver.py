import os
import shutil
import torch
from collections import OrderedDict
import glob
import sys
import datetime

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
        # d_today = datetime.datetime.utcnow().date()
        # t_now = datetime.datetime.now().time()
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).time()
        #run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}_{}'.format(str(today),str(now)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)


    def save_images(self, image_name, image):
        name = image_name
        img = image
        if not os.path.exists(self.experiment_dir + '/result'):
            os.makedirs(self.experiment_dir + '/result')

        #print(img.shape)
        #sys.exit()



    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
    
    def save_checkpoint_r(self, state, is_best, filename='checkpoint_r.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred_r.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best_r.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best_r.pth.tar'))
    
    def save_checkpoint_last(self, state, is_best, filename='last.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred_last.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred_last.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'last.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'last.pth.tar'))


    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size
        #p['dataset'] = self.args.datasets_list

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()


    def save_checkpoint2(self, state, filename='checkpoint2.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        # best_pred = state['best_pred']
        # with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
        #     f.write(str(best_pred))
        # if self.runs:
        #     previous_miou = [0.0]
        #     for run in self.runs:
        #         run_id = run.split('_')[-1]
        #         path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
        #         if os.path.exists(path):
        #             with open(path, 'r') as f:
        #                 miou = float(f.readline())
        #                 previous_miou.append(miou)
        #         else:
        #             continue
        #     max_miou = max(previous_miou)
        #     if best_pred > max_miou:
        #         shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        # else:
        #     shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
