import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from core.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from core.sync_batchnorm.replicate import patch_replication_callback
from core.sync_batchnorm import convert_model, DataParallelWithCallback
from tqdm import tqdm

from core.model import WeTr_bn2d
from datasets_ import make_data_loader
from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW
from utils.saver import Saver
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/cityscapes.yaml',
                    type=str,
                    help="config")
parser.add_argument("--dataset",
                    default='cityscapes',
                    type=str,
                    help="datset")
parser.add_argument('--gpu-ids', type=str, default='0,1',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

parser.add_argument('--datasets_list', nargs='+',
                    help='datasets list for training', required=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)



def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def slide_inference(img,cls_ind, model, rescale,stride=(512,512), crop_size = (768,768), num_classes = 19):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                #crop_seg_logit = self.encode_decode(crop_img, img_meta)
                crop_seg_logit = model(crop_img,cls_ind)
                crop_seg_logit = F.interpolate(crop_seg_logit,
                                            size=crop_img.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

def validate(model=None, criterion=None, data_loader=None, cfg=None, data_id=None,num_classes=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        #for cls_ind in range(len(data_loader)):
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)
            output = slide_inference(inputs,
                                     data_id,
                                     model,
                                     rescale=False,
                                     crop_size=(cfg.dataset.crop_size,cfg.dataset.crop_size),
                                     num_classes=num_classes[data_id])

            #outputs = model(inputs, data_id)
            labels = labels.long().to(outputs.device)

            # resized_outputs = F.interpolate(outputs,
            #                                 size=labels.shape[1:],
            #                                 mode='bilinear',
            #                                 align_corners=False)

            loss = criterion(output, labels)
            val_loss += loss

            preds += list(
                torch.argmax(output,
                            dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds,num_classes=cfg.dataset.num_classes)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(cfg):

    num_workers = 16
    # if args.local_rank==0:
    #     saver = Saver(args)
    #     print(args)
    #saver.save_experiment_config()

    #torch.cuda.set_device(args.local_rank)
    #dist.init_process_group(backend=args.backend,)
    dataloader_list_train=[]
    dataloader_list_val=[]
    dataloader_list_nclass=[]
    num_classes = []

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

    for i in range(len(args.datasets_list)):
        dataset = args.datasets_list[i]
    
        train_loader, val_loader, test_loader, _, num_classe = make_data_loader(cfg, args.dataset, num_workers)
        train_size += len(train_loader)
        dataloader_list_train.append(train_loader)
        dataloader_list_val.append(val_loader)
        dataloader_list_nclass.append(nclass)
        num_classes.append(num_classe)



    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device = torch.device('cuda')
    #device  =torch.device(args.local_rank)
    wetr = WeTr_bn2d(backbone=cfg.exp.backbone,
                num_classes=num_classes
                embedding_dim=768,
                pretrained=True)
    param_groups = wetr.get_param_groups()
    
   
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    #wetr, optimizer = amp.initialize(wetr, optimizer, opt_level="O1")
    #wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    # wetr = torch.nn.DataParallel(wetr, device_ids=args.gpu_ids)
    # patch_replication_callback(wetr)
    # wetr.to(device)
    wetr = convert_model(wetr).cuda() # Batch NormをSync Batch Normに変換
    wetr = DataParallelWithCallback(wetr, device_ids=args.gpu_ids)

    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)

    #train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)
    data_iter_list = []
    data_iter_list_val = []
    for i in range(len(dataloader_list_train)):
        data_iter_list.append(iter(dataloader_list_train[i]))
        data_iter_list_val.append(iter(dataloader_list_val[i]))

    best_IoU= 0

    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    for n_iter in range(cfg.train.max_iters):
        losss = 0
        for cls_ind in range(len(dataloader_list_train)):
        
            try:
                _, inputs, labels = next(data_iter_list[cls_ind])
            except:
                #train_sampler.set_epoch(n_iter)
                data_iter_list[cls_ind] =  iter(dataloader_list_train[cls_ind])
                _, inputs, labels = next(data_iter_list[cls_ind])

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = wetr(inputs, cls_ind)
            #sys.exit()
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            #sys.exit()
            if losss == 0:
                seg_loss = criterion(outputs, labels.type(torch.long))
            else:
                seg_loss += criterion(outputs, labels.type(torch.long))
            #sys.exit()


        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()
        
        #if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
        if (n_iter+1) % cfg.train.log_iters == 0: 
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, seg_loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            #if args.local_rank==0:
            logging.info('Validating...')
            sum_iou= 0
            for data_id in range(len(data_iter_list_val)):
                val_loss, val_score = validate(model=wetr, criterion=criterion, data_loader=data_iter_list_val[data_id],cfg = cfg, data_id = data_id, num_classes =num_classes)
                #if args.local_rank==0:
                logging.info(val_score)
                sum_iou += val_score["Mean IoU"]
            
            if best_IoU < sum_iou:
                is_best = True
                best_IoU = sum_iou
                saver.save_checkpoint({
                    'iter': n_iter+1,
                    'state_dict': wetr.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_IoU,
                }, is_best)


    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    #if args.local_rank == 0:
    setup_logger()
    logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
