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
from tqdm import tqdm

from core.model import WeTr
from datasets import make_data_loader
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
                    default='GTA5_to_cityscapes',
                    type=str,
                    help="datset")
parser.add_argument("--dataset_sorce",
                    default='GTA5',
                    type=str,
                    help="datset")
parser.add_argument("--dataset_target",
                    default='cityscapes',
                    type=str,
                    help="datset")
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test_city.log'):
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

def validate(model=None, criterion=None, data_loader=None, cfg=None, class_type=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(
                torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    if class_type == 'SYNTHIA':
        synthia = True
    else:
        synthia = False
    score = eval_seg.scores(gts, preds,num_classes=cfg.dataset.num_classes, synthia=synthia)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(cfg):
    #scaler = torch.cuda.amp.GradScaler()
    num_workers = 8
    if args.local_rank==0:
        saver = Saver(args)
    print(args)
    world_size = 8
    #scaler = torch.cuda.amp.GradScaler()
    #saver.save_experiment_config()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    sorce_train_loader, _, _, _ = make_data_loader(cfg, args.dataset_sorce, num_workers)
    train_loader, val_loader, test_loader, train_sampler = make_data_loader(cfg, args.dataset_target, num_workers)

    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device  =torch.device(args.local_rank)
    wetr = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)
    param_groups = wetr.get_param_groups()
    
    wetr.to(device)
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
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
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)
    start_iter =0

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        start_iter = checkpoint['iter']
        wetr.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_sampler.set_epoch(0)
    sorce_train_loader_iter = iter(sorce_train_loader)
    train_loader_iter = iter(train_loader)

    best_IoU= 0

    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    for n_iter in range(start_iter, cfg.train.max_iters):
        if random.random() < 0.5:
            try:
                _, inputs, labels = next(train_loader_iter)
            except:
                train_sampler.set_epoch(n_iter)
                train_loader_iter = iter(train_loader)
                _, inputs, labels = next(train_loader_iter)
        else:
            try:
                _, inputs, labels = next(sorce_train_loader_iter)
            except:
                train_sampler.set_epoch(n_iter)
                sorce_train_loader_iter = iter(sorce_train_loader)
                _, inputs, labels = next(sorce_train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        #with torch.cuda.amp.autocast():
        optimizer.zero_grad()
        outputs = wetr(inputs)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, labels.type(torch.long))
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        loss = loss / world_size
        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # #scaler.step(optimizer_r)
        # # Updates the scale for next iteration.
        # scaler.update()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=wetr, criterion=criterion, data_loader=val_loader,cfg = cfg, class_type = args.dataset_sorce)
            is_best = False
            if args.local_rank==0:
                logging.info(val_score)
                if best_IoU < val_score["Mean IoU"]:
                    is_best = True
                    best_IoU = val_score["Mean IoU"]

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

    if args.local_rank == 0:
        setup_logger(filename='mix_city_' + args.dataset+ '.log')
        logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
