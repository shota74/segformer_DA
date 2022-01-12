import argparse
import datetime
import logging
import os
import random
import cv2
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
                    default='configs/cityscapes16.yaml',
                    type=str,
                    help="config")
parser.add_argument("--dataset",
                    default='cityscapes_16',
                    type=str,
                    help="datset")
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
parser.add_argument('--result', type=str, default='./result/',
                    help='set the checkpoint name')
parser.add_argument('--name', type=str, default='./result/',
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

def cityscapes():
    label = np.array([
        [128,64,128],
        [232,35,244],
        [70,70,70],
        [156,102,102],
        [153,153,190],
        [153,153,153],
        [30,170,250],
        [0,220,220],
        [35,142,107],
        #[152,251,152],
        [180,130,70],
        [60,20,220],
        [0,0,255],
        [142,0,0],
        #[70,0,0],
        [100,60,0],
        #[100,80,0],
        [230,0,0],
        [32,11,119]], dtype=np.uint8)
    return label

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

def validate(model=None, criterion=None, data_loader=None, cfg=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()
    class_color = cityscapes()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            label_ = labels
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)
            #print(resized_outputs.shape)
            

            loss = criterion(resized_outputs, labels)
            val_loss += loss
            resu = label_.cpu().numpy()
            #resu = np.argmax(resu, axis=1)
            resu = np.transpose(resu, axes=[1, 2, 0])

            height, width, color = resu.shape # 幅・高さ・色を取得
            dst = np.zeros((height, width, 3), dtype = "uint8") # 合成画像用の変数を作成
            for ii in range(0, class_color.shape[0]):
                    dst[np.where((resu == [ii, ii, ii]).all(axis=2))] = class_color[ii]
            #print(name[0])
            _img_path2 =os.path.basename(name[0])
            cv2.imwrite(args.result + _img_path2+'.png', dst)

            preds += list(torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    #score13 = eval_seg.scores(gts, preds,num_classes=cfg.dataset.num_classes, synthia=True)
    score = eval_seg.scores(gts, preds,num_classes=cfg.dataset.num_classes, synthia=False)

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
    
    _, val_loader, test_loader, train_sampler = make_data_loader(cfg, args.dataset, num_workers)

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

    best_IoU= 0
    val_loss, val_score = validate(model=wetr, criterion=criterion, data_loader=val_loader,cfg = cfg)
    #print(val_score)
    if args.local_rank == 0:
        #setup_logger(filename='result_' + args.dataset+ '.log')
        logging.info('\nresult: %s' % val_score)
        #logging.info('\nresult_16: %s' % val_score16)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    if args.local_rank == 0:
        setup_logger(filename='result_' + args.name+ '.log')
        logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
