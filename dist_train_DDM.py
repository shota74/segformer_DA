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
from datasets_cutmix import make_data_loader
from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW
from utils.loss import ProbOhemCrossEntropy2d
from utils.saver import Saver
import sys

import mask_gen
from custom_collate import SegCollate

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/cityscapes.yaml',
                    type=str,
                    help="config")
parser.add_argument("--taerget_dataset",
                    default='synscapes',
                    type=str,
                    help="datset")
parser.add_argument("--sorce_dataset",
                    default='cityscapes',
                    type=str,
                    help="unsupervised_datset")
parser.add_argument("--mix_dataset",
                    default='synscapes',
                    type=str,
                    help="datset")

parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--resume_l', type=str, default=None,
                        help='put the path to resuming file if needed')
parser.add_argument('--resume_r', type=str, default=None,
                        help='put the path to resuming file if needed')
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_params(params_path):
   state_dict = torch.load(params_path, map_location=lambda storage, loc: storage)
   return state_dict

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

def slide_inference(img, model,stride=(512,512), crop_size = (768,768), num_classes = 19):
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
                crop_seg_logit = model(crop_img)
                #print(crop_seg_logit.shape)
                #print(crop_img.shape)
                crop_seg_logit = F.interpolate(crop_seg_logit,
                                            size=crop_img.shape[2:],
                                            mode='bilinear',
                                            align_corners=False)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        # if rescale:
        #     preds = resize(
        #         preds,
        #         size=img_meta[0]['ori_shape'][:2],
        #         mode='bilinear',
        #         align_corners=self.align_corners,
        #         warning=False)
        return preds

def validate(model=None, criterion=None, data_loader=None, cfg=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()
    device  =torch.device(args.local_rank)

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            sample = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)
            #inputs = inputs.to(device, non_blocking=True)
            inputs = sample["image"].to(device, non_blocking=True)
            labels = sample['label'] .to(device, non_blocking=True)
            outputs = slide_inference(inputs, model)

            #outputs = model(inputs)
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

    score = eval_seg.scores(gts, preds,num_classes=cfg.dataset.num_classes)

    return val_loss.cpu().numpy() / float(len(data_loader)), score


def Region_level_teacher(cfg):
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=(0.25, 0.5), n_boxes=3,
                                            random_aspect_ratio=not False,
                                            prop_by_area=not False, within_bounds=not False,
                                            invert=not False)

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    scaler = torch.cuda.amp.GradScaler()
    num_workers = 8
    if args.local_rank==0:
        saver = Saver(args)
    print(args)
    world_size = 8
    #saver.save_experiment_config()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    # 
    traeget_train_loader, target_val_loader, test_loader, train_sampler = make_data_loader(cfg, args.dataset, num_workers)

    sorce_train_loader, sorce_val_loader, _, sorce_sampler = make_data_loader(cfg, args.unsupervised_dataset, num_workers, mask_collate_fn)zw
    device  =torch.device(args.local_rank)zw
    teacher_1 = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)
    param_groups_1 = teacher_1.get_param_groups()    
    teacher_1.to(device)
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups_1[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_1[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_1[2],
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

    start_iter = 0
    best_IoU_= 0
    if args.resume_l is not None:
        if not os.path.isfile(args.resume_l):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume_l))
        checkpoint = load_params(args.resume_l)
        #start_iter = checkpoint['iter']
        #if args.cuda:
        teacher_1.load_state_dict(checkpoint['state_dict'])
        # optimizer_l.load_state_dict(checkpoint['optimizer'])
        # best_pred_l = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(args.resume_l, checkpoint['iter']))
        del checkpoint

    teacher_1= DistributedDataParallel(teacher_1, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)

    train_sampler.set_epoch(0)
    trarget_train_loader_iter = iter(traeget_train_loader)
    sorce_train_loader_iter = iter(sorce_train_loader)

    best_IoU_r = 0
    best_IoU_l = 0

    for n_iter in range(start_iter, cfg.train.max_iters):
        teacher_1.train()
        try:
            sample_trarget = next(trarget_train_loader_iter)
            sample_sorce = next(sorce_train_loader_iter)
            #sample_unsup_1 = next(unsupervised_loader_iter_2)
        except:
            train_sampler.set_epoch(n_iter)
            trarget_train_loader_iter = iter(traeget_train_loader)
            sorce_train_loader_iter = iter(sorce_train_loader)
            sample_trarget = next(trarget_train_loader_iter)
            sample_sorce = next(sorce_train_loader_iter)
             
        optimizer.zero_grad()
        inputs_traget = sample_trarget["image"].to(device, non_blocking=True)
        labels_traget = sample_trarget['label'] .to(device, non_blocking=True)
        inputs_sorce = sample_sorce["image"].to(device, non_blocking=True)
        labels_sorce = sample_sorce['label'] .to(device, non_blocking=True)

        batch_mix_masks = mask_params
        img_mixed = inputs_sorce * (1 - batch_mix_masks) + inputs_traget * batch_mix_masks
        label_mixed = labels_sorce * (1 - batch_mix_masks) + labels_traget * batch_mix_masks
        with torch.cuda.amp.autocast():
            outputs = teacher_1(img_mixed)
            outputs = F.interpolate(outputs, size=label_mixed.shape[1:], mode='bilinear', align_corners=False)
        
            loss = criterion(outputs, label_mixed.type(torch.long))
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss = loss / world_size

        scaler.scale(loss).backward()
        scaler.step(optimizer_l)
        scaler.step(optimizer_r)
        scaler.update()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer_r.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=teacher_1, criterion=criterion, data_loader=target_val_loader,cfg = cfg)
         
            if args.local_rank==0:
                logging.info(val_score)
                if best_IoU < val_score["Mean IoU"]:
                    is_best = True
                else:
                    is_best = False
    
                best_IoU = val_score["Mean IoU"]
                saver.save_checkpoint({
                    'iter': n_iter,
                    'state_dict': teacher_1.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_IoU,
                }, is_best)


def Sample_level_teacher(cfg):

    scaler = torch.cuda.amp.GradScaler()
    num_workers = 8
    if args.local_rank==0:
        saver = Saver(args)
    #print(args)
    world_size = 8
    #saver.save_experiment_config()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    train_loader, val_loader, test_loader, train_sampler = make_data_loader(cfg, args.mix_dataset, num_workers)

    device  =torch.device(args.local_rank)

    teacher_1 = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)

    param_groups_1 = teacher_1.get_param_groups()    
    teacher_1.to(device)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups_1[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_1[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_1[2],
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

    start_iter = 0
    best_IoU_= 0

    if args.resume_l is not None:
        if not os.path.isfile(args.resume_l):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume_l))
        checkpoint = load_params(args.resume_l)
        #start_iter = checkpoint['iter']
        #if args.cuda:
        teacher_1.load_state_dict(checkpoint['state_dict'])
        # optimizer_l.load_state_dict(checkpoint['optimizer'])
        # best_pred_l = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(args.resume_l, checkpoint['iter']))
        del checkpoint

    teacher_1= DistributedDataParallel(teacher_1, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)

    criterion = criterion.to(device)

    train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)
    #sorce_train_loader_iter = iter(sorce_train_loader)

    for n_iter in range(start_iter, cfg.train.max_iters):
        teacher_1.train()
        try:
            sample = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            sample = next(train_loader_iter)
            
        #sample["image"], sample['label'] 
        optimizer.zero_grad()
        inputs = sample["image"].to(device, non_blocking=True)
        labels = sample['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            #Cutmixラベル生成
            outputs = teacher_1(inputs)
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        
            loss = criterion(outputs, labels.type(torch.long))
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss = loss / world_size

        scaler.scale(loss).backward()
        scaler.step(optimizer_l)
        scaler.step(optimizer_r)
        scaler.update()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer_r.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=teacher_1, criterion=criterion, data_loader=target_val_loader,cfg = cfg)
         
            if args.local_rank==0:
                logging.info(val_score)
                if best_IoU < val_score["Mean IoU"]:
                    is_best = True
                else:
                    is_best = False
    
                best_IoU = val_score["Mean IoU"]
                saver.save_checkpoint({
                    'iter': n_iter,
                    'state_dict': teacher_1.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_IoU,
                }, is_best)

def student(cfg, teacher_1, teacher_2, round = 0):
    scaler = torch.cuda.amp.GradScaler()
    num_workers = 8
    if args.local_rank==0:
        saver = Saver(args)
    #print(args)
    world_size = 8
    #saver.save_experiment_config()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    train_loader, val_loader, test_loader, train_sampler = make_data_loader(cfg, args.taerget_dataset, num_workers)

    if round > 0 :
        psudo_train_loader, psudo_val_loader, _, pusdo_train_sampler = make_data_loader(cfg, args.psudo_dataset, num_workers)
    
    device  =torch.device(args.local_rank)

    student = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)

    param_groups = student.get_param_groups()    
    student.to(device)

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

    start_iter = 0
    best_IoU_= 0

    if args.resume_l is not None:
        if not os.path.isfile(args.resume_l):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume_l))
        checkpoint = load_params(args.resume_l)
        #start_iter = checkpoint['iter']
        #if args.cuda:
        teacher_1.load_state_dict(checkpoint['state_dict'])
        # optimizer_l.load_state_dict(checkpoint['optimizer'])
        # best_pred_l = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(args.resume_l, checkpoint['iter']))
        del checkpoint

    student= DistributedDataParallel(student, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)

    KL = torch.nn.KLDivLoss()
    KL = KL.to(device)

    train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)
    psudo_train_loader_iter = iter(psudo_train_loader)

    for n_iter in range(start_iter, cfg.train.max_iters):
        student.train()
        # 正解ラベル or 疑似ラベル
        if random.random() < 0.5 and round == 0:
            try:
                sample = next(train_loader_iter)
            except:
                train_sampler.set_epoch(n_iter)
                train_loader_iter = iter(train_loader)
                sample = next(train_loader_iter)
        else:
            try:
                sample = next(psudo_train_loader_iter)
            except:
                train_sampler.set_epoch(n_iter)
                psudo_train_loader_iter = iter(psudo_train_loader)
                sample = next(psudo_train_loader_iter)

            
        #sample["image"], sample['label']
        
        optimizer.zero_grad()
        inputs = sample["image"].to(device, non_blocking=True)
        labels = sample['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output_teacher_1 = teacher_1(inputs)
                output_teacher_2 = teacher_2(inputs)

            teacher_logit = (output_teacher_1 + output_teacher_2) / 2

            outputs = student(inputs)
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        
            CE_loss = criterion(outputs, labels.type(torch.long))
            dist.all_reduce(CE_loss, dist.ReduceOp.SUM)
            CE_loss = loss / world_size

            KL_loss = KL(outputs, teacher_logit)
            dist.all_reduce(KL_loss, dist.ReduceOp.SUM)
            KL_loss = KL_loss / world_size
        
        loss = 0.5 * KL_loss + CE_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=student, criterion=criterion, data_loader=target_val_loader,cfg = cfg)
         
            if args.local_rank==0:
                logging.info(val_score)
                if best_IoU < val_score["Mean IoU"]:
                    is_best = True
                else:
                    is_best = False
    
                best_IoU = val_score["Mean IoU"]
                saver.save_checkpoint({
                    'iter': n_iter,
                    'state_dict': student.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_IoU,
                }, is_best)


def train(cfg):
    
    for i in range(args.round):

        teacher_1 = Region_level_teacher(cfg)

        teacher_2 = Sample_level_teacher(cfg)

        student = student(cfg, teacher_1, teacher_2, round = i)
       


    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.local_rank == 0:
        setup_logger()
        logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
