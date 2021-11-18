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
parser.add_argument("--dataset",
                    default='synscapes',
                    type=str,
                    help="datset")
parser.add_argument("--unsupervised_dataset",
                    default='cityscapes',
                    type=str,
                    help="unsupervised_datset")
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

def train(cfg):
    
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
    
    train_loader, _, test_loader, train_sampler = make_data_loader(cfg, args.dataset, num_workers)

    unsupervised_loader, val_loader, _, unsupervised_sampler = make_data_loader(cfg, args.unsupervised_dataset, num_workers)
    #unsupervised_loader_2, val_loader_2, _, unsupervised_sampler_2 = make_data_loader(cfg, args.unsupervised_dataset, num_workers)


    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device  =torch.device(args.local_rank)

    wetr_l = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)

    wetr_r = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)



    param_groups_l = wetr_l.get_param_groups()    
    wetr_l.to(device)

    param_groups_r = wetr_r.get_param_groups()    
    wetr_r.to(device)
    
    
    optimizer_r = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups_r[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_r[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_r[2],
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

    optimizer_l = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups_l[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_l[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups_l[2],
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
    best_IoU_r = 0
    best_IoU_l = 0

    if args.resume_l is not None:
        if not os.path.isfile(args.resume_l):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume_l))
        checkpoint = load_params(args.resume_l)
        #start_iter = checkpoint['iter']
        #if args.cuda:
        wetr_l.load_state_dict(checkpoint['state_dict'])
        # optimizer_l.load_state_dict(checkpoint['optimizer'])
        # best_pred_l = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(args.resume_l, checkpoint['iter']))
        del checkpoint
    
    if args.resume_r is not None:
        if not os.path.isfile(args.resume_r):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume_r))
        checkpoint = load_params(args.resume_r)
        #start_iter = checkpoint['iter']+1
        #if args.cuda:
        wetr_r.load_state_dict(checkpoint['state_dict'])
        # optimizer_r.load_state_dict(checkpoint['optimizer'])
        # best_pred_r = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(args.resume_r, checkpoint['iter']))
        del checkpoint

    
    #wetr, optimizer = amp.initialize(wetr, optimizer, opt_level="O1")
    wetr_r = DistributedDataParallel(wetr_r, device_ids=[args.local_rank], find_unused_parameters=True)
    wetr_l = DistributedDataParallel(wetr_l, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion
    pixel_num = 50000 * cfg.train.samples_per_gpu
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    # ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
    #                                    min_kept=pixel_num, use_weight=False)
    criterion = criterion.to(device)


    criterion_cps = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion_cps = criterion_cps.to(device)

    train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)
    unsupervised_loader_iter = iter(unsupervised_loader)
    #unsupervised_loader_iter_2 = iter(unsupervised_loader_2)
    best_IoU_r = 0
    best_IoU_l = 0

    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    for n_iter in range(start_iter, cfg.train.max_iters):
        wetr_l.train()
        wetr_r.train()
        try:
            sample_sup = next(train_loader_iter)
            sample_unsup_0 = next(unsupervised_loader_iter)
            #sample_unsup_1 = next(unsupervised_loader_iter_2)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            sample_sup = next(train_loader_iter)
            unsupervised_loader_iter = iter(unsupervised_loader)
            sample_unsup_0 = next(unsupervised_loader_iter)
            # unsupervised_loader_iter_2 = iter(unsupervised_loader_2)
            # sample_unsup_1 = next(unsupervised_loader_iter_2)
            
        #sample["image"], sample['label'] 
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()
        inputs = sample_sup["image"].to(device, non_blocking=True)
        labels = sample_sup['label'] .to(device, non_blocking=True)
        uns_inputs_0 = sample_unsup_0["image"].to(device, non_blocking=True)
        #uns_inputs_1 = sample_unsup_1["image"].to(device, non_blocking=True)
        #mask_params = sample_unsup_0['mask_params'].to(device, non_blocking=True)
        #print(mask_params)

        #batch_mix_masks = mask_params
        #unsup_imgs_mixed = uns_inputs_0 * (1 - batch_mix_masks) + uns_inputs_1 * batch_mix_masks
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                #Cutmixラベル生成
                pred_unsup_l_0 = wetr_r(uns_inputs_0)
                #pred_unsup_l_1 = wetr_r(uns_inputs_1)
                pred_unsup_l_0 = F.interpolate(pred_unsup_l_0, size=labels.shape[1:], mode='bilinear', align_corners=False).detach()
                #pred_unsup_l_1 = F.interpolate(pred_unsup_l_1, size=labels.shape[1:], mode='bilinear', align_corners=False).detach()
                

                pred_unsup_r_0 = wetr_l(uns_inputs_0)
                #pred_unsup_r_1 = wetr_l(uns_inputs_1)
                pred_unsup_r_0 = F.interpolate(pred_unsup_r_0, size=labels.shape[1:], mode='bilinear', align_corners=False).detach() 
                #pred_unsup_r_1 = F.interpolate(pred_unsup_r_1, size=labels.shape[1:], mode='bilinear', align_corners=False).detach()

            
            #pred_unsup_l = pred_unsup_l_0 * (1 - batch_mix_masks) + pred_unsup_l_1 * batch_mix_masks
            _, ps_label_l = torch.max(pred_unsup_l_0, dim=1)
            ps_label_l = ps_label_l.long()
            #pred_unsup_r = pred_unsup_r_0 * (1 - batch_mix_masks) + pred_unsup_r_1 * batch_mix_masks
            _, ps_label_r = torch.max(pred_unsup_r_0, dim=1)
            ps_label_r = ps_label_r.long()

            outputs_r = wetr_r(inputs)
            outputs_l = wetr_l(inputs)
            outputs_l = F.interpolate(outputs_l, size=labels.shape[1:], mode='bilinear', align_corners=False)
            outputs_r = F.interpolate(outputs_r, size=labels.shape[1:], mode='bilinear', align_corners=False)

            pred_unsup_l = wetr_r(uns_inputs_0)
            pred_unsup_r = wetr_l(uns_inputs_0)
            pred_unsup_l = F.interpolate(pred_unsup_l, size=labels.shape[1:], mode='bilinear', align_corners=False)
            pred_unsup_r = F.interpolate(pred_unsup_r, size=labels.shape[1:], mode='bilinear', align_corners=False)

            # pred_l = torch.cat([outputs_l, pred_unsup_l], dim=0)
            # pred_r = torch.cat([outputs_r, pred_unsup_r], dim=0)

            # _, max_l = torch.max(pred_l, dim=1)
            # _, max_r = torch.max(pred_r, dim=1)
            # max_l = max_l.long()
            # max_r = max_r.long()

            cps_loss = criterion_cps(pred_unsup_l, ps_label_r) + criterion_cps(pred_unsup_r, ps_label_l)
            dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            cps_loss = cps_loss / world_size
            cps_loss = cps_loss * 6
        
            loss_sup = criterion(outputs_l, labels.type(torch.long))
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / world_size

            loss_sup_r = criterion(outputs_r, labels.type(torch.long))
            dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / world_size

            #seg_loss = criterion(outputs, labels.type(torch.long))
            #sys.exit()
            loss = loss_sup + loss_sup_r + cps_loss
            #print(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer_l)
        scaler.step(optimizer_r)
        # Updates the scale for next iteration.
        scaler.update()

        # loss.backward()
        # optimizer_l.step()
        # optimizer_r.step()

        # optimizer.zero_grad()
        # seg_loss.backward()
        # optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer_r.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=wetr_l, criterion=criterion, data_loader=val_loader,cfg = cfg)
            val_loss, val_score_r = validate(model=wetr_r, criterion=criterion, data_loader=val_loader,cfg = cfg)
            if args.local_rank==0:
                logging.info(val_score)
                
                if best_IoU_l < val_score["Mean IoU"]:
                    is_best_l = True
                else:
                    is_best_l = False
                
                logging.info(val_score_r)
                if best_IoU_r < val_score_r["Mean IoU"]:
                    is_best_r = True
                else:
                    is_best_r = False

                best_IoU_l = val_score["Mean IoU"]
                saver.save_checkpoint({
                    'iter': n_iter,
                    'state_dict': wetr_l.module.state_dict(),
                    'optimizer': optimizer_l.state_dict(),
                    'best_pred': best_IoU_l,
                }, is_best_l)

                best_IoU_r = val_score_r["Mean IoU"]
                saver.save_checkpoint_r({
                    'iter': n_iter,
                    'state_dict': wetr_r.module.state_dict(),
                    'optimizer': optimizer_r.state_dict(),
                    'best_pred': best_IoU_r,
                }, is_best_r)

                
                


    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.local_rank == 0:
        setup_logger()
        logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
