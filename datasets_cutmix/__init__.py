from datasets_cutmix import cityscapes, synscapes
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def make_data_loader(cfg, dataset=None, num_workers=None, collate_fn=None):

    if dataset == 'BDD':
        train_set = bdd.BDDSegmentation(args, split='train')
        val_set = bdd.BDDSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        #print(num_class)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    
    elif dataset== 'synscapes':

        train_dataset = synscapes.cityscapesSegDataset(
                        root_dir="/workspace/dataset/Synscapes/img_trainval/",
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.train.split,
                        stage='train',
                        aug=True,
                        resize_range=cfg.dataset.resize_range,
                        rescale_range=cfg.dataset.rescale_range,
                        crop_size=cfg.dataset.crop_size,
                        img_fliplr=True,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                        )
    
        val_dataset = synscapes.cityscapesSegDataset(
                        root_dir="/workspace/dataset/Synscapes/img_trainval/",
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.val.split,
                        stage='val',
                        aug=False,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                    )

        #test_dataset = None
            
        train_sampler = DistributedSampler(train_dataset,shuffle=True)
        #train_sampler = None
        train_loader = DataLoader(train_dataset,
                                batch_size=cfg.train.samples_per_gpu,
                                #shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                sampler=train_sampler,
                                prefetch_factor=4,
                                collate_fn=collate_fn)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)
        
        test_loader = None
    
        return train_loader, val_loader, test_loader, train_sampler


    elif dataset== 'cityscapes':

        train_dataset = cityscapes.cityscapesSegDataset(
                        root_dir=cfg.dataset.root_dir,
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.train.split,
                        stage='train',
                        aug=True,
                        resize_range=cfg.dataset.resize_range,
                        rescale_range=cfg.dataset.rescale_range,
                        crop_size=cfg.dataset.crop_size,
                        img_fliplr=True,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                        )
    
        val_dataset = cityscapes.cityscapesSegDataset(
                        root_dir=cfg.dataset.root_dir,
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.val.split,
                        stage='val',
                        aug=False,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                    )

        test_dataset = cityscapes.cityscapesSegDataset(
                        root_dir=cfg.dataset.root_dir,
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.val.split,
                        stage='val',
                        aug=False,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                    )
            
        train_sampler = DistributedSampler(train_dataset,shuffle=True)
        #train_sampler = None
        train_loader = DataLoader(train_dataset,
                                batch_size=cfg.train.samples_per_gpu,
                                #shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                sampler=train_sampler,
                                prefetch_factor=4,
                                collate_fn=collate_fn)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)
        
        test_loader = None
    
        return train_loader, val_loader, test_loader, train_sampler

    else:
        raise NotImplementedError
