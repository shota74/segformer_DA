from datasets_ import cityscapes,BDD,synscapes
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def make_data_loader(cfg, dataset=None, num_workers=None):

    if dataset == 'BDD':
        train_dataset = BDD.cityscapesSegDataset(
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
                        
        val_dataset = BDD.cityscapesSegDataset(
                        root_dir=cfg.dataset.root_dir,
                        name_list_dir=cfg.dataset.name_list_dir,
                        split=cfg.val.split,
                        stage='val',
                        aug=False,
                        ignore_index=cfg.dataset.ignore_index,
                        num_classes=cfg.dataset.num_classes,
                    )
        # if args.use_sbd:
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
        train_sampler = None
        num_class = 19
        #print(num_class)
        train_loader = DataLoader(train_dataset,
                                batch_size=cfg.train.samples_per_gpu,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                #sampler=train_sampler,
                                prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader,train_sampler, num_class

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

        train_sampler = None
        train_loader = DataLoader(train_dataset,
                                batch_size=cfg.train.samples_per_gpu,
                                #shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                sampler=train_sampler,
                                prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)
        
        test_loader = None
        num_class =19
    
        return train_loader, val_loader, test_loader, train_sampler, num_class
        

    elif dataset== 'cityscapes':
        num_class =19

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
            
        #train_sampler = DistributedSampler(train_dataset,shuffle=True)
        train_sampler = None
        train_loader = DataLoader(train_dataset,
                                batch_size=cfg.train.samples_per_gpu,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                #sampler=train_sampler,
                                prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)
        
        test_loader = None
    
        return train_loader, val_loader, test_loader, train_sampler, num_class

    else:
        raise NotImplementedError
