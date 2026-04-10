
import torch
import torch.utils.data.distributed
from .get_medDataloader import create_data_loaders

import random

class GetMedDataloader():
    def __init__(self, args, mode):

        if mode == 'train':
            train_data = None

            train_data, _ = create_data_loaders(args)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            else:
                train_sampler = None
            self.data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                    shuffle=(train_sampler is None), num_workers=args.num_threads,
                                                    pin_memory=True, sampler=train_sampler,
                                                    drop_last=True)

        elif mode == 'online_eval':
            val_data = None

            _, val_data = create_data_loaders(args)
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            self.data = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.num_threads, pin_memory=True,
                                                    sampler=val_sampler)

        elif mode == 'arch_search':
            train_data, _ = create_data_loaders(args)
            dataset_size = len(train_data)
            indices = list(range(dataset_size))
            split = int(dataset_size * args.train_data_ratio)

            if args.seed is not None:
                random.seed(args.seed)
            random.shuffle(indices)

            train_indices, arch_indices = indices[:split], indices[split:]
            train_subset = torch.utils.data.Subset(train_data, train_indices)
            arch_subset = torch.utils.data.Subset(train_data, arch_indices)

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset)
                arch_sampler = torch.utils.data.distributed.DistributedSampler(arch_subset)
            else:
                train_sampler = None
                arch_sampler = None
            
            self.train_data = torch.utils.data.DataLoader(train_subset, 
                                                          batch_size=args.batch_size, 
                                                          shuffle=(train_sampler is None), num_workers=args.num_threads, pin_memory=True, sampler=train_sampler, drop_last=True)
            self.arch_data = torch.utils.data.DataLoader(arch_subset, 
                                                          batch_size=args.batch_size, 
                                                          shuffle=(arch_sampler is None), num_workers=args.num_threads, pin_memory=True, sampler=arch_sampler, drop_last=True)