import torch
import numpy as np

import argparse
import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # TODO
from .dataloader_total import MedicalDataloader

def create_data_loaders(args):
    # Data loading code
    # print("=> creating data loaders ...")

    if args.dataset == 'smalldata':
        root_path = r"/data/dataset/Medical/smalldata"

    elif args.dataset == 'blender':
        root_path = r"/data/dataset/Medical/blender-duodenum-5-211126"

    elif args.dataset == 'colon':
        root_path = r"/data/dataset/Medical/Colonscopy"


    train_dataset = MedicalDataloader(root_path, type='train', dataset=args.dataset)
    val_dataset = MedicalDataloader(root_path, type='val', dataset=args.dataset)

    # # set batch size to be 1 for validation
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # # put construction of train loader here, for those who are interested in testing only
    # train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.workers, pin_memory=True, sampler=None,
    #         worker_init_fn=lambda work_id: np.random.seed(work_id))
    #     # worker_init_fn ensures different sampling patterns for each data loading thread

    # print("=> data loaders created.")
    return train_dataset, val_dataset

