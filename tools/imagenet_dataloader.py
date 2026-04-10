import torch
import torch.utils.data.distributed
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import PIL
import cv2
from PIL import Image

def custom_collate_fn(batch):
    images, labels = zip(*batch)  # 分解成 images 和 labels
    return {'image': torch.stack(images, 0), 'label': torch.tensor(labels)}

class GetImageNetDataloader():
    def __init__(self, args, mode):
        if mode == 'train':
            assert os.path.exists(args.data_path)
            train_data = datasets.ImageFolder(
                args.data_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(0.5),
                    ToBGRTensor(),
                ])
            )
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            else:
                train_sampler = None

            self.data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                    shuffle=(train_sampler is None), num_workers=args.num_threads, pin_memory=True,
                                                    sampler=train_sampler,
                                                    drop_last=True, collate_fn=custom_collate_fn  # 使用自定义的 collate_fn
                                                    )

        elif mode == 'online_eval':
            assert os.path.exists(args.data_path_eval)
            val_data = datasets.ImageFolder(args.data_path_eval, transforms.Compose([
                OpencvResize(256),
                transforms.CenterCrop(224),
                ToBGRTensor(),
            ]))
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            self.data = torch.utils.data.DataLoader(val_data, batch_size=200,
                                                    shuffle=False, num_workers=args.num_threads,
                                                    pin_memory=True, sampler=val_sampler, collate_fn=custom_collate_fn)  # 使用自定义的 collate_fn
            
        elif mode == 'arch_search':
            assert os.path.exists(args.data_path)
            train_data = datasets.ImageFolder(
                args.data_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(0.5),
                    ToBGRTensor(),
                ])
            )

            # from torch.utils.data import Subset  # TODO
            # subset_indices = list(range(128))
            # train_data = Subset(train_data,subset_indices)

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            else:
                train_sampler = None

            self.train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                    shuffle=(train_sampler is None), num_workers=args.num_threads, pin_memory=True,
                                                    sampler=train_sampler,
                                                    drop_last=True, collate_fn=custom_collate_fn)
            self.arch_data = self.train_data


class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img
    
class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img