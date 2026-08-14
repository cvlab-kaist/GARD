import os 
import glob
import torch 
import random
import numpy as np
from typing import  Optional
from torch.utils.data import DistributedSampler, Sampler


class PhoBatchSampler(Sampler):
    def __init__(
        self,
        sampler,
        batch_size,
        max_num_input_view,
        epoch=0,
        seed=42,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_num_input_view = max_num_input_view
        self.rng = random.Random()
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        sampler_iter = iter(self.sampler)

        while True:
            try:
                # sample num_input_view ONCE per batch
                num_input_view = self.rng.randint(1, self.max_num_input_view)
                # print('sampler', num_input_view)
                batch = []
                for _ in range(self.batch_size):
                    idx = next(sampler_iter)
                    batch.append((idx, num_input_view))
                yield batch

            except StopIteration:
                break

    def __len__(self):
        return len(self.sampler) // self.batch_size



# class PhoSampler(DistributedSampler):
#     """
#     Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
#     parameters, which can be passed into the dataset's __getitem__ method.
#     """
#     def __init__(self,
#         dataset,
#         num_replicas: Optional[int] = None,
#         rank: Optional[int] = None,
#         shuffle: bool = False,
#         seed: int = 0,
#         drop_last: bool = False,
#     ):
#         super().__init__(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=seed,
#             drop_last=drop_last
#         )

#         # self.image_num = None

#     def __iter__(self):
#         """
#         Yields a sequence of (index, image_num, aspect_ratio).
#         Relies on the parent class's logic for shuffling/distributing
#         the indices across replicas, then attaches extra parameters.
#         """
#         indices_iter = super().__iter__()

#         for idx in indices_iter:
#             # yield (idx, self.image_num,)
#             yield idx 

        
#     def update_parameters(self, image_num):
#         """
#         Updates dynamic parameters for each new epoch or iteration.

#         Args:
#             aspect_ratio: The aspect ratio to set.
#             image_num: The number of images to set.
#         """
#         self.image_num = image_num



class PhoSampler(DistributedSampler):

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

        self.dataset = dataset
        self.num_datasets = len(dataset.datasets)

        # compute per-dataset index ranges
        self.dataset_ranges = []
        start = 0
        for ds in dataset.datasets:
            end = start + len(ds)
            self.dataset_ranges.append((start, end))
            start = end

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # sample equal amount from each dataset
        per_dataset_indices = []

        min_size = min(end - start for start, end in self.dataset_ranges)

        for start, end in self.dataset_ranges:
            indices = torch.arange(start, end)

            if self.shuffle:
                indices = indices[torch.randperm(len(indices), generator=g)]

            # take only min_size to balance
            indices = indices[:min_size]

            per_dataset_indices.append(indices)

        # concatenate balanced indices
        indices = torch.cat(per_dataset_indices)

        # shuffle combined indices
        if self.shuffle:
            indices = indices[torch.randperm(len(indices), generator=g)]

        # split for DDP
        total_size = len(indices)
        indices = indices[self.rank:total_size:self.num_replicas]

        return iter(indices.tolist())

    def __len__(self):
        min_size = min(end - start for start, end in self.dataset_ranges)
        return (min_size * self.num_datasets) // self.num_replicas