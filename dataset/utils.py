import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG

class BaseDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y = self.examples[index]
        x = Image.open(x).convert('RGB')
        x = self.transform(x).to(CONFIG.dtype)
        y = torch.tensor(y).long()
        return x, y

######################################################
# TODO: modify 'BaseDataset' for the Domain Adaptation setting.
# Hint: randomly sample 'target_examples' to obtain targ_x
class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.transform = transform
    
    def __len__(self):
        return len(self.source_examples)
    
    def __getitem__(self, index):
        # Unpack source domain example
        src_x_path, src_y = self.source_examples[index]
        src_x = Image.open(src_x_path).convert('RGB')
        src_x = self.transform(src_x)
        src_y = torch.tensor(src_y).long()

        # Unpack target domain example
        # Assuming each element in self.target_examples is a tuple (image_path, label)
        targ_x_path, _ = self.target_examples[index % len(self.target_examples)]
        targ_x = Image.open(targ_x_path).convert('RGB')
        targ_x = self.transform(targ_x)

        return src_x, src_y, targ_x
    
# [OPTIONAL] TODO: modify 'BaseDataset' for the Domain Generalization setting. 
# Hint: combine the examples from the 3 source domains into a single 'examples' list
class DomainGeneralizationDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        domain_data = self.examples[index]

        # Ensure that domain_data contains three tuples (one for each domain)
        if len(domain_data) != 3:
            raise ValueError(f"Expected 3 tuples per item, got {len(domain_data)}")

        results = []
        for img_path, label in domain_data:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            label = torch.tensor(label).long()
            results.extend([img, label])  # This should add 2 items per domain

        if len(results) != 6:
            raise ValueError(f"Expected 6 items (3 images and 3 labels), got {len(results)}")

        return tuple(results)
    
    
class SeededDataLoader(DataLoader):
    def init(self, dataset: Dataset, batch_size=1, shuffle=None, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, collate_fn=None, 
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, 
                 generator=None, *, prefetch_factor=None, persistent_workers=False, 
                 pin_memory_device=""):

        if not CONFIG.use_nondeterministic:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = seed_worker

        super().init(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)

