import torch
import os
import torchvision.transforms as T
from dataset.utils import BaseDataset , DomainAdaptationDataset ,DomainGeneralizationDataset
from dataset.utils import SeededDataLoader
from itertools import cycle

from globals import CONFIG

def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.Resize(256))
        transform.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.0)))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)


def load_data():
    CONFIG.num_classes = 7
    CONFIG.data_input_size = (3, 224, 224)

    # Create transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    train_transform = get_transform(size=224, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=224, mean=mean, std=std, preprocess=False)

    # Load examples & create Dataset
    if CONFIG.experiment in ['baseline']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = BaseDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)

    ######################################################
    #elif... TODO: Add here how to create the Dataset object for the other experiments
    elif CONFIG.experiment in ['ActivationShaping']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = BaseDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)
    #########################################################################################
    elif CONFIG.experiment in ['RandomActivationMap']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = BaseDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)
    ####################################################################################""""
    elif CONFIG.experiment in ['DomainAdaptation']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = DomainAdaptationDataset(source_examples, target_examples,transform=train_transform)
        test_dataset = DomainAdaptationDataset(source_examples, target_examples, transform=test_transform)
    ######################################################
    elif CONFIG.experiment in ['DomainGeneralization'] :
        domains = ['art_painting', 'photo', 'cartoon', 'sketch']
        domain_data = {domain: [] for domain in domains}

        # Load data for all domains
        for domain in domains:
            with open(os.path.join(CONFIG.dataset_args['root'], f"{domain}.txt"), 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                path, label = line[0].split('/')[1:], int(line[1])
                domain_data[domain].append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Determine source and target domains
        target_domain = CONFIG.dataset_args['target_domain']
        source_domains = [domain for domain in domains if domain != target_domain]

        # Prepare combined source examples
        from itertools import cycle, islice

        # Create iterators for each source domain
        iterators = [cycle(domain_data[domain]) for domain in source_domains]

        # Determine the number of examples (use the length of the largest domain)
        num_examples = max(len(domain_data[domain]) for domain in source_domains)

        # Combine examples from each source domain into tuples
        source_examples = [(next(iterators[0]), next(iterators[1]), next(iterators[2])) for _ in range(num_examples)]
        # Create Domain Generalization datasets
        train_dataset = DomainGeneralizationDataset(source_examples, transform=train_transform)
        test_dataset = DomainGeneralizationDataset(domain_data[target_domain], transform=test_transform)
        
            # Dataloaders
    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}