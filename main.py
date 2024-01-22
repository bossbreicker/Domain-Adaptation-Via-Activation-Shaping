import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm
from itertools import cycle

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18 , ModifiedResNet , ASHResNet18

from globals import CONFIG

@torch.no_grad()
def evaluate(model, data):
    model.eval()

    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]

    for batch in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            if CONFIG.experiment in ['DomainAdaptation']:
                x, y, _ = batch  # Ignore the third element for DomainAdaptation
            else:
                x, y = batch

            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)

    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
          
            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                
                if CONFIG.experiment in ['baseline']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                ######################################################
                #elif... TODO: Add here train logic for the other experiments
                elif CONFIG.experiment in  ['ActivationShaping'] :
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                elif CONFIG.experiment in ['RandomActivationMap'] :
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                elif CONFIG.experiment in ['DomainAdaptation']:
                    # Extract source and target data
                    src_x, src_y, targ_x = batch
                    src_x, src_y, targ_x = src_x, src_y, targ_x
                    targ_x = targ_x.to(dtype=torch.half)
                    targ_x = targ_x.to(dtype=model.base_model.conv1.weight.dtype)
                    # Forward target through the network to record activation maps Mt
                    Mt = model.record_activation_maps(targ_x)
                    # Apply Mt using custom activation shaping layers when forwarding xs
                    zs = model.forward_with_activation_shaping(src_x, Mt)
                    model = model.to('cuda:0')
                    zs = zs.to('cuda:0')
                    src_y = src_y.to('cuda:0')
                    loss = F.cross_entropy(zs, src_y)
                    
                elif CONFIG.experiment in ['DomainGeneralization']:
                    # Unpack the batch (three inputs from different domains)
                    x1, y1, x2, y2, x3, y3 = batch
                    x1, y1 = x1.to(CONFIG.device), y1.to(CONFIG.device)
                    x2, y2 = x2.to(CONFIG.device), y2.to(CONFIG.device)
                    x3, y3 = x3.to(CONFIG.device), y3.to(CONFIG.device)

                    # Forward each input through the network
                    # Note: No activation shaping is applied in this case, just a regular forward pass
                    logits1 = model(x1)
                    logits2 = model(x2)
                    logits3 = model(x3)

                    # Compute loss for each input and take the average
                    loss1 = F.cross_entropy(logits1, y1)
                    loss2 = F.cross_entropy(logits2, y2)
                    loss3 = F.cross_entropy(logits3, y3)
                    loss = (loss1 + loss2 + loss3) / 3

                ######################################################

            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()

            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()

    ######################################################
    #elif... TODO: Add here model loading for the other experiments (eg. DA and optionally DG)
    elif CONFIG.experiment in ['ActivationShaping'] :
        model = ModifiedResNet()

    elif CONFIG.experiment in ['RandomActivationMap'] :
        model = ModifiedResNet()
    
    elif CONFIG.experiment in ['DomainAdaptation'] :
        model = ASHResNet18(num_classes=7)

    elif CONFIG.experiment in ['DomainGeneralization'] :
        model = ASHResNet18(num_classes=7)
    ######################################################
    
    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
