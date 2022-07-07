from matplotlib.pyplot import title
import torch
import wandb
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from loss import loss_func


def train_one_epoch(config, model, epoch, dataloader, optimizer, scheduler, device='cpu', batch_size=1):
    model.train()

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    loss_sum = 0
    epoch_loss = 0

    for step, (img, label) in bar:
        """  
        Args :
            img (torch.Tensor): (B, C, H, W)
            label (list) : list of dictionary
        """

        img = img.to(device)
        pred = model(img)        # (batch, 7, 7, 30)

        optimizer.zero_grad()

        pred = pred.to('cpu')
        loss = loss_func(config=config, pred=pred, label_list=label)

        loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item()

        epoch_loss = loss_sum / (step + 1)

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]["lr"])

    return epoch_loss


def val_one_epoch(config, model, epoch, dataloader, device='cpu'):
    model.eval()

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    loss_sum: int = 0
    epoch_loss: int = 0

    for step, (img, label) in bar:
        img = img.to(device)

        pred = model(img)

        pred = pred.to('cpu')

        loss = loss_func(config=config, pred=pred, label_list=label)

        loss_sum += loss.item()

        epoch_loss = loss_sum / (step + 1)

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    return epoch_loss


def run_training(config, model, train_dataloader, val_dataloader, optimizer, scheduler):
    Epoch = config.MODEL.EPOCH
    FINETUNE_EPOCH = config.MODEL.FINETUNE_EPOCH

    # To log gradient
    if (config.WANDB.USE):
        wandb.watch(model)

    best_loss = np.inf
    history = defaultdict(list)

    # --------------------------------------------------------
    # FINE TUNING (Backbone is freezed)
    # --------------------------------------------------------
    for epoch in range(0, FINETUNE_EPOCH):
        train_loss = train_one_epoch(config=config, model=model, epoch=epoch, dataloader=train_dataloader,
                                     optimizer=optimizer, scheduler=scheduler, device=config.TRAINING.DEVICE, batch_size=config.MODEL.BATCH_SIZE)

        val_loss = val_one_epoch(
            config=config, model=model, epoch=epoch, dataloader=val_dataloader,)

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)

        if (config.WANDB.USE):
            wandb.log(
                {
                    'Train Loss': train_loss,
                    'Valid Loss': val_loss,
                }
            )

        if val_loss < best_loss:
            best_loss = val_loss
            file_prefix = 'YOLOV1'
            save_path = "{}epoch{:.0f}_Loss{:.4f}.bin".format(
                file_prefix, epoch, best_loss)
            torch.save(model.state_dict(), save_path)
            if (config.WANDB.USE):
                wandb.save(save_path)

   # --------------------------------------------------------
   # UNFREEZE ALL LAYERS
   # --------------------------------------------------------

    for name, param in model.named_parameters():
        param.requires_grad = True

   # --------------------------------------------------------
   # FULL MODEL TRAINING
   # --------------------------------------------------------
    for epoch in range(FINETUNE_EPOCH, Epoch):
        train_loss = train_one_epoch(config=config, model=model, epoch=epoch, dataloader=train_dataloader,
                                     optimizer=optimizer, scheduler=scheduler, device=config.TRAINING.DEVICE, batch_size=config.MODEL.BATCH_SIZE)

        val_loss = val_one_epoch(
            config=config, model=model, epoch=epoch, dataloader=val_dataloader,)

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)

        if (config.WANDB.USE):
            wandb.log(
                {
                    'Train Loss': train_loss,
                    'Valid Loss': val_loss,
                }
            )

        if val_loss < best_loss:
            best_loss = val_loss
            file_prefix = 'YOLOV1'
            model_name = "{}epoch{:.0f}_Loss{:.4f}.bin".format(
                file_prefix, epoch, best_loss)
            save_path = Path(config.TRAINING.SAVE_PATH) / Path(model_name)
            torch.save(model.state_dict(), str(save_path))
            if (config.WANDB.USE):
                wandb.save(save_path)

    print("Best Loss: {:.4f}".format(best_loss))
