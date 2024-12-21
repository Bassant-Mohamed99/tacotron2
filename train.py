import os
import time
import argparse
import math
from numpy import finfo

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


def prepare_dataloaders(hparams):
    # Get data, data loaders, and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    # Load the model on CPU
    model = Tacotron2(hparams).to(torch.device("cpu"))
    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, collate_fn, logger, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print(f"Validation loss {iteration}: {val_loss:9f}")
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, hparams):
    """Training and validation logging results to tensorboard and stdout"""
    torch.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(output_directory, log_directory, 0)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            duration = time.perf_counter() - start
            print(f"Train loss {iteration} {loss.item():.6f} Grad Norm {grad_norm:.6f} {duration:.2f}s/it")
            logger.log_training(loss.item(), grad_norm, learning_rate, duration, iteration)

            if iteration % hparams.iters_per_checkpoint == 0:
                validate(model, criterion, valset, iteration, hparams.batch_size, collate_fn, logger, 0)
                checkpoint_path = os.path.join(output_directory, f"checkpoint_{iteration}")
                save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true', help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    print("Running on CPU")
    train(args.output_directory, args.log_directory, args.checkpoint_path, args.warm_start, hparams)
