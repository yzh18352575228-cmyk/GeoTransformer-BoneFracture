import argparse
import os.path as osp
import time

import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)

        # Load pretrained weights if specified
        if cfg.pretrained is not None:
            self._load_pretrained(model, cfg.pretrained)

        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def _load_pretrained(self, model, pretrained_path):
        self.logger.info('Loading pretrained weights from {}'.format(pretrained_path))
        state_dict = torch.load(pretrained_path)
        model_dict = state_dict['model'] if 'model' in state_dict else state_dict

        # Check compatibility
        model_keys = set(model.state_dict().keys())
        pretrained_keys = set(model_dict.keys())
        missing = model_keys - pretrained_keys
        unexpected = pretrained_keys - model_keys

        if missing:
            self.logger.info('  Missing keys (will be random): {}'.format(len(missing)))
        if unexpected:
            self.logger.info('  Unexpected keys (ignored): {}'.format(len(unexpected)))

        model.load_state_dict(model_dict, strict=False)
        self.logger.info('  Loaded {}/{} layers'.format(
            len(pretrained_keys - unexpected), len(model_keys)))

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
