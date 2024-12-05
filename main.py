import time
import json
import pprint
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from accelerate import Accelerator

from utils.config import BaseOptions
from utils.basic_utils import set_seed, save_jsonl
from model import setup_model, setup_trainer
from dataset import build_dataset


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    train_dataset = build_dataset(opt, 'train')
    eval_dataset = build_dataset(opt, 'val')

    model = setup_model(opt)
    # accelerator = Accelerator()
    # train_loader, test_loader, model, optimizer = accelerator.prepare(
    #     train_loader, test_loader, model, optimizer
    # )
    
    trainer = setup_trainer(opt, model, train_dataset, eval_dataset)

    trainer.train()

if __name__ == '__main__':
    start_training()
