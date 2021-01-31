import json
import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from models import MyCNN
from utils import device
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, MLFlowLogger
from system import MySystem
from logger import logging

if __name__ == '__main__':
    project = Project()
    cnn = MyCNN()
    cnn.load_state_dict('./checkpoint/1612018874.1357496/epoch=54-val_loss=0.09.ckpt')