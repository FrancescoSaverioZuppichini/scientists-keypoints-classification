import json
from logging import log
import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from models import MyCNN, MyLinear
from utils import device
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from system import MySystem
from logger import logging

if __name__ == '__main__':
    project = Project()

    with open('./secrets.json', 'r') as f:
        secrets = json.load(f)

    seed_everything(0)
    # our hyperparameters
    params = {
        'lr': 1e-3,
        'batch_size': 64,
        'epochs': 50,
        'model': 'my-linear-[32-128]-drop-only-last',
        'id': time.time(),
        'shuffle': True,
        'seq_len': 96
    }
    logging.info(f'Using device={device} 🚀')
    # everything starts with the data
    train_dl, val_dl, test_dl = get_dataloaders(
        project.data_dir,
        val_transform=val_transform,
        train_transform=train_transform,
        batch_size=params['batch_size'],
        pin_memory=True,
        num_workers=4,
        seq_len=params['seq_len']
    )
    model = MyCNN().to(device)
    # print the model summary to show useful information
    if params['seq_len'] != 1:
        logging.info(summary(model, (36, params['seq_len'])))
    else:
        logging.warning("torchsummary doesn't work with 1 dim size")
    # # define and create the model's chekpoints dir
    model_checkpoint_dir = project.checkpoint_dir / str(params['id'])
    model_checkpoint_dir.mkdir(exist_ok=True)
    # our callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=model_checkpoint_dir,
            filename='best'
        ),
        EarlyStopping(monitor='val_loss', patience=10, verbose=True)
    ]
    # using commet
    logger = CometLogger(
        api_key=secrets['COMET_API_KEY'],
        project_name="scientists-keypoints",
        workspace="francescosaveriozuppichini"
    )
    logger.log_hyperparams(params)

    system = MySystem(model=model, lr=params['lr'])

    trainer = Trainer(gpus=1,
                      min_epochs=params['epochs'],
                      max_epochs=params['epochs'],
                      progress_bar_refresh_rate=20,
                      logger=logger,
                      callbacks=callbacks)

    trainer.fit(system, train_dl, val_dl)

    # print(trainer.test(test_dataloaders=test_dl))
