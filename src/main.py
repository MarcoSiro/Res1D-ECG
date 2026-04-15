
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import wfdb          

from utilities import LightningModel, PTBXLDataModule, plot_test_metrics, plot_loss_curves, ResNet1D, plot_xai_examples

def main():
    input_channels = 12  #12 leads in the ECG signal
    num_classes = 5      #5 macro-categories of the PTB-XL dataset: NORM, MI, STTC, CD, HYP

    pytorch_model = ResNet1D(input_channels=input_channels, num_classes=num_classes) #Inizializing Pytorch model

    L.pytorch.seed_everything(123)

    dm = PTBXLDataModule(data_path="./data", batch_size=64, num_workers=4) #Inzializing DataModule

    lightning_model = LightningModel(
        model=pytorch_model, 
        learning_rate=1e-3, 
        num_classes=num_classes
    )

    #Configuration of Lightning Trainer
    trainer = L.Trainer(
        max_epochs=20,          #20 epoches for rapid MVP 
        accelerator="auto",     
        devices=1,
        logger=CSVLogger(save_dir="logs/", name="ecg-resnet1d"), 
        deterministic=True,     
    )

    print("Starting training...")       #Training the model
    trainer.fit(model=lightning_model, datamodule=dm)

    print("Starting final test...")     #Test on test dataset
    trainer.test(model=lightning_model, datamodule=dm)

    log_dir = trainer.logger.log_dir
    print(f"\nSaving graphics in the folder: {log_dir}")

    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    plot_loss_curves(log_dir) # Plotting training and validation loss curves
    
    plot_test_metrics(lightning_model, class_names, log_dir) #Reporting results on test set (AUC, F1 score, ROC curve)

    plot_xai_examples(lightning_model.model, dm, class_names, log_dir, num_examples=3)   # Plotting some XAI examples using Vanilla Saliency

if __name__ == '__main__':
    main()
