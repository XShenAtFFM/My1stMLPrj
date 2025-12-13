import CifarNNMdl
import CifarDataModule
import lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

if __name__ == "__main__":

    # create datamodule
    cifar_data = CifarDataModule.CifarDataModule()
    # create nn model
    cifar_nn = CifarNNMdl.CifarNNML()

    # Checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath = "checkpoints",
        filename = "cifar-{epoch:02d}-{val_acc:.4f}",
        save_top_k = 3,  # keep best 3 models
        mode = "max",
        save_last = True,
        monitor = "val_acc"
    )
    # early stop call back
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0.002,
        patience=30,
        mode="max"
    )
    # Check accelerator
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

    ## Define trainer
    # Set trainer arguments
    trainer_kwargs = dict(
        accelerator = accelerator,
        devices = 1,
        max_epochs = 300,
        callbacks = [checkpoint_callback, early_stop_callback],
        logger = TensorBoardLogger("cifar_log", name="CifarNNMdl")
    )

    cifar_trainer = L.Trainer(
        enable_progress_bar = False,
        log_every_n_steps = 20,
        **trainer_kwargs)


    checkpoint_dir = "./checkpoints"
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    cifar_trainer.fit(cifar_nn, cifar_data, ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None )