import ResNetTransfer
import CifarDataModule
import lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
import time

if __name__ == "__main__":
    mdl_type = 'resnet18'
    # create datamodule
    cifar_data = CifarDataModule.CifarDataModule()
    # create nn model
    resnet_transfer = ResNetTransfer.ResNetTransfer(lr = 1e-4, model_type = mdl_type)

    # Checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath = "checkpoints",
        filename = f"transfer-{mdl_type}-tune-l4" + "{epoch:02d}-{val_acc:.4f}",
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
        max_epochs = 10,
        callbacks = [checkpoint_callback, early_stop_callback],
        logger = TensorBoardLogger("resnettransfer_log", name="ResnetTransfer")
    )

    transfer_trainer = L.Trainer(
        enable_progress_bar = False,
        log_every_n_steps = 20,
        **trainer_kwargs)


    checkpoint_dir = "./checkpoints"
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    #cifar_trainer.fit(cifar_nn, cifar_data, ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None )
    start_time = time.time()
    transfer_trainer.fit(resnet_transfer, cifar_data)
    print("--- %s seconds ---" % (time.time() - start_time))