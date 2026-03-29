import torch
from tensorflow.python.platform.tf_logging import log_every_n

from  MyOwnTransformer import MyTransformer
from TinyShakespeareDataModule import TinyShakespeareDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import lightning as L
import os

if __name__ == "__main__":
    # hyper param
    seq_length = 64
    heads = 4
    d_model = 128
    batch_size = 64
    layers = 4

    # create lightning data module instance
    tiny_shakespeare_datamodule = TinyShakespeareDataModule(seq_length, batch_size)
    vocab_size = tiny_shakespeare_datamodule.vocab_size

    # create my transformer instance
    tiny_shakespeare_transformer = MyTransformer(vocab_size, d_model, seq_length * 2, seq_length, layers, heads)

    # Checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath = "checkpoints",
        filename = "tinyshakespeare-{epoch:02d}-{val_loss:.4f}",
        save_top_k = 3,  # keep best 3 models
        mode = "min",
        save_last = True,
        monitor = "val_loss"
    )

    # early stop call back
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.002,
        patience=30,
        mode="min"
    )

    # Check accelerator
    # accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

    ## Define trainer
    # Set trainer arguments
    trainer_kwargs = dict(
        accelerator = "auto", # easy implementation ref
        devices = 1,
        max_epochs = 300,
        callbacks = [checkpoint_callback, early_stop_callback],
        logger = TensorBoardLogger("tinyshakespeare_log", name="TinyShakespeare")
    )

    transformer_trainer = L.Trainer(
        enable_progress_bar = False,
        log_every_n_steps = 20,
        gradient_clip_val = 1.0, # best practice for transformers, always clip gradients to 1.0 or similar small values
        **trainer_kwargs)


    checkpoint_dir = "./checkpoints"
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    # keyword "last" lightning resolves last check point internally. It is better to avoid corrupted last check point
    transformer_trainer.fit(tiny_shakespeare_transformer, tiny_shakespeare_datamodule, ckpt_path = "last" if os.path.exists(last_ckpt_path) else None )