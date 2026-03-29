import pandas as pd
from asammdf import MDF
import torch
import sys
import os
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

sys.path.insert(0, '../')
from pathlib import Path
from SideSlipAgWiPytorch.SideSlipAgAiMdl import SideSlipAgAiMdl
from SideSlipAgWiPytorch.DataModule import DataModule
import argparse

def get_args():
    # initialize parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ResumCheckPoint", help="Give the name of the check point")
    parser.add_argument("-a", "--action", choices=["prediction", "training"], required=True,
                        help="Choice action 'training', 'prediction'")
    return parser.parse_args()


def tmpplot(data):
    # type of data is list, each item in list is tuple data
    # with 2 element, x_hat and y
    for y_hat, y in data:
        figure4 = plt.figure()
        # remove the batch dimension
        y_hat=y_hat.squeeze(0)
        y = y.squeeze(0)
        # subpot velocity x
        plt.subplot(2, 1, 1)
        plt.plot(y_hat[:,0], linestyle = 'None', marker = 'o', markersize=2, label='estimated Vx')
        plt.plot(y[:, 0], linestyle = 'None', marker = '.', markersize=2, label='measured Vx')
        plt.ylabel('longitudinal velocity Vx')
        # subpot velocity y
        plt.subplot(2, 1, 2)
        plt.plot(y_hat[:,1],linestyle = 'None', marker = 'o', markersize=2, label='estimated Vy')
        plt.plot(y[:, 1],linestyle = 'None', marker = '.', markersize=2, label='measured Vy')
        plt.ylabel('lateral velocity Vy')
        plt.legend()
        plt.show()

def main():
    # Get argument from command line
    args = get_args()

    # Load data sets
    train_data_path = './DataSets/Corrsys'
    file_List = [p.as_posix() for p in Path(train_data_path).iterdir() if p.suffix == '.mf4']
    vehicle_measurements = DataModule(file_List, is_standardized = True if args.ResumCheckPoint else False )

    # Instance an SideSlipAgAiMdl
    velocity_estimator_mdl = SideSlipAgAiMdl(len(vehicle_measurements._dataset._features))

    # Check accelerator
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="velocity_estimator-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,  # keep best 3 models
        mode="min",
        save_last=True,
        monitor="val_loss"
    )

    ## Define trainer
    # Set trainer arguments
    trainer_kwargs = dict(
        accelerator = accelerator,
        devices = 1,
        max_epochs = 2000,
        callbacks = [checkpoint_callback],
    )

    velocity_estimator_trainer = Trainer(**trainer_kwargs)

    # start action
    if args.action == "prediction" and args.ResumCheckPoint is not None:
        predict_results = velocity_estimator_trainer.predict(velocity_estimator_mdl, vehicle_measurements, ckpt_path=args.ResumCheckPoint)
        results = []
        for result in predict_results:
            y_hat, y, file_idx = result
            results.append({
                "filename": vehicle_measurements._dataset.filename[file_idx],
                "dataframe": pd.DataFrame(torch.cat([y_hat, y], dim=1).numpy(), columns=["Vx_hat", "Vy_hat", "Vx_measurd", "Vy_measured" ])
            })

    elif args.action == "training":
        velocity_estimator_trainer.fit(velocity_estimator_mdl, vehicle_measurements, ckpt_path=args.ResumCheckPoint if args.ResumCheckPoint else None)
    else:
        raise Exception("Invalid action")

if __name__ == '__main__':
    main()