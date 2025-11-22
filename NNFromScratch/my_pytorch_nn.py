import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import Callback

class My_Loss_Recorder(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics["train_loss"].item()
        self.epoch_losses.append(loss)


class My_Ref_ML_NN(L.LightningModule):
    """
    A sub cloass of LightningModule.
    """
    def __init__(self,  in_features, parameters_init, optimizer_method):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 8)
        self.layer2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
        self.loss_fn = nn.BCELoss()
        self._init_weights(parameters_init)
        self.optimizer_method = optimizer_method


    def _init_weights(self, parameters_init):
        self.layer1.weight.data.copy_(torch.from_numpy(parameters_init['weights_layer1']))
        self.layer1.bias.data.copy_(torch.from_numpy(parameters_init['bias_layer1'].squeeze()))

        self.layer2.weight.data.copy_(torch.from_numpy(parameters_init['weights_layer2']))
        self.layer2.bias.data.copy_(torch.from_numpy(parameters_init['bias_layer2'].squeeze()))

        self.output.weight.data.copy_(torch.from_numpy(parameters_init['weights_output']))
        self.output.bias.data.copy_(torch.from_numpy(parameters_init['bias_output'].squeeze()))

    def forward(self, x):
        """
        Feed data to the nn
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def training_step(self, batch, batch_idx):
        """
        training step: ref the parent method
        """
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        """
        ref the parent method
        """
        if self.optimizer_method == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08)
        elif self.optimizer_method == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        else:
            raise NotImplementedError
        return optimizer

def train_my_ref_nn(x,y, epochs, init_parameters, optimizer_method = 'sgd'):
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()

    dataset = TensorDataset(x_torch, y_torch)
    loader = DataLoader(dataset, batch_size = 100, shuffle = False)

    ref_model = My_Ref_ML_NN(x.shape[1], init_parameters, optimizer_method)
    loss_recorder = My_Loss_Recorder()

    trainer = L.Trainer(max_epochs = epochs,
                        log_every_n_steps = 1,
                        callbacks = [loss_recorder]
                        )

    trainer.fit(ref_model, loader)
    return ref_model, trainer, loss_recorder



