import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class My_Ref_ML_NN(L.LightningModule):
    """
    A sub cloass of LightningModule.
    """
    def __init__(self,  in_features, parameters_init):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 8)
        self.layer2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)

        self.loss_fn = nn.BCELoss()

        slef._init_weights(parameters_init)


    def _init_weights(self, parameters_init):
        self.layer1.weight.data.copy_(torch.from_numpy(parameters_init['weights_layer1']))
        self.layer1.bias.data.copy_(torch.from_numpy(parameters_init['bias_layer1']))

        self.layer2.weight.data.copy_(torch.from_numpy(parameters_init['weights_layer2']))
        self.layer2.bias.data.copy_(torch.from_numpy(parameters_init['bias_layer2']))

        self.output.weight.data.copy_(torch.from_numpy(parameters_init['weights_output']))
        self.output.bias.data.copy_(torch.from_numpy(parameters_init['bias_output']))

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
        return loss

    def configure_optimizers(self):
        """
        ref the parent method
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

def train_my_ref_nn(x,y, epochs):
    x_torch = torch.from_numpy(x)
    y_torch = torch.from_numpy(y)

    dataset = TensorDataset(x_torch, y_torch)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    ref_model = My_Ref_ML_NN(input_dim)
    trainer = L.Trainer(max_epochs = epochs)

    trainer.fit(ref_model, loader)


