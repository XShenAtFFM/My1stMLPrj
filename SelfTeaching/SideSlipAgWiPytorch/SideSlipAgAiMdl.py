import lightning as L
import torch

class SideSlipAgAiMdl(L.LightningModule):
    """
    A sub cloass of LightningModule. It built an autoencoder model with 3 linear model + Relu
    2 LSTM + 3 linear model + Relu.

    The number of the layers are compromise between computing power and feasible solutions.

    Due to fewer LSTM layer no residual step is considered

    :attributes:
        encoder: encoder model nn -> nn -> -> nn -> ReLu
        lstm1: 1st LSTM layer
        lstm2: 2nd LSTM layer
        decoder: decoder model, similar as encoder
        output: output, a linear neuronal net, size = 2
    """
    def __init__(self,  in_features):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(in_features, 256),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(256, 256),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(256, 256),
                                     torch.nn.ReLU())
        self.lstm1 = torch.nn.LSTM(256, 256//2, bidirectional = True, batch_first = True)
        self.lstm2 = torch.nn.LSTM(256, 256//2, bidirectional = True, batch_first = True)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                    torch.nn.ReLU(),
                                     torch.nn.Linear(256, 256),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(256, 256),
                                     torch.nn.ReLU())
        # Model output, [longitudinal velocity, lateral velocity, log_variance of the 1st, log_variance of the 2nd ]
        self.output = torch.nn.Linear(256, 4)

    def forward(self, x):
        """
        Feed data to the ai model
        :param x: sequence data
        :return: output of the ai model
        """
        out_encoder = self.encoder(x)
        # padded data temporary
        padded_length = 500
        data = list(torch.split(out_encoder.squeeze(0), padded_length, dim=0))
        lengths = [len(x) for x in data]
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        data = torch.nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False
        )
        out_lstm1, _ = self.lstm1(data)
        out_lstm2, _ = self.lstm2(out_lstm1)
        data, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_lstm2, batch_first=True)
        data = torch.nn.utils.rnn.unpad_sequence(data, lengths, batch_first=True)
        data = torch.concat(data, dim=0)

        # Residual net inactive, because only 2 lstm layer are implemented.
        #data = self.decoder(data + out_encoder)

        out_layer = self.output(data)
        return out_layer

    def training_step(self, batch, batch_idx):
        """
        training step: ref the parent method
        """
        x, y,_ = batch
        x_hat = self(x)
        # Add the batch dimension
        x_hat = x_hat.unsqueeze(0)
        # Heteroscedastic Gaussian NLL
        estimated_velocity = x_hat[..., :2]
        log_var = x_hat[..., 2:].clamp(-10, 10)  # keep stable
        res = y - estimated_velocity
        loss = 0.5 * (res ** 2) * torch.exp(-log_var) + 0.5 * log_var
        loss = loss.mean()

        #loss = torch.nn.functional.mse_loss(x_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: ref the parent method.
        The validation method is the same as training step..
        """
        x, y,_ = batch
        x_hat = self(x)
        x_hat = x_hat.unsqueeze(0)
        # Heteroscedastic Gaussian NLL
        estimated_velocity = x_hat[..., :2]
        log_var = x_hat[..., 2:].clamp(-10, 10)  # keep stable
        res = y - estimated_velocity
        loss = 0.5 * (res ** 2) * torch.exp(-log_var) + 0.5 * log_var
        loss = loss.mean()
        # loss = torch.nn.functional.mse_loss(x_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        ref the parent method
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def predict_step(self, batch, batch_idx):
        """
        ref the parent method
        :return:
        """
        x, y, idx_datasets = batch
        x_hat = self(x)
        y_hat = x_hat.unsqueeze(0)
        return y_hat[...,:2], y, idx_datasets



