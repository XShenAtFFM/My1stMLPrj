import lightning as L
import torch
from torchmetrics import Accuracy
from torchvision.models import resnet18


class CifarNNML(L.LightningModule):
    def __init__(self, enable_batchnorm = True, enable_residual = True, std_filter_size = 3,
                 std_pool_size = 2, optimize = 'adam', model_type = 'myown', **kwargs):
        super(CifarNNML, self).__init__()
        self.std_filter_size = 3
        self.std_pool_size = 2
        self.enable_batchnorm = enable_batchnorm
        self.enable_residual = enable_residual
        self.optimize = optimize
        self.model_type = model_type

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.model_type == 'myown':
            self._my_nn_model()
        else:
            self._resnet()

    def _resnet(self):
        self._resnet18 = resnet18(num_classes=10)
        self._resnet18.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self._resnet18.maxpool = torch.nn.Identity()

    def _my_nn_model(self):
        # block_1_0 2 times convolution with batch normalization, at the end no relu because of switchable residual connection
        self.block_1_0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, self.std_filter_size, 1, padding = 'same', bias = True),
            torch.nn.BatchNorm2d(32, affine = True) if self.enable_batchnorm else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, self.std_filter_size, 1, padding = 'same', bias = True),
            torch.nn.BatchNorm2d(32, affine = True) if self.enable_batchnorm else torch.nn.Identity())

        # make the input and output of block_1_0 same shape, required for residual connection
        self.block_1_1 = torch.nn.Conv2d(3, 32, kernel_size=1, stride = 1, bias = False)

        # block_1_2, the construction might look strange, because switchable residual connection between block 1_0 and 1_2
        self.block_1_2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(self.std_pool_size, stride = 2))

        # block_2_0 2 times convolution with batch normalization, at the end no relu because of switchable residual connection
        self.block_2_0 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, self.std_filter_size, 1, padding = 'same', bias = True),
            torch.nn.BatchNorm2d(64, affine = True) if self.enable_batchnorm else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, self.std_filter_size, 1, padding = 'same', bias = True),
            torch.nn.BatchNorm2d(64, affine = True) if self.enable_batchnorm else torch.nn.Identity())

        # make the output of the blcck 1_2 and output of block_2_0 same shape
        self.block_2_1 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False)

        self.block_2_2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(self.std_pool_size, stride = 2))

        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(4096, 256, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))

        self.output = torch.nn.Linear(256, 10, bias = True)

    def forward(self, x):
        if self.model_type == 'myown':
            return self._my_nn_forward(x)
        else:
            return self._resnet_forward(x)

    def _my_nn_forward(self, x):
        x1 = self.block_1_0(x)
        if self.enable_residual:
            x1 = x1 + self.block_1_1(x)
        x1 = self.block_1_2(x1)
        x2 = self.block_2_0(x1)
        if self.enable_residual:
            x2 = x2 + self.block_2_1(x1)
        x2 = self.block_2_2(x2)
        x2 = torch.flatten(x2, 1)
        x3 = self.fully_connected(x2)
        output = self.output(x3)
        return output

    def _resnet_forward(self, x):
        return self._resnet18(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.train_acc(x_hat, y)

        self.log("train_loss", loss )
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        loss = self.loss_fn(x_hat, y)
        self.val_acc(x_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        self.test_acc(x_hat, y)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        y_hat = torch.argmax(x_hat, dim=1)
        return {'pred': y_hat}

    def configure_optimizers(self):
        """
        ref the parent method
        """
        if self.model_type == 'resnet':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=3e-4, weight_decay=1e-4
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = 300
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
