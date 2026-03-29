import lightning as L
import torch
from pycparser.c_ast import Default
from torchmetrics import Accuracy
from torchvision.models import resnet18, resnet50, ResNet50_Weights, ResNet18_Weights


class ResNetTransfer(L.LightningModule):
    def __init__(self, num_classes = 10, lr = 1e-3, model_type = 'resnet18', **kwargs):
        super().__init__()
        # It will enable Lightning to store all the provided arguments under the self.hparams attribute.
        # These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        self.save_hyperparameters()

        self.train_acc = Accuracy(task = "multiclass", num_classes = num_classes)
        self.val_acc = Accuracy(task = "multiclass", num_classes = num_classes)
        self.test_acc = Accuracy(task = "multiclass", num_classes = num_classes)
        if model_type == 'resnet50':
            self._model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        elif model_type == 'resnet18':
            self._model = resnet18(weights = ResNet18_Weights.DEFAULT)
        else:
            raise NotImplementedError
        # The following line add an additional dense layer at the end of the resnet18/50
        # the output of the resnet18/50 has dimension as fc.in_features
        # the output of the dense layer is num_classes
        self._model.fc = torch.nn.Linear(self._model.fc.in_features, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Resnet structure
        # layer1 → low - level features(edges, textures)
        # layer2 → mid - level patterns
        # layer3 → more abstract
        # layer4 → high - level semantic features
        # fc     → classifier

        # Freeze whole pretrained Resnet
        for param in self._model.parameters():
            param.requires_grad = False

        # Unfreeze the parameter of last add dense layer
        for param in self._model.fc.parameters():
            param.requires_grad = True

        # Unfreeze the 2nd last resnet layer
        for param in self._model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.train_acc(x_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.val_acc(x_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        self.test_acc(x_hat, y)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)

        y_hat = torch.argmax(x_hat, dim=1)
        return {'pred': y_hat}

    def configure_optimizers(self):
        """
        ref the parent method
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr
        )

        return optimizer

