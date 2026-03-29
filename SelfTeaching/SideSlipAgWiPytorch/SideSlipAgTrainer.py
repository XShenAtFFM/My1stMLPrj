import lightning as L
import torch
import sys
sys.path.insert(0, './')
from SideSlipAgWiPytorch.SideSlipAgAiMdl import SideSlipAgAiMdl


class SideSlipAgTrainer(L.LightningModule):
    def __init__(self,in_features):
        super().__init__()
        self.train =  L.Trainer(max_epochs = 100)



