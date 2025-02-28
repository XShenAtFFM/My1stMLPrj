import datetime

class NeuronMultiDayPrice:
    def __init__(self,firstdate, prices, std):
        self.BeginAtDate = firstdate
        self.MeanPrice = prices
        self.Std = std


  