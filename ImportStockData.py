import json
import numpy as np
from datetime import datetime

def ReadStockData(jsonfile='applestock2024.json'):
    with open(jsonfile,'r') as f:
        data = json.load(f)
    datalist = data["Time Series (Daily)"]
    return datalist

    
class StockDailyData:
        def __init__(self, date,detailsinfo):
            self.Date = datetime.strptime(date,'%Y-%m-%d').date()
            self.OpenP = float(detailsinfo["1. open"])
            self.DailyH = float(detailsinfo["2. high"])
            self.DailyL = float(detailsinfo["3. low"])          
            self.CloseP = float(detailsinfo["4. close"])
            self.Volume = float(detailsinfo["5. volume"])
        
        def __str__(self):
            return f"{self.Date}: {self.OpenP}; {self.DailyH}; {self.DailyL}; {self.CloseP}; {self.Volume}"

if __name__ == "__main__": 
    ListOfStockDailyData = []
    for daily, dailydetails in ReadStockData().items():
        ListOfStockDailyData.append(StockDailyData(daily,dailydetails))
    
    ListOfStockDailyData = reversed(ListOfStockDailyData)    
    for items in ListOfStockDailyData:
        print(items)
    