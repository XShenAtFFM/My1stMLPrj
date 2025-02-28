import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LR

def main():
    #load stock data from json files
    with open('applestock2024.json','r') as f:
        dataraw = json.load(f)
    #remove the meta data
    data = dataraw["Time Series (Daily)"]
    #create pandas dataframe
    ReadDataRaw = pd.DataFrame(data)
    #print(ReadDataRaw)
    #transform the dataframe
    StockData = pd.DataFrame(ReadDataRaw.T)
    #print(StockData)

    #reverse the  frame older date first
    StockData = StockData.iloc[::-1]
    StockData = StockData.reset_index() #otherwise the date is used as index    
    StockData['4. close'] = StockData['4. close'].astype(float)

    #Cal 5MA
    StockData['MA5'] = StockData['4. close'].rolling(window=5).mean()    
    #plt.plot(StockData.index,StockData['MA5'] )
    #plt.show()
    
    #Cal Average
    StockData['5DaysIdx'] = StockData.index // 5

    print(StockData.head())
    FiveDaysAv = StockData.groupby('5DaysIdx')['4. close'].mean().reset_index()
    FiveDaysAv = FiveDaysAv.rename(columns={'4. close': '5DaysAv'})
    print(FiveDaysAv)


if __name__ == "__main__": 
    main()