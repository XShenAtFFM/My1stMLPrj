from ImportStockData import ReadStockData, StockDailyData
from NeuronMultiDayPrice import NeuronMultiDayPrice
import numpy as np

def main():
    #create input list based on stock daily price info and volume
    readdata = []
    for daily, dailydetails in ReadStockData(jsonfile='applestock2024.json').items():
        readdata.insert(0,StockDailyData(daily,dailydetails))
    #create 1st level neuron daily stock  
    numberofdays = 5; # the data in each neuron is calculated on numberofdays days stock daliy info
    ListOfStockDailyData = readdata[0:((len(readdata)//numberofdays)*numberofdays)]
    #create 2nd level neuron numberofdays days stock info
    MultiDaysDate = []
    for i in range(0, len(ListOfStockDailyData), numberofdays):
        firstdate = ListOfStockDailyData[i].Date
        prices = [item.CloseP for item in ListOfStockDailyData[i:(i+numberofdays)]]
        meanprice = np.mean(prices)
        stddeviation = np.std(prices, mean=meanprice)
        MultiDaysDate.insert(0,NeuronMultiDayPrice(firstdate,meanprice,stddeviation))
    
    # print(close)
        # ListOfStockDailyData.insert(NeuronMultiDayPrice(ListOfStockDailyData[i].Date,\
        # ListOfStockDailyData[i:i+numberofdays-1]


if __name__ == "__main__": 
    main()