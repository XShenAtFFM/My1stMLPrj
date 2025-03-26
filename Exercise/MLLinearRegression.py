import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == '__main__':
    # load data
    data = pd.read_csv('USHousPrices.csv')
    data = data.drop(['Address'], axis=1)

    # plot the relationship between the prices and other factors like population, income
    pltidx = 321
    factors = data.columns.values
    factors = np.delete(factors, np.where(factors =='Price'))
    for i in factors:
    	figure = plt.subplot(pltidx)
    	plt.scatter(data[i],data['Price'])
    	plt.title(i + ' vs. Price')
    	pltidx += 1

    #create the input data array, drop colum "Avg. Area Number of Bedrooms". The prices range are
    #independent from the number of bed rooms. The 'Price' colum is driped too because it is y
    x = data.drop(['Avg. Area Number of Bedrooms','Price'],axis=1)
    y = data.loc[:,'Price']

    #calculation the parameters of the linear funciton 
    LR =LinearRegression().fit(x,y)

    #print(LR.coef_)
    #calculate the predicted price based on the x
    y_pred = LR.predict(x)
    
    #caclulation the error and r2
    Accuracy = mean_squared_error(y,y_pred)
    R2Score = r2_score(y,y_pred)
    print(f'The accuracy score {Accuracy:2.2f} and R2 score {R2Score:2.2f}')

    #calculation the parameters of the linear funciton with scaled data
    NormedX = minmax_scale(x,axis=0)
    #NormedY = minmax_scale(y,axis=0)
    NormedLR = LinearRegression().fit(NormedX,y)

    #calculate the predicted price based on the NormedX
    PredNormedY = NormedLR.predict(NormedX) #*(y.max()-y.min()) + y.min()
    print(y_pred-(PredNormedY)) #*(y.max()-y.min())+y.min()))

    #caclulation the error and r2 with normlized data
    Accuracy = mean_squared_error(y,PredNormedY)
    R2Score = r2_score(y,PredNormedY)
    print(f'The normlized accuracy score {Accuracy:2.2f} and R2 score {R2Score:2.2f}')

    #plot the predicted prices and real prices
    figure2 = plt.figure()
    plt.scatter(y,y_pred)
    plt.show()