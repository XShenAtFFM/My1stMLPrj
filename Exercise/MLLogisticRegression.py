import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def CalBoundary(x):
    #g(x,y) = a * y^2 + b * y + c = 0 ref lin 32 for function g
    # cal y based on x
    np.seterr(invalid='ignore') # ignore sqrt( < 0)
    a= LR.coef_[0][3]
    b= LR.coef_[0][1] +  LR.coef_[0][4]*x
    c = LR.intercept_ + LR.coef_[0][0]*x + LR.coef_[0][2]*(x**2)
    y1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    y2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return y1,y2


if __name__ == '__main__':
    # load data
    data = pd.read_csv('chip_test.csv')
    # check the load data
    # print(data.head())

    # plot pass failed data
    mask = data.loc[:,'pass']==1
    #passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
    #failed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][~mask])
    #plt.show()

    ############## Logistical Regrssion ################
    # Logistical Regrssion
    # g(x1,x2) = intercept_ + coef_[0][0] * test1 + coef_[0][1] * test2
    #            + coef[0][2] * test1^2 + coef[0][3] * test2^2 +
    #            + coef[0][4] * test1 * test2 
    XNew = pd.DataFrame({'x1':data['test1'],'x2':data['test2'], \
                        'X1_X1':data['test1'].pow(2), \
                        'X2_X2':data['test2'].pow(2), \
                        'X1_X2':data['test1']*data['test2'] \
                        })
    #print(XNew.head())
    LR = LogisticRegression()
    LR.fit(XNew,data.loc[:,'pass'])

    # Cal Accuracy
    # predicating based on the logical regression
    PredictResults = LR.predict(XNew)
    Accuracy = accuracy_score(data['pass'],PredictResults)
    #print(Accuracy)

    #coef[0][3] * test2^2  + coef_[0][1] * test2 + coef[0][4] * test1 * test2  +
    #      ntercept_ + coef_[0][0] * test1 + coef[0][2]*test1^2  = 0 
    xaxisstart = data['test1'].min()
    xaxisend = data['test1'].max()
    xaxis_range = [xaxisstart + (xaxisend-xaxisstart)/10000 * x for x in range (0,10000)]
    xaxis_range = np.array(xaxis_range)

    # cal new boundary 
    boundary1 = []
    boundary2 = []
    for x in xaxis_range:
        y1,y2 = CalBoundary(x)
        boundary1.append(y1)
        boundary2.append(y2)
    boundary1 = np.array(boundary1)
    boundary2 = np.array(boundary2)


    #plot results
    passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
    failed = plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
    lowerbd = plt.plot(xaxis_range,boundary1)
    upperbd = plt.plot(xaxis_range,boundary2)
    plt.title(f'Logistic Regression (accuray: {(Accuracy*100):2.2f})')
    plt.show()
    
