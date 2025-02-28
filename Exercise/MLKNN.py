import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	data = pd.read_csv('data.csv')
	Label = data['labels']

	KNN = KNeighborsClassifier(n_neighbors=3).fit(data.loc[:,['V1','V2']],data['labels'])
	PredLabel = KNN.predict(data.loc[:,['V1','V2']])
	Accuracy = accuracy_score(data['labels'],PredLabel)

	
	figure0 = plt.figure()
	label0 = plt.scatter(data['V1'][Label==0],data['V2'][Label==0], marker = 4, s=10)
	label1 = plt.scatter(data['V1'][Label==1],data['V2'][Label==1], marker = 4, s=10)
	label2 = plt.scatter(data['V1'][Label==2],data['V2'][Label==2], marker = 4, s=10)
	Plabel0 = plt.scatter(data['V1'][PredLabel==0],data['V2'][PredLabel==0], marker = 5, s=10)
	Plabel1 = plt.scatter(data['V1'][PredLabel==1],data['V2'][PredLabel==1], marker = 5, s=10)
	Plabel2 = plt.scatter(data['V1'][PredLabel==2],data['V2'][PredLabel==2], marker = 5, s=10)
	plt.title(f'clustering estimator with KMeans accuracy score ({(Accuracy*100):2.2f}%)')	
	plt.show()
	