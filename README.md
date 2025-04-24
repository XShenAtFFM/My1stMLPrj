# My First Machine Learning Project
- Brief Introduction
	- This repor is created to share my learning way of machine learning

- MLIrisDataSet
	- This jupyter notebook uses the iris dataset for learning and practising typical ML algorithms of sample clustering and cataloguing
		- 1. Decision Tree with entropy measure
		- 2. As a, but the data are down sized with PCA 
			- Standarize data (mean = 0, std = 1)
			- Calculate the explained covariance by calling PCA with orginal number of features
			- Create a new data frame with reduced data dimension            - 
		- 3. KMeans (unsupervised learning) to clustering the data
		- 4. KNN(K-Nearest-Neighbors) 
		- 5. MeanShift (unsupervised learning)
		- 6. Build a MLP(Multi-Layer Perceptron) with 2 Dense layers
    - The method Decision Tree, KNN indicate the best results. The performance of the Decision Tree on the PCA processed data and MLP is little bit lower than the method a, d. The method c and e are unsupervised learning. In this way the performance of these are pretty poor.

- AnomalyDetection
	- 5 different methods for anomlay detection are implemented in this jupyter notebook.
		1) Calcuate the mean and standard deviation value, if some data are far away from the mean value
		2) Cluster data. Assume the 99% data are normal. Pick the 1% data which are at farthest away from the cluster central are otlier
			- Use KMeans and ellow method to find out the suitable number of culsters,
			- Use PCA(Principal component analysis) to reduce the data amount by calculating covariance, eigenvalue (eigenvactor is also caulcated but not used)
			- Use KMeans to cluster the dimension reduced dataset and calcuation the distance beteeen the data point and central of the related cluster the 1% data which at farthest to the cluster centra are anomaly
		3) Using IsolationForest method
		4) Oneclass SVM
		5) EllipticEnvelope for outlier detection of data set with gauss distribution, a very important condition for this method, the data set must be gauss distributed

- CNN
	- In the jupyter notebook, My1stCNN, a CNN model with 2 converluation layers are built. The model is trained to recognize cats or dogs on the fotos.
		1. Inputlayer with preprocessing (rescaling, resizing)
		2. 1st Converlution layer 
		3. Maxpooling layer
		4. 2nd Converlution layer
		5. Maxpooling layer
		6. Flatten layer
		7. Dense layer with Relu
		8. Output layer with sigmoid
	-  In the EvaMy1stCnn the created CNN model is evaluated with test data set and fotos from internet.
- Exercise
	- Coding examples with some essential machine learning algorithms
		- Linear regression
		- Logistic regression
		- KMeans clsuter
		- KNN
		- MeanShift