# Weather challenge: implementing machine learning models to predict temperature and weather disposition
The objective of this project was to build a model to predict temperature and weather disposition of different cities.
To do the predictions I decided to build an individual model to fit with the data for each city.  
Since the target points in time to be predictive for the test were distant in the future from the last observation in the training set I decided not to use an autoregressive model approach, instead I picked machine-learning models to do the predictions based on the provided features.  
The available variables for each city were temperature, weather disposition, humidity, wind speed and direction, pressure and city attributes.  
As expected the hourly data temperature observations shows daily and yearly seasonality.  
To extract information to use from the time point of each observation I used the sine and cosine of the hour of the day and the day of the year. This allows to preserve the cyclical nature of the data, in example the hour 1 is very close to 24, the same with the 31st of December and the 1st of January. I prefer to use both sine and cosine so each hour or day can be clearly differentiated using the information in both variables.  
Next I decided to use forward fill to deal with the missing values. I prefer this method to using backward fill or interpolation when dealing with time series because these last two would involve using information from “future” data points. The variables use to change only slightly between each observation, supporting that the use of forward fill would not introduce misinformation. On the other hand, some of the gaps in the data seem big, in some cases bigger than 50 consecutive observations (‘this is not shown in the final version of the script’). Another possible strategy in this case could have been to drop this observations, but I preferred to preserved them to avoid losing information of the rest of the variables.
Before fitting the models minimal scaling is done of the predictors.  
To solve the prediction of the temperature I tried different regression models and compared them.  
For this I used grid search of different hyper parameters to tune the models and then compared the best models and picked one for each city.
The regression models used were: Random Forest, XGBoost and K-nearest neighbors.  
To evaluate the models to pick the best parameters and to estimate the performance of each and to be able to compare them I used 5 fold cross-validation and the mean squared error as the metric. A limitation was that 10 fold cross validation and LOOCV were not tried because of the higher time required to do the computations.  
The hyper parameters tuned were   
Random Forest:  
'n_estimators': [100,200,300],  
'max_features': ['sqrt', 'log2'],  
'max_depth' : [ 1, 11, 21, 31, 41, 51],  
XGBoost:  
'n_estimators': [10,20,30,40,50,100],  
'max_depth' : [ 1, 11, 21, 31, 41],  
KNN:  
´n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  
´weights': ['uniform','distance']   
XGBoost outperformed the other two in every city so it was the model I used to the final predictions the highest MSE was 3.96 (Denver) or an RMSE of 1.99 suggesting that the errors are usually less than 2°K.  
For the prediction of the weather disposition prediction I used a similar methodology. Here the metric I used was accuracy. An important limitation of the methodology	is the presence of important class imbalance. I used Non stratified 5 fold cross validation due to the fact that some classes had fewer observations than folds.  
The models I compared where in this case KNN and Random Forests with the following hyper parameters that were tuned:  
Random Forest:  
'n_estimators': [100,200,300],  
'max_features': ['sqrt', 'log2'],  
'max_depth' : [ 1, 11, 21, 31, 41, 51],  
KNN:  
´n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  
´weights': ['uniform','distance']  
In this case Random Forest performed better in all cities so it was the model I chose to do the final predictions.  
Another limitation is the limited hyper parameters that were tuned. All of the models have more hyper parameters that can be tuned. I explored tuning some of them in a preliminar analysis only fitting a model on the data of only one city (not shown in the submitted solution) but non of them showed great improvements in the performance of the models so I chose the presented above to try the models on all the data to lower the computation time.  
A further limitation is the fact that I could have used other models that might perform better like SVM, XGBoost classifier and Neural Networks.  
