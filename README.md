# Panama-s-Daily-Short-Term-Load-Forecasting
A STLF case study for the country of Panama based on daily granularity using various regression techniques and time series methodologies(SARIMAX).


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Introduction](#introduction)
* [Data Source](#data-source)
* [Tools](#tools)
* [Data Preprocessing](#data-preprocessing) 
  * Missing Data
  * Outlier Detection
  * Data Resampling
  * Feature selection
  * Statistical parameters

* [Prediction Models](#prediction-models)
  * Machine Learning Technique : Regression Methodologies
  * Time Series Analysis and Forecasting : SARIMAX
* [Error Metrics](#error-metrics)
* [Results](#Results)
* [Conclusion](#Conclusion)
* [Recommendations and Future Work](#recommendations-and-future-work)

<!-- INTRODUCTION -->
## Introduction

This project focuses on Short Term Electricity Load Forecasting for the country of Panama on a daily basis. Machine learning techniques such as multivariate regression methodologies and time series analysis and forecasting models such as ARIMA and SARIMA were used to predict the amount of daily national electricity consumption for Panama. Features such as weekday, weekend, time of the month, holidays, schoolday etc. and environmental factors such as average temperature, humidity and wind speed of the day were also taken into consideration for model building.

<!--DATA SOURCE-->
## Data Source

The dataset for Panama's national electricity consumption was downloaded from [kaggle](https://www.kaggle.com/ernestojaguilar/shortterm-electricity-load-forecasting-panama) website for the datetime: 03-01-2015  01:00:00 to 27-06-2020 00:00

* CSV File : continuous_dataset.csv
* Description of data and its features: Columns metadata and train-test splits details.pdf


<!--TOOLS-->
## Tools
* Machine learning and time series models have been coded in Python programming language using the Jupyter notebook editor.

<!--DATA PREPROCESSING-->
## Data Preprocessing

* **Missing values:**
The data has no missing values.

* **Outlier detection:**
Outliers detection was attempted using the IQR Method. A data point is an outlier if it lies below Q1-1.5(IQR) or Q3+1.5(IQR). No strict outliers were found in the dataset.

* **Data resampling:**
Data was originally available with hourly granuality for national electricity consumption and has been transformed to daily granuality for the sake of forecasting daily electricity consumption.

* **Feature selection:** Intially the following features were selected and engineered:
  * Type of the Day:
    * Sunday
    * Saturday
    * Weekday
    * National Holiday
    * Weekday
    * Lockdown
  * Time of the year divided by months quaterly (Q1, Q2, Q3, Q4)
  * Environmental Factors (of biggest 3 cities of Panama: Tocumen, David, Santiago):
    * Average Temperature 
    * Average Humdity
    * Average Wind speed
    * Average Precipitation
  * Lagged Demand (Electricity demand of the same day previous week)
  * XGBoost Regressor, an ensemble feature selection technique was used to compute feature importances.
 
 * **Statistical parameters:** The following three parameters were calculated to ensure integrity and validity of features in the model:
   * Multicollinearity using pearson correltaion and Variance Inflation Factor (VIF) technique was checked.
   * Pearson Correlation: Correlation matrix was formed to eliminate any features that broadcasted a correlation > 0.7 amongst themselves:
   ![image](https://user-images.githubusercontent.com/81852314/127386356-7e40e19a-a1d0-4d22-8edc-a643d54c5a61.png)
   * VIF: VIF was computed amongst independent varaibles to elimate if any variable showcased a VIF > 4. None of the variables showcased VIF above the elimination threshold.
   * Partial Autocorrelation: To include the lagged demand variable of order 7 to account for the trend of the increase in electricty consumption, Partial Autocorrelation     Function (PACF) was plotted to check for the significance of order of lagged variable.
 ![image](https://user-images.githubusercontent.com/81852314/127388645-950d855e-5785-417f-b21c-11577826cb16.png)




<!-- PREDICTION MODELS-->
## Prediction Models
* **Model 1: Regression Techniques** Initially, the data have been trained on the following regression models:
  * **Multiple Linear Regression:** </br>
    * Number of features: 12
    * Params: (const:	-645.77, Weekday:	879.88, holiday:	-3011.05, lag_demand:	0.48, Sunday:	-988.74, Q3:	-551.94, Q1:	-57.96, avg_prec:	-7763.40, avg_humidity:	3.562e+05,
               Q2:	-412.71, avg_wind:	38.93, avg_temp:	358.13, Lockdown:	-1204.17)

  * **Random Forest Regression:**  </br>
    * Number of features: 13
    * Params: RandomForestRegressor(n_estimators= 1400, min_samples_split=5, 
                                  min_samples_leaf= 1, max_features='sqrt', max_depth = 30, bootstrap= True)
  * **KNN Regression:** </br>
    * Number of features: 6
    * Params: KNeighborsRegressor(weights= 'uniform', n_neighbors= 42, metric='manhattan', leaf_size= 46)
    
  * **Support Vector Machine:** </br>
     * Number of features: 10
     * Params: SVR(C=2.5, kernel='linear', tol= 6.362524107592032e-05)
     
  * **Gradient Boost Regression:** </br>
     * Number of features: 15
     * Params: GradientBoostingRegressor(n_estimators= 300, min_samples_split=10, min_samples_leaf=1,
                                          max_features = 'auto', max_depth=3, loss= 'ls', learning_rate= 0.05) 
  
  
* **Model 2: Time Series Forecasting Model: SARIMAX:** 
  * Seasonal Autoregressive Integrated Moving Average model with an Exogenous Variable(Holiday) was built using the auto_arima function in pmdarima library.
  * Params: pm.auto_arima(train_data, m = 7, seasonal = True, exog = train4'holiday',                    
                    start_p = 0, start_q = 0, max_order = 5, test = 'kpss',
                     error_action = 'ignore', suppress_warnings = True,
                     stepwise = True, trace = True)
  * KPSS test statistic was used to check for stationarity in the data and differencing of order 7 was used to eliminate the stationarity.
  * Per the formula SARIMA(p,d,q)x(P,D,Q,s), the parameters for the model are as follows:
      * p and seasonal P: indicate number of autoregressive terms (lags of the stationarized series)
      * d and seasonal D: indicate differencing that must be done to stationarize series
      * q and seasonal Q: indicate number of moving average terms (lags of the forecast errors)
      * s: indicates seasonal length in the data
    Data was randomly split for sequential training and testing purpose. 7 weeks from 2019 and 2020 were randomly selected. The model with parameters ARIMA(5,1,0)(2,0,0)[7]  
    performed good on the weeks split in 2019 but performed not so well for the pandemic hit weeks in 2020.
    
    
<!-- ERROR METRICS-->
## Error Metrics
* **RMSE:** Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). </br>
    `from sklearn.metrics import mean_squared_error` </br>
    `mse = mean_squared_error(y_test, y_pred)` </br>
    `rmse = pow(mse,1/2)` </br>
* **Adjusted R squared:** It measures the proportion of variation explained by only those independent variables that really help in explaining the dependent variable. It penalizes you for adding independent variable that do not help in predicting the dependent variable.

  `import numpy as np` </br>
  `from sklearn.metrics import r2_score` </br>
  `def adj_r2(actual, pred, i):` </br>
   `--->return 1 - ((1-r2_score(actual,pred))(len(actual)-1)/(len(actual)-i-1))` </br>
    
* **MAPE:** The mean absolute percentage error (MAPE) is a measure of prediction accuracy of a forecasting method in statistics. It usually expresses the accuracy as a ratio.

  `import numpy as np`</br>
  `def mape(actual, pred):`</br> 
  ` --->actual, pred = np.array(actual), np.array(pred)`</br>
  ` --->return np.mean(np.abs((actual - pred) / actual)) * 100`</br>
      
    
<!--RESULTS-->
## Results
* **Model 1: Regression Techniques** 
  * **Multiple Linear Regression:** </br>
    * RMSE for test set is:  1237.67
    * Adjusted R2 score for test:  0.70
    * MAPE for test:  3.44
   
   
  * **Random Forest Regression:**  </br>
    * RMSE for test set is:  1130.40
    * Adjusted R2 score for test:  0.75
    * MAPE for test:  8.50
    
  * **KNN Regression:** </br>
    * RMSE for test set is:  1671.60
    * Adjusted R2 score for test:  0.47
    * MAPE for test:  4.33
    
  * **Support Vector Machine:** </br>
     * RMSE for test set is:  1619.41
     * Adjusted R2 score for test:  0.49
     * MAPE for test:  8.24
     
  * **Gradient Boost Regression:** </br>
     * RMSE for test set is:  1124.57
     * Adjusted R2 score for test:  0.75
     * MAPE for test:  8.79

  
  
* **Model 2: Time Series Forecasting Model: SARIMAX:** 
    * MAPE for test:  3.75
    
    
<!-- CONCLUSION-->
## Conclusion
* Multiple Linear Regression:</br>
    * Multiple Linear Regression seems to generalize the model the best for both train and test set: 
      * Adjusted R2 score for test data:  0.70
      * Adjusted R2 score for train data:  0.72
    * We get the least MAPE value for MLR: 3.44
    * The P value is significant for all the features (<0.005)
    * Error Analysis: The error terms are also normally distributed.</br>
    ![image](https://user-images.githubusercontent.com/81852314/127469916-527f987a-5f08-412d-96ab-aa90f58283c4.png)
    
 * Random Forest Regression model overfits with Adjusted R2 score for test:  0.75 and Adjusted R2 score for training:  0.94
 * The error terms for Gradient Boost Regresssion are not stable.
 * The independent features do not seem to fit KNN and Support Vector Regression well.
 
* SARIMA model with an exogenous effect of holiday factor seems to work well with a low MAPE score for the data except for the weeks at beginning of lockdown in the year 2020, but seems to adjust itself in the succeeding weeks:
* Model performance before the lockdown:
![image](https://user-images.githubusercontent.com/81852314/127471668-513b1d82-a4f8-49c6-8f3c-e58cd2390e34.png)
* Model performance right at the beginning of lockdown:
![image](https://user-images.githubusercontent.com/81852314/127472202-04264326-c05d-4172-a46e-fdc0ea5b3dd2.png)

* Model performance after adjusting to the lockdown
![image](https://user-images.githubusercontent.com/81852314/127471804-6cd4de6a-3437-48d0-80a7-02c67238ebef.png)

<!--RECOMMENDATION AND FUTURE WORK-->
## Recommendations and Future Work
A separate model for household electricity load consumption and industrial electricity load consumption can be built for better and precise predictions.
