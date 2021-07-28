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
* [Results](#Results)
* [Conclusion](#Conclusion)
* [Recommendations and Future Work](#recommendations-and-future-work)

<!-- INTRODUCTION -->
## Introduction

This project focuses on Short Term Electricity Load Forecasting for the country of Panama on a daily basis. Machine Learning techniques such as Multivariate Regression methodologies and Time Series Analysis and Forecasting models such as ARIMA and SARIMA were used to predict the amount of daily national electricity consumption for Panama. Features such as weekday, weekend, time of the month, holidays, schoolday etc. and environmental factors such as average temperature, humidity and wind speed of the day were also taken into consideration for model building.

<!--DATA SOURCE-->
## Data Source

The Dataset for Panama's National Electricity Consumption was downloaded from [kaggle](https://www.kaggle.com/ernestojaguilar/shortterm-electricity-load-forecasting-panama) website for the datetime: 03-01-2015  01:00:00 to 27-06-2020 00:00

* CSV File : continuous_dataset.csv
* Description of data and its features: Columns metadata and train-test splits details.pdf


<!--TOOLS-->
## Tools
* Machine Learning and Time Series models have been coded in Python programming language using the Jupyter Notebook editor.

<!--DATA PREPROCESSING-->
## Data Preprocessing

* **Missing values:**
The data has no missing values.

* **Outlier detection:**
Outliers detection was attempted using the IQR Method. A data point is an outlier if it lies below Q1-1.5(IQR) or Q1+1.5(IQR). No strict outliers were found in the dataset.

* **Data resampling:**
Data was originally available with hourly granuality for National Electricity Consumption and has been transformed to daily granuality for the sake of forecasting daily electricity consumption.

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
    * Params: 
  * **Random Forest Regression:**  </br>
    * Number of features: 13
    * Params: RandomForestRegressor(n_estimators= 1400, min_samples_split=5, 
                                  min_samples_leaf= 1, max_features='sqrt', max_depth = 30, bootstrap= True)
  * **KNN Regression:** </br>
    * Number of features: 6
    * Params: KNeighborsRegressor(weights= 'uniform', n_neighbors= 42, metric='manhattan', leaf_size= 46)
    
  * **Support Vector Machine:** </br>
     * Number of features: 10
     * Params: RandomForestRegressorSVR(C=2.5, kernel='linear', tol= 6.362524107592032e-05)
     
  * **Gradient Boost Regression:** </br>
     * Number of features: 15
     * Params: GradientBoostingRegressor(n_estimators= 300, min_samples_split=10, min_samples_leaf=1,
                                          max_features = 'auto', max_depth=3, loss= 'ls', learning_rate= 0.05) 
  *  **XG Boost Regression:** </br>
     * Number of features: 15
     * Params: XGBRegressor(subsample = 0.799, n_estimators = 215, min_child_weight = 4, max_depth=3, 
                             learning_rate= 0.1, colsample_bytree = 0.899, colsample_bylevel = 0.4)
  
  
* **Model 2: Time Series Forecasting Model: SARIMAX:** 
  * Seasonal Autoregressive Integrated Moving Average model with an Exogenous Variable(Holiday) was built using the auto_arima function in pmdarima library.
    Data was randomly split for sequential training and testing purpose. 7 weeks from 2019 and 2020 were randomly selected. The model with parameters ARIMA(5,1,0)(2,0,0)[7]  
    performed good on the weeks split in 2019 but performed not so well for the pandemic hit weeks in 2020.



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
     
  *  **XG Boost Regression:** </br>
     * RMSE for test set is:  1144.94
     * Adjusted R2 score for test:  0.74
     * MAPE for test:  8.80
  
  
* **Model 2: Time Series Forecasting Model: SARIMAX:** 
  * Seasonal Autoregressive Integrated Moving Average model with an Exogenous Variable(Holiday) was built using the auto_arima function in pmdarima library.
    Data was randomly split for sequential training and testing purpose. 7 weeks from 2019 and 2020 were randomly selected. The model with parameters ARIMA(5,1,0)(2,0,0)[7]  
    performed good on the weeks split in 2019 but performed not so well for the pandemic hit weeks in 2020.


<!-- CONCLUSION-->
## Conclusion

<!--RECOMMENDATION AND FUTURE WORK-->
## Recommendations and Future Work

