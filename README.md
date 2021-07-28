# Panama-s-Daily-Short-Term-Load-Forecasting
A STLF case study for the country of Panama based on daily granularity using various regression techniques and time series methodologies(SARIMAX).


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Introduction](#introduction)
* [Data Source](#data-source)
* [Tools](#tools)
* [Data Preprocessing](#data-preprocessing) 
  * Feature selection
  * Statistical methods and plotting
  * Outlier detection
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

The Dataset for Panama's National Electricity Consumption was downloaded from [kaggle](https://www.kaggle.com/ernestojaguilar/shortterm-electricity-load-forecasting-panama) website.
* CSV File : continuous_dataset.csv
* Description of data and its features: Columns metadata and train-test splits details.pdf


<!--TOOLS-->
## Tools
* Machine Learning and Time Series models have been coded in Python programming language using the Jupyter Notebook editor.

<!--DATA PREPROCESSING-->
## Data Preprocessing

* **Missing Values:**
The data has no missing values.


* **Outlier detection:**
![image](![image](https://user-images.githubusercontent.com/81852314/127360335-172d4690-4f5b-447f-9ee8-e0154a81e3f0.png))


<!-- PREDICTION MODELS-->
## Prediction Models
* **Model 1:** Initially, the data have been trained on 4 models:
  * Linear Regression
  * Decision Tree Regression
  * Random Forest Regression
  * Support Vector Machine
  The results (accuracies) from these models were
not up to the mark and thus thumbnail is introduced in II
part to improve the output.
* **Model 2:**  The training of the model has been done in 2 parts: <br/>
  * The first part focuses on extracting the feature, ”thumbnail
result”. Convolutional Neural Network has been
used for the same. Dataset for the CNN consists of the
Thumbnails and the labelled views. The actual views
have been mapped into labelled views by classifying the
views into 10 classes. The data is then trained with the
CNN and the resulting labels, giving an approximation of
the actual views are stored as “thumbnail result” to be used
further. <br/>
  **Architecture of CNN:** <br/>
Number of Layers: 6 <br/>
*Layer1:* Convolution Layer with, kernel size = 5, padding
= 2, stride = 1, dropout = 0.25, mapping one input channel
into 16 channels. <br/>
After this, ReLu is applied on the outputs.<br/>
*Layer2:* Max Pooling Layer with stride = 2 <br/>
*Layer3:* Convolution Layer with, kernel size = 5,
padding = 2, stride = 1, dropout = 0.25, mapping 16 input
channel into 32 channels.<br/>
After this, ReLu is applied on the outputs.<br/>
*Layer4:* Max Pooling Layer with stride = 2<br/>
*Layer5:* Fully Connected Layer mapping into 100
neurons, dropout = 0.25.<br/>
*Layer6:* Fully Connected Layer mapping 100 neurons
into 10 class probabilities. <br/>
* The ”thumbnail result” feature is now used with
the above-mentioned features and has been trained with the
following models, with the target variable being the actual
number of views.
    * Support Vector Regressor
    * Random Forest Regressor
     ![image](https://user-images.githubusercontent.com/81852314/120922056-aea7ae80-c6e4-11eb-8516-a98ac286be7b.png)


<!--RESULTS-->
## Results
* Model 1:
  * Linear Regression: <br/>
R2 Score on the Training Data: 0.6364104183115034 <br/>
R2 Score on the Test Data: 0.6282098478473409 <br/>
  * Decision Tree Regressor: <br/>
R2 Score on the Training Data: 0.999999740665929 <br/>
R2 Score on the Test Data: 0.5009736557269098  <br/>
  * SVR: <br/>
R2 Score on the Training Data: 0.7536050917458403 <br/>
R2 Score on the Test Data: 0.7255821853378172 <br/>
* Model 2:  <br/>
  * SVR:  <br/>
R2 Score on the Training Data: 0.955  <br/>
R2 Score on the Test Data: 0.92  <br/>
  * RandomForest:  <br/>
R2 Score on the Training Data: 0.97  <br/>
R2 Score on the Test Data: 0.95  <br/>

<!-- CONCLUSION-->
## Conclusion
Thumbnail proved out an extremely important feature
for sure as the results improved significantly.
Seeing the results without thumbnail, only 2 models
were considered most optimal, the SVR and the Random
Forest Regressor.
CNN + Random Forest turns out to be the best model
for this dataset.

<!--RECOMMENDATION AND FUTURE WORK-->
## Recommendations and Future Work
* A simple application could be made using thde developed algorithms
where a user can enter all the features of their upcoming
video having details of thumbnail along with other
attributes.
* Few more important features like
Title of the video and Description of the video can be used as inputs and an NLP Model can be used to predict more reliable results.
