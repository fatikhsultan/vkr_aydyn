# vkr_aydyn
Source Code for the Practical Part of the Master's Thesis: 'Analysis of Pricing in the Moscow Residential Real Estate Market Using Machine Learning Methods' by Aidyn Fatikh Sultan

The repository contains 3 folders.

regression_analysis contains Jupyter Notebook file with EDA and implementation of regression models presented in chapter 3 of the study.

time_series_analysis contains 2 notebooks related to autoregressive models implementation and LSTM

web_service folder contains all the necessary files for web-service operation. There is a "templates" folder that contains HTML-templates of the service. Additionally, there is a main.py file, which is the main module of the web application implementing the server-side logic. It processes incoming HTTP requests, receives data from the form, performs prediction using a trained model, and returns the result to the user. Also, there is a folder named static which contains the image of time series prediction. Also, there is run.ipynb file which allows to start the web service locally. Also, web_service folder contains a file with a pretrained XGB model, which is used for making predictions 
