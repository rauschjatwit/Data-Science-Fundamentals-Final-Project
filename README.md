# Data-Science-Fundamentals-Final-Project
# Introduction
The objective of this project was to see if it was possible to accurately predict a student’s final grade in math based off various social factors in their life. This was done using two different methods, a linear regression model and a neural network model. 

# Selection of Data
The predictions were made using a student performance dataset found online from Kaggle and can be found here. This is two datasets, one with values pertain to student grades in a Portuguese class and a math class. For simplicity's sake, this project only deals with the math class dataset. 
This dataset is built up of 33 rows of attributes including the student’s first, second, third, and final grades in the class with 397 samples. Although this dataset is not great in size, the data was interesting enough that it held out hope that strong correlations could be drawn between them and final grades. 
An important first step when creating predication models was to clean the data and take out certain attributes that had no bearing on grades along with certain outliers in the dataset, both would only confuse the prediction models and increase the error. However, there was lots of caution used when selecting what to keep and what to drop especially when it came to taking out outliers since in certain cases, they help make a more dynamic and reliable neural network worthy of making more accurate predictions. 

# Methods
# Justin
The first approach was creating a neural network using the TensorFlow python library. This model was built off the idea that the student’s final grade could be closely approximated using the midterm grades. The original data was all dropped when training this model except for all 397 rows of G2 (midterm grades) and G3 (final grades). The data was then split into a training dataset, which was 80% of the original data, and a testing dataset, the remaining 20%. 
Instead of jumping right into the neural network, the model was kept linear and was used to find the line of best fit.
![image](https://user-images.githubusercontent.com/70958977/207210560-8891cf5b-3389-4671-822d-1a27b5ceff50.png)

This performed surprisingly well, but the desired model would show some appreciation for the outliers on the bottom of the graph and attempt to reduce the error. 
The next step was to add some layers in the neural network and play with the epoch to see how the same graph as before would change to be more in line with the data it was trained off.

# Results

# Discussion

# Summary

