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

![image](https://user-images.githubusercontent.com/70958977/207210750-9c800b93-dd3a-4c1e-891b-e775c1c4358d.png)

As you can see above, this model does much better at being more lenient when it comes to some of the outliers at the bottom of the graph and still performs when it comes to predicting the rest of the values.

![image](https://user-images.githubusercontent.com/70958977/207211444-901f4df3-b6f6-4b14-a19d-f2e0c089224f.png)

The above graph shows the frequency of the error error in these predictions. The graph indicates that the model is a little generous in its predictions, but despite some outliers, is rather close to the actual values.

# Jake: 
Another approach looked at used multivariable regression. This solution allowed for multiple factors to be looked at. Based on the equation: 

![image](https://user-images.githubusercontent.com/71090844/207212535-e11140f3-451e-4fa6-8997-65e74414e6ec.png)

The equation would have n amount of variables (in this case the amount of factors being looked at):

y = is the output predicted value (G3)
a = Coefficient(slope) for the variable x(the specific factor)
x = the input of the specific factor for the student’s G3 being predicted
b = the intercept

![image](https://user-images.githubusercontent.com/71090844/207212338-7b5dd7c5-aa46-461b-adab-ca713d558069.png)

Using a similar approach to the G2 TensorFlow model from earlier:
-	Trainable variables are defined
-	A model is defined
-	An optimizer object is used (this case it’s the Adam Optimizer)
-	A loop is created (tested at 2,000, 10,000, and 50,000 iterations)
//insert code

# Results

# Discussion

# Summary

