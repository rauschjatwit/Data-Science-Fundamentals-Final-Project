# Data-Science-Fundamentals-Final-Project
# Introduction
The objective of this project was to see if it was possible to accurately predict a student’s final grade in math based off various social factors in their life. This was done using two different methods, a linear regression model and a neural network model. 

# Selection of Data
The predictions were made using a student performance dataset found online from Kaggle and can be found here. This is two datasets, one with values pertain to student grades in a Portuguese class and a math class. For simplicity's sake, this project only deals with the math class dataset. 
This dataset is built up of 33 rows of attributes including the student’s first, second, third, and final grades in the class with 397 samples. Although this dataset is not great in size, the data was interesting enough that it held out hope that strong correlations could be drawn between them and final grades. 
An important first step when creating predication models was to clean the data and take out certain attributes that had no bearing on grades along with certain outliers in the dataset, both would only confuse the prediction models and increase the error. However, there was lots of caution used when selecting what to keep and what to drop especially when it came to taking out outliers since in certain cases, they help make a more dynamic and reliable neural network worthy of making more accurate predictions. 

# Justin Method and Results Strategy 1:
The first approach was creating a neural network using the TensorFlow python library. This model was built off the idea that the student’s final grade could be closely approximated using the midterm grades. The original data was all dropped when training this model except for all 397 rows of G2 (midterm grades) and G3 (final grades). The data was then split into a training dataset, which was 80% of the original data, and a testing dataset, the remaining 20%. 
Instead of jumping right into the neural network, the model was kept linear and was used to find the line of best fit.

![image](https://user-images.githubusercontent.com/70958977/207210560-8891cf5b-3389-4671-822d-1a27b5ceff50.png)

This performed surprisingly well, but the desired model would show some appreciation for the outliers on the bottom of the graph and attempt to reduce the error. 
The next step was to add some layers in the neural network and play with the epoch to see how the same graph as before would change to be more in line with the data it was trained off.

![image](https://user-images.githubusercontent.com/70958977/207210750-9c800b93-dd3a-4c1e-891b-e775c1c4358d.png)

As you can see above, this model does much better at being more lenient when it comes to some of the outliers at the bottom of the graph and still performs when it comes to predicting the rest of the values.

![image](https://user-images.githubusercontent.com/70958977/207211444-901f4df3-b6f6-4b14-a19d-f2e0c089224f.png)

The above graph shows the frequency of the error error in these predictions. The graph indicates that the model is a little generous in its predictions, but despite some outliers, is rather close to the actual values. This was the result of the following neural network in TensorFlow:

![image](https://user-images.githubusercontent.com/70958977/207212798-0ff07061-68d3-4c40-b404-46367b387c1e.png)

# Jake Method and Results Strategy 2: 
Another approach looked at used multivariable regression. This solution allowed for multiple factors to be looked at. Based on the equation: 

![image](https://user-images.githubusercontent.com/71090844/207212535-e11140f3-451e-4fa6-8997-65e74414e6ec.png)

The equation would have n amount of variables (in this case the amount of factors being looked at):

y = is the output predicted value (G3)
a = Coefficient(slope) for the variable x(the specific factor)
x = the input of the specific factor for the student’s G3 being predicted
b = the intercept

![image](https://user-images.githubusercontent.com/71090844/207213106-da363faf-ff2a-490e-bc91-ab5ea398b6b1.png)

Using a similar approach to the G2 TensorFlow model from earlier:
-	Trainable variables are defined
-	A model is defined
-	An optimizer object is used (this case it’s the Adam Optimizer)
-	A loop is created (tested at 2,000, 10,000, and 50,000 iterations)

![image](https://user-images.githubusercontent.com/71090844/207212338-7b5dd7c5-aa46-461b-adab-ca713d558069.png)

To test the multivariable regression, I edited the code so that n=2 and the data was retrieving the columns of G1 and G2. 

![image](https://user-images.githubusercontent.com/71090844/207213172-726b4be1-8504-4a83-86bf-549b73fadb17.png)

![image](https://user-images.githubusercontent.com/71090844/207213243-3392532d-f418-4b6a-a923-64df2bfc058a.png)

As seen above the first test resulted in an answer of 10.7. This resulted in a close score, but not the exact answer of 11. 

![image](https://user-images.githubusercontent.com/71090844/207213285-45cde265-bf97-458c-a2a3-28122a6c72d1.png)

The second test resulted in an anoswer of 15.2 but the G3 was actually 16. 

By changing the n variable to 24 and gathering all 24 columns of data two more tests were performed: 

![image](https://user-images.githubusercontent.com/71090844/207213707-054e58e7-2caa-418c-ad4b-0e18fb0a65ce.png)

As seen above the 1st result was negative. This resulted in a result meant to be 0, but the program was not aware that a result could be not negative. 

![image](https://user-images.githubusercontent.com/71090844/207213740-ebbbec27-dd4a-4c12-872b-fee8872cbf74.png)

The second result for this 24 parameter test was 9.4 when the result was meant to be 10. Still close but the error appeared just as great as the two variable test. 

Lastly, the next question being asked is would more iterations in the for loop create a more accurate result?

![image](https://user-images.githubusercontent.com/71090844/207214258-7dc6f529-48db-48f6-8564-3015cfff2aef.png)

![image](https://user-images.githubusercontent.com/71090844/207214278-ca1b8a56-5351-4542-8662-6d1d25ff98ad.png)

As seen above a result of 7.68 was the output when G3 was really 9. 

Based on the tests performed with multivariable regression because the slope is being calculated, it is difficult for G3 to be predicted exactly. However, with the outliers eliminated, more data (more student scores), and parameters regarding rounding, an exact score outcome is possible, but the likelyhood of 100% accuracy for every student is low. 
# Discussion
After tampering with the neural network, I believe that the parameters could be changed to allow the prediction of the student’s final grade (G3) by using more than just the student’s midterm grade (G2). This would involve combining all the parameters and using the ‘flatten’ functions to turn them into one column. This would allow the neural network to take all these parameters in as inputs and train the weights and biases accordingly.

# Summary
This project allows us to see that although the predictions our models made were not as accurate as they could be, there is a lot of potential for them to do better.  

One of the downfalls of the models is the amount of data they are attempting to train from. Compared to most large datasets, 397 samples are not ‘a lot’ of data. However, more data cannot be the solution to everything. The TensorFlow model would have greatly benefitted from taking in more factors in order to make a prediction, especially since the outliers of the data in the G2 column may not appear in other factors and may end up alleviating these outliers and even be able to predict them.
