# autoML 
##### Information technology study about machine learning using the libray scikit-learn for the Python programming language and their related concepts. 

First, we need to know that machine learning is the field of study that gives computers the ability to learn without being explicitly programmed, this is a denifition 
described by Arthur Samuel, an American pioneer in the field of computer gaming and artificial intelligence and the coined the term **"Machine Learning"** in 1959 while at 
IBM. Tom Mitchell provides a more modern definition: "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**,
if its performance at tasks in T, as measured by P, improves with experience E. So, int machine learning there are two types of algorithms to solve problems of data 
analytics: the **supervised Learning** and the **unsupervised learning**. Finally, in learning data science with Python, generally, we use the **scikit-learn**, a open source Python
library that implements a wide variety of machine learning,  preprocessing, cross-validation and visualization algorithms with the help of a unified interface. Depending
on the input data, we will use different methods implemented in the the library, as shown in the image below:
<p align="center"> <img src="./images/01_sklearn_algorithms.png"> </p>

## Supervised Learning 

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a **continuous output**, 
meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words,
we are trying to map input variables into discrete categories. Examples of supervised learning are: Regression, Decision Tree, Random Forest, KNN, Logistic Regression, etc.

###### Linear Regression

Linear regression attempts to model the relationship between two variables by **fitting** a linear equation to **observed** data. One variable is considered to be an 
explanatory variable, and the other is considered to be a dependent variable.  Before attempting to fit a linear model to observed data, a modeler should first
determine whether or not there is a relationship between the variables of interest. This does not necessarily imply that one variable causes the other, but that
there is some significant association between the two variables. 
A linear regression line has an equation of the form **Y = a + bX**, where X is the **explanatory** variable and Y is the **dependent** variable. The slope of the line is
b, and a is the intercept (the value of y when x = 0). See the example below. Here we have identified the best fit line having the linear equation y = 0.2811x + 13.9.
Now, using this equation, we can find the weight, knowing the height of a person.
<p align="center"> <img src="./images/02_linear_regression.png"> </p>
Finally, linear regressions can be two types: Simple Linear Regression and Multiple Linear Regression. Simple Linear Regression is characterized by an 
independent variable and multiple linear regression is characterized by multiple independent variables and by looking for the best fit line. So below there is a python code with an example using linear regression:

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/linear_model.html).
###### Logistic Regression
```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
###### Decision Tree
```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/tree.html).
###### SVM
```python
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/svm.html).
###### Naive Bayes
```python
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/naive_bayes.html).
###### KNN
```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/neighbors.html).
###### Random Forest
```python
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/ensemble.html).

###### Gradient Boost and Adaboost
```python
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/ensemble.html).

###### Neural Network
```python
#Import Library
from sklearn.model_selection import train_test_split
#Split in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
# Create an instance of the model. For this parameter you pass in a tuple consisting of the number of neurons you want at each layer, 
# where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model. We will choose 3 layers with the same
# number of neurons as there are features in our data set along with 500 max iterations.
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
# Built in metrics such as a classification report and confusion matrix to evaluate how well our model performed
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. 
We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by clustering the data based on relationships among the variables in the data.
Examples of Unsupervised Learning are: Apriori Algorithms and averages approximation.

###### K-Means
```python
#Import Library
from sklearn.cluster import KMeans
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model 
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X)
#Predict Output
predicted= model.predict(x_test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/clustering.html#k-means).

###### Dimensionality Reduction Algorithm
```python
#Import Library
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)
```
For more detail on this, please refer  this [link](http://scikit-learn.org/stable/modules/decomposition.html).

## References

VOOO. [Fundamentos dos Algoritmos de Machine Learning (com código Python e R)](https://www.vooo.pro/insights/fundamentos-dos-algoritmos-de-machine-learning-com-codigo-python-e-r/).

Wikipedia. [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning).

Coursera. [Machine Learning](https://www.coursera.org/learn/machine-learning/).

Datacamp. [Scikit-Learn Cheat Sheet: Python Machine Learning] (https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet).

Scikit Learn. [User guide: contents] (http://scikit-learn.org/stable/user_guide.html).

Scikit Learn. [API Reference] (http://scikit-learn.org/stable/modules/classes.html).

Github. [Introduction to machine learning with scikit-learn] (https://github.com/justmarkham/scikit-learn-videos).

Yale. [Linear Regression] (http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm).

Springboard. [A Beginner’s Guide to Neural Networks in Python and SciKit Learn 0.18]  (https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/).
