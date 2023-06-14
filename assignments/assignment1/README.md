# CSE 6363: Assignment 1

This assignment covers Linear Models for regression and classification. Linear Regression is a method of predicting real values given some input. Logistic Regression and Linear Discriminant Analysis will be used as classifiers.

These models have been implemented over and over again and are available in many popular machine learning frameworks. It is important to implement the models yourself so that you gain a deeper understanding of them.

# 1 Implementation of the `LinearRegression` class

Before evaluating any data, we need some code to actually `fit`, `predict`, and `score` samples. This will be implemented in `LinearRegression.py` provided in this repository. The skeleton of the class is already there. In part 1, you will need to implement the `fit`, `predict`, and `score` functions.

After implementing these 3 functions, you will be able to use this model simply with any regression task.

## 1.2 The `fit` method

The `fit` method should accept 6 parameters:
1. the input data
2. the target values
3. `batch_size, int` - The size of each batch during training
4. `regularization, int` - The factor of L2 regularization to add, default to 0
5. `max_epochs, int` - The maximum number of times the model should train through the entire training set
6. `patience, int` - The number of epochs to wait for the validation set to decrease

Other parameters can be added as long as they are optional.

This method should use gradient descent to optimize the model parameters using mean squared error as the loss function. So that the model will converge to a solution, early stopping must be used. To do this, set aside 10% of the training data as a validation set. After each step of gradient descent, evaluate the loss on the validation set. If the loss on the validation set increases for 3 consecutive steps, stop training. If it decreases, save the current model parameters. After training is complete, used the saved parameters to set the model parameters.

## 1.3 The `predict` method

The `predict` method should accept 1 parameter:
1. the input data

This method should run a forward pass of the model and return the predicted values. Given $n$ samples with $d$ features each and $m$ output values, let $X \in \mathbb{R}^{n \times d}$ be the input data, $W \in \mathbb{R}^{d \times m}$ be the model parameters, and $\mathbf{b} \in \mathbb{R}^{n \times m}$ be the bias terms. The predicted values are given by:

$$
\mathbf{y} = X W + \mathbf{b} \in \mathbb{R}^{n \times m}
$$

## 1.4 The `score` method

The `score` method should accept 2 parameters:
1. the input data
2. the target values

This method will predict the values for the input data and then compute the mean squared error between the predicted values and the target values. The mean squared error is given by:

$$
\text{MSE} = \frac{1}{nm} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2
$$

where $n$ is the number of samples, $m$ is the output size, $y_i$ is the target value for the $i$th sample, and $\hat{y}_i$ is the predicted value for the $i$th sample.

## 1.5 Saving and Loading Weights

After training, the model parameters should be saved to a file. The `save` method should accept a file path. This method should save the model parameters to the given file path. The model parameters should be saved in a format that can be easily loaded back into the model.

The `load` method should also accept a file path and load the model parameters accordingly.

## 1.6 Regression with a single output

The Iris flower dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set) was organized by Ronald Fisher in 1936. It is a commonly used dataset for introductory machine learning concepts. You will use this dataset for fitting and evaluating your regression model.

**The training and testing should be contained in a single evaluation script so that the results can be easily reproduced.**

### 1.6.1 Preparing the Data

Much of machine learning is in understanding the data you are working with. For our regression task, we want to predict continuous outputs given some input feature(s).

Each sample in the Iris dataset has 4 features:
- sepal length
- sepal width
- petal length
- petal width

We may want to know which feature is most predictive of another. Does knowledge of the sepal length provide good estimates of the sepal width? What about the petal width? What if we use the sepal length and width to predict the petal length or width?

In this section of the assignment, you will create 4 different regression models to answer some of these questions. This will be trivial to do once the code for the model is finished.

To begin, load the data using `scikit-learn`. In order to verify the models that you will create in the following two sections, you will need to take some portion of the dataset and reserve it for testing. Randomly select 10% of the dataset, ensuring an even split of each class. This will be your **test** set. Note that this is different than the random 10% that is taken from the training set when in the `fit` method. The rest of the data will serve as your **training** set.

### 1.6.2 Training

Select 4 different combinations of input and output features to use to train 4 different models. For example, one model could predict the petal width given petal length and sepal width. Another could predict sepal length using only petal features. It does not matter which combination you choose as long as you have 4 unique combinations.

Your models should be trained using batch gradient descent with a batch size (optional parameter) of 32 using mean squared error as your loss function.

For each model, train for $n = 100$ steps (optional parameter) OR until the loss on the validation set increases for a number of consecutive epochs determined by `patience` (default to 3).

As each model trains, record the loss averaged over the batch size for each **step**. A single step is the processing of a single batch. One way to save this data is to either return an array from the fit method or save it as an internal class member that can be retrieved after training is complete.

**After each model trains, plot the loss against the step number and save it. These plots should also be added to your report.**

To observe the effects of regularization, pick one of your trained models and inspect the weights. Train an identical model again, except this time you will add L2 regularization to the loss. Record the difference in parameters between the regularized and non-regularized model.

**Record these values into your report so they can be verified.**

Create a separate training script for each model that you created. Name the scripts `train_regression1.py`, `train_regression2.py`, etc. This should include training the model, saving the model parameters, and plotting the loss.

### 1.6.3 Testing

For each model you created, test its performance on unseen data by evaluating the mean squared error against the test dataset that you set aside previously. This should be implemented as 4 separate scripts. Each script should load the model parameters from the respective model and then evaluate the model on the test set. The mean squared error should be printed to the console. Name the scripts `eval_regression1.py`, `eval_regression2.py`, etc.

In your report, briefly describe which input feature is most predictive of its corresponding output feature based on your experiments.

# 2 Classification

Similar to part 1, you will need to create a class for each classification method. These classes should implement both a `predict` and `fit` method.

The `fit` method should take as input an `ndarray` of data samples and target values (the classes). It should then optimize the set of parameters following the respective training method for that classification method.

The `predict` method should take as input an `ndarray` of samples to predict.

For each classification method that is implemented, you will need to compare 3 variants of input features:
1. petal length/width
2. sepal length/width
3. all features

For the first two, include visualizations of the classifier using `plot_decision_regions` from `mlxtend` (https://github.com/rasbt/mlxtend). This plotting function works with your trained classifier, assuming you have implemented a `predict` method.

## 2.1 Logistic Regression

For the first classifier, implement a `LogisticRegression` class similar to how the `LinearRegression` class was implemented. The `fit` method should use either the normal equations or gradient descent to come up with an optimal set of parameters.

## 2.2 Linear Discriminant Analysis

The second model you will explore in this assignment is Linear Discriminant Analysis. Implement both a `fit` and `predict` method following the details [https://dillhoffaj.utasites.cloud/posts/linear_discriminant_analysis](described here.)

The parameter update equations were derived via Maximum Likelihood Estimation and can be estimated directly from the data. You do not need to create a covariance matrix for each class. Instead, use a shared covariance matrix which is computed as

$$
\Sigma = \frac{1}{n} \sum_{k=1}^K n_k \Sigma_k,
$$

where $n$ is the total number of samples, $n_k$ is the number of samples belonging to class $k$, and $\Sigma_k$ is the covariance matrix for class $k$.

## 2.3 Testing

For each trained model, compute the accuracy on the test set that was set aside for each data variant. Since there are 3 variants, there should be 3 comparisons of Logistic Regression versus LDA. Implement each variant evaluation as a separate script. Name the scripts `eval_classifiers1.py`, `eval_classifiers2.py`, etc.

These scripts should load the best trained weights, evaluate the accuracy on the test set, and print the accuracy to the console.

# Submission

Create a zip file that includes all of your code as well as your report. The TA should
be able to easily run the code to reproduce all plots and results. Include any additional
instructions, if necessary.
