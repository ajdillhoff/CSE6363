# Assignment 2

This assignment covers a couple methods for classification.

## 1. The Iris Dataset

As with assignment 1, we will be using the Iris flower data set (https://en.wikipedia.org/wiki/Iris_flower_data_set).

### 1.1 Preparing the Data

As before, load the Iris data.
`scikit-learn` provides a function for doing this quickly.

Set aside 10\% of the data set for testing.
Randomly select 10\%, ensuring an even split of each class.
The rest of the data will be used for training.

## 2. Classification

Similar to assignment 1, you will need to create a class for each classification method.
These classes should implement both a `predict` and `fit` method.
The `fit` method should take as input an `ndarray` of data samples and target values (the classes).
It should then optimize the set of parameters following the respective training method for that classification method.

The `predict` method should take as input an `ndarray` of samples to predict.

For each classification method that is implemented, you will need to compare 3 variants of input features:
1. petal length/width
2. sepal length/width
3. all features

For the first two, include visualizations of the classifier using `plot_decision_regions` from `mlxtend` (https://github.com/rasbt/mlxtend).
This plotting function works with your trained classifier, assuming you have implemented a `predict` method.

### 2.1 Logistic Regression

For the first classifier, implement a `LogisticRegression` class.
The `fit` method should use either the normal equations or gradient descent to come up with an optimal set of parameters.

### 2.2 Linear Discriminant Analysis

The second model you will explore in this assignment is Linear Discriminant Analysis.
Implement both a `fit` and `predict` method following the details [https://dillhoffaj.utasites.cloud/posts/linear_discriminant_analysis](described here.)

The parameter update equations were derived via Maximum Likelihood Estimation and can be estimated directly from the data.
You do not need to create a covariance matrix for each class.
Instead, use a shared covariance matrix which is computed as

$$
\Sigma = \frac{1}{n} \sum_{k=1}^K n_k \Sigma_k,
$$

where $n$ is the total number of samples, $n_k$ is the number of samples belonging to class $k$, and $\Sigma_k$ is the covariance matrix for class $k$.

## 3. Testing

For each trained model, compute the accuracy on the test set that was set aside for each data variant.
In your notebook, clearly display the accuracies.
Since there are 3 variants, there should be 3 comparisons of Logistic Regression versus LDA.

# Submission

Create a zip file that includes all of your code.
The TA should be able to easily run the code to reproduce all plots and results.
Include any additional instructions, if necessary.
