# Assignment 1

This assignment covers Linear Regression, Gradient Descent, Linear Discriminant Analysis, Logistic Regression, and Naive Bayes. These are classical methods that are very useful depending on your dataset, even if they are only used as a baseline.

These models have been implemented over and over again and are available in many popular machine learning frameworks. It is important to implement the models yourself so that you gain a deeper understanding of them.

## 1. The Iris Dataset

The Iris flower dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set) was organized by Ronald Fisher in 1936.
It is a commonly used dataset for introductory machine learning concepts.
You will use this dataset for both classification and regression.

### 1.1 Preparing the Data

To begin, load the data using `scikit-learn`.
As we saw during class, the setosa samples are very clearly linearly separable given any combination of two features.
However, the versicolor and virginica usually have some overlap.

In order to verify the models that you will create in the following two sections, you will need to take some portion of the dataset and reserve it for testing.
Randomly select 10% of the dataset, ensuring an even split of each class.
This will be your **test** set.
The rest of the data will serve as your **training** set.

## 2. Regression

Much of machine learning is in understanding the data you are working with.
We start with a regression task.
That is, we want to predict continuous outputs given some input feature(s).

Each sample in the Iris dataset has 4 features:
- sepal length
- sepal width
- petal length
- petal width

We may want to know which feature is most predictive of another.
Does knowledge of the sepal length provide good estimates of the sepal width?
What about the petal width?
What if we use the sepal length and width to predict the petal length or width?
In this section of the assignment, you will create several different regression models to answer some of these questions.

### 2.1 Model Definition

Your implementation should define a class for `LinearRegression` which includes at least a `fit` and `predict` method.
Additional methods can be added as you see fit.

The `fit` method should accept 2 parameters:
1. the input data
2. and the target values.

Other parameters can be added as long as they are optional.

You will use this class to train and compare 6 models.
Each model should use a different set of input features and outputs.
You can pick any combinations you would like.
For example, one model could use the petal width of setosa samples to predict the petal length.

You might also try to combine the petal length feature from all iris samples to predict petal width.
However, you should ask whether or not combining features from different types of iris would work.
The distribution of petal lengths is different between each of the three types.
A possible solution to this would be to normalize the data.
One normalization technique would be to subtract the mean sample from each of the sample in the dataset and divide by the standard deviation of the data.
You could apply this strategy to the entire training set, or normalize the data on a per-class basis.

You are encouraged to try this or other strategies.
Don't worry if the resulting model doesn't perform well, your grade is not dependent on model accuracy.

### 2.2 Training

Your models should be trained using batch gradient descent with a batch size (optional parameter) of 32.
Use mean squared error as your loss function.
For each model, train for $n = 100$ steps (optional parameter).
As each model trains, record the loss average over the batch size against the current step number.
One way to save this data is to either return an array from the fit method or save it as an internal class member that can be retrieved after training is complete.
**Plot the loss against the step number and save it.
This will go in your report.**

To observe the effects of regularization, pick one of your trained models and inspect the weights.
Train an identical model again, except this time you will add L2 regularization to the loss.
Record the difference in parameters between the regularized and non-regularized model.
**In your report, include the weight values in this comparison.**

### 2.3 Testing

For each model you created, test its performance on unseen data by evaluating the mean squared error against the test dataset that you set aside previously.
Based on these results, which input feature is most predictive of its corresponding output feature?
Create a table of results that summarized the testing accuracy of each model and put it in your report.

# Submission

Create a zip file that includes all of your code as well as your report. The TA should
be able to easily run the code to reproduce all plots and results. Include any additional
instructions, if necessary.
