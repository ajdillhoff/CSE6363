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
