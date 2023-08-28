# Assignment 3

Topics: Decision Trees, Random Forests, and Boosting

# Decision Trees

The first step is to implement a `DecisionTree` class with a `fit` and `predict` method.
This class will be used as your base classifier and further used with random forests and boosting.
It should accept the following input parameters when creating an object:
- `criterion` - Either misclassification rate, Gini impurity, or entropy.
- `max_depth` - The maximum depth the tree should grow.
- `min_samples_split` - The minimum number of samples required to split.
- `min_samples_leaf` - The minimum number of samples required for a leaf node.

## The `fit` method

The first method to implement is `fit`, which will train a single decision tree.
This method take two parameters:
1. `X` - The data of size `(n_samples, n_features)`.
2. `y` - The labels of size `(n_samples)`.

For reference to how this method should be implemented, refer to Chapter 18.1 from Kevin P. Murphy's [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html).

## The `predict` method

After the model is `fit` to the data, the `predict` method will perform inference on new samples. In this assignment, you only need to consider binary classification.

# Random Forests

Random Forests were formalized in 2001 by [Leo Breiman][1]. There are many approaches described in the original paper as well as modifications in subsequent work. For this assignment, you will implement random forests using two techniques.
1. Sample $n$ samples with replacement
2. Select a random subset of features when splitting

Create a class called `RandomForest` that takes as input a classifier object, an integer `num_trees` specifying the number of trees in the model, and the minimum number `min_features` to consider when selecting a subset of features. Fitting the model and making predictions will largely use your implementation of decisions trees with only a few minor changes.

[1]: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf "Random Forests"

## The `predict` method

To predict a sample using a random forest, loop through each tree in the forest and call its `predict` method. Return the prediction with the most votes. For example, if you trained 10 trees in your forest and 6 of them predict class 1, then your final output will be class 1.

## The `fit` method

For this assignment, fitting a random forest will use two techniques as described below.  Each of these should be applied when training each tree. For example, tree 1 will sample a different subset of the data using a random selection of features. Tree 2 would use its own random subset and selections.

### Sampling with Replacement

Given a dataset of $n$ samples, random select a subset of data **with replacement**. For example, if your original dataset looks like this: `[1 2 3 4 5]`, then a random selection of 5 samples might produce `[1 1 2 4 5]`.

### Selecting Features

For each tree, random select a number of features in range [`min_features`, `num_features`], where `min_features` is the hyperparameter supplied in your constructor and `num_features` is the total number of features in the original training data. You should verify that `min_features` is less than or equal to `num_features`.

For example, if you have 10 features in your dataset and `min_features` is 6, then you might generate a list of indices `[0, 1, 3, 6, 7, 8, 9]` specifying which feature columns to select when training this particular tree. A different list would be generated for each tree in the forest.

# Boosting

The second classifier you will implement in this assignment is a variation of boosting called AdaBoost. The details of this algorithm are described in both the [lecture notes][2] as well as section 18.5.3 of [Murphy][3].

Create a class called `AdaBoost` which takes the following input parameters:
- `weak_learner` - The classifier used as a weak learner. For this assignment, this will be an object of your `DecisionTree` class.
- `num_learners` - The maximum number of learners to use when fitting the ensemble. If a perfect fit is achieved before reaching this number, the `predict` method should stop early.
- `learning_rate` - The weight applied to each weak learner per iteration.

[2]: https://dillhoffaj.utasites.cloud/posts/boosting/ "Boosting"
[3]: https://probml.github.io/pml-book/book1.html "Probabilistic Machine Learning: An Introduction"

## The `predict` method

This function should produce a class prediction for an input sample `x` defined as

$$
f(\mathbf{x}) = \text{sgn}\Big[\sum_{m=1}^M \alpha_m F_m(\mathbf{x})\Big].
$$

## The `fit` method

Constructs a boosted classifier given a training set $(X, y)$, where $X \in \mathbb{R}^{n \times d}$ are $n$ samples with $d$ features and $y$ are the class labels where each $y_i \in \{-1, +1\}$.

# The Titanic Dataset

To test your implementations of the above algorithms, we will use the Titanic dataset. This dataset features information on passengers of the Titanic and whether or not they survived the fateful wreck. The raw dataset is not formatted for use as input to any of these classifiers, so some data preparation and analysis is required.

To this end, reference the popular notebook on Kaggle ["Introduction to Decision Trees (Titanic dataset)"](https://www.kaggle.com/code/dmilla/introduction-to-decision-trees-titanic-dataset). The data is also available from there as well. You can use the same pre-processing steps as shown in the article, or you can vary it to select a different set of features.

## Evaluation

To evaluate the performance of each of the 3 models for this assignment, train them on the Titanic dataset and evaluate their final classification performance on a test set. In your code or notebook, produce a clear table of results.