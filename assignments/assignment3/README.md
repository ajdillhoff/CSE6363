# Non-Linear SVM

The code given at [https://github.com/ajdillhoff/CSE6363/blob/main/svm/smo.ipynb]() implements a linear SVM.
Adapt this to support a non-linear SVM.

Modify the `SVM` class given so that if the input for `kernel` is `poly`, the SVM will use a polynomial kernel.
To test this, use `sklearn.datasets.make_circles` to generate a non-linear dataset and fit your SVM to it.
It should be able to correctly separate the two classes.

An example of how to generate the dataset is given [https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py](here).
Make sure to set aside 10% of the samples as a testing set.

Compare your implementation with `sklearn.svm.SVC` using both a linear kernel and polynomial kernel.
**Your implementation is not expected to perform better, but should behave similarly.**

In your code, print the accuracy result of the test set and visualize the output using `mlxtend.plot_decision_regions`.

# Multi-class SVM

Add multi-class support to your implementation following a One-versus-All approach.
Given $K$ classes, you will need to train $K$ SVMs to classify one class versus all others combined.

When implementing this, create another python class named `MultiSVM` which internally represents the individual binary SVMs.
The prediction function should classify the sample according to which binary classifier gives the largest score.

Compare your implementation with `sklearn.svm.SVC` using both a linear kernel and polynomial kernel.
Use the Iris dataset provided by `sklearn.datasets.load_iris`.
**Your implementation is not expected to perform better, but should behave similarly.**

# Submission

Create a zip file that includes all of your code as well as your report. The TA should be able to easily run the code to reproduce all plots and results. Include any additional instructions, if necessary.
