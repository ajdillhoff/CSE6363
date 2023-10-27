# Assignment 2: Support Vector Machines

## (20 points) The Negative $\eta$ Case

In the [original paper for SMO](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/), the second derivative of the objective function is evaluated in order to update the Lagrange multipliers. Platt states that under normal circumstances, this value will be positive definite. A negative value occurs if the chosen kernel $K$ violates Mercer's theorem, which states:

> A symmetric function $K(x, y)$ defines a valid kernel if and only if for any finite set of vectors $x_1, x_2, ..., x_n$, the corresponding kernel matrix is positive semidefinite.

If this happens, positive progress can still be made by evaluating the objective function at each end of the line segement. From the paper, the given pseudocode is:

```
Lobj = objective function at a2=L
Hobj = objective function at a2=H
if (Lobj < Hobj-eps)
    a2 = L
else if (Lobj > Hobj+eps)
    a2 = H
else 
    a2 = alph2
```

Equation (19) from the paper expresses how to compute the objective function at each end:

$$
\begin{align*}
    f_1 &= y_1(E_1 + b) - \alpha_1 K_{11} - \alpha_2 K_{12} \\
    f_2 &= y_2(E_2 + b) - s \alpha_1 K_{12} - \alpha_2 K_{22} \\
    L_1 &= \alpha_1 + s(\alpha_2 - \alpha_1) \\
    H_1 &= \alpha_1 + s(\alpha_2 - \alpha_1) \\
    \Psi_{L} &= L_1 f_1 + L f_2 + \frac{1}{2} L_1^2 K_{11} + \frac{1}{2} L^2 K_{22} + s L L_1 K_{12} \\
    \Psi_{H} &= H_1 f_1 + H f_2 + \frac{1}{2} H_1^2 K_{11} + \frac{1}{2} H^2 K_{22} + s H H_1 K_{12}
\end{align*}
$$

where $s = y_1 y_2$ and $L$ and $H$ are compute as described in equations (13) and (14), respectively.

### Tasks
1. Implement this case in the provided SMO algorithm available [here](https://github.com/ajdillhoff/CSE6363/blob/main/svm/smo.ipynb).
2. If $\eta$ is negative, this implies that the kernel matrix is not positive semidefinite. This can occur if more than one training example has the same input vector. Provide an example of a kernel function that would cause this to happen. In other words, given two input vectors of your choosing, show the resulting kernel matrix and show that it is not positive semidefinite.

## (40 points) Non-linear SVM

Adap the [code in the course repository](https://github.com/ajdillhoff/CSE6363/blob/main/svm/smo.ipynb) to support non-linear SVMs. Modify the `SVM` class so that if the input for `kernel` is `poly`, the SVM will use a polynomial kernel. To testthis, use `sklearn.datasets.make_circles` to generate a non-linear dataset and fit your SVM to it. It should be able to correctly separate the two classes.

An example of how to generate the dataset is given [here](https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py). Make sure to set aside 10% of the data for testing.

Compare your implementation with `sklearn.svm.SVC` using both a linear kernel and polynomial kernel. **Your implementation is not expected to perform better, but should behave similarly.**

In your code, print the accuracy result of the test set and visualize the output using `plot_decision_regions`.

## (40 points) Multi-class SVM

Add multi-class support to your implementation following a One-vs-Rest (OvR) approach. Given $K$ classes, you will need to train $K$ SVMs to classify one class versus all others combined.

When implementing this, create another python class named `MultiSVM` which internally represents the individual binary SVMs. The prediction function should classify the sample according to which binary classifier gives the largest score.

Compare your implementation with `sklearn.svm.SVC` using both a linear kernel and polynomial kernel. Use the Iris dataset provided by `sklearn.datasets.load_iris`. **Your implementation is not expected to perform better, but should behave similarly.**

# Submission

Create a zip file that includes all of your code and a PDF or similar document with your answers to the question in the first section. We should be able to run your code and reproduce the results easily. Include any additional instructions, if necessary.