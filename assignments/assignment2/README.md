For this assignment, we will be using the CIFAR-10 dataset which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset is available from many places online, and a direct download link is available in Canvas. You can also find more information about the dataset at https://www.cs.toronto.edu/~kriz/cifar.html.

# Running the Benchmark Models

Although you will not be graded based on your model's output, it is a good idea to reference the benchmark models to ensure that your implementation is correct. The benchmark models are available in the `assignment2/benchmarks` directory. You can run the benchmark models as follows:

```bash
python assignment2/benchmark.py
```

This may take a few minutes on your machine to run.

# Downloading and Preprocessing the Data

After downloading the data, extract the batch files to the `cifar10` directory that is already in this repository. A function is provided in `assignment2/utils.py` to extract and load the data to `numpy` arrays. You can use this function to load the data as follows:

```python
from utils import load_and_prepare_data

X_train, y_train, X_test, y_test = load_and_prepare_data()
```

# Linear Discriminant Analysis

Complete the class definition of `LDA` in `assignment2/models.py`. The `fit` method should estimate the class means and the shared covariance matrix. The `predict` method should use these estimates to make predictions.

As we saw in class, the class means can be estimated as the mean of each class in the training data. The shared covariance matrix can be estimated as the weighted sum of the covariance matrices of each class, where the weight is the number of samples in each class.

**Class Means**
$$
\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} X_i
$$

where $N_k$ is the number of samples in class $k$, and $X_i$ is the $i^{\text{th}}$ sample in class $k$.

**Covariance Matrix**
$$
\Sigma = \frac{1}{N} \sum_{i=1}^N (X_i - \mu)(X_i - \mu)^T
$$

where $N$ is the number of samples, $X_i$ is the $i^{\text{th}}$ sample, and $\mu$ is the mean of the samples.

In `assignment2/lda_main.py`, use your implementation of LDA to fit the model to the training data and make predictions on the test data. Report the accuracy of your model on the test data. You should evaluate this on both RGB and Grayscale versions of the dataset.

# Quadratic Discriminant Analysis

Complete the class definition of `QDA` in `assignment2/models.py`. The `fit` method should estimate the class means and the class covariance matrices. The `predict` method should use these estimates to make predictions.

**Class Means**
$$
\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} X_i
$$

where $N_k$ is the number of samples in class $k$, and $X_i$ is the $i^{\text{th}}$ sample in class $k$.

**Covariance Matrix**
$$
\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{N_k} (X_i - \mu_k)(X_i - \mu_k)^T
$$

where $N_k$ is the number of samples in class $k$, $X_i$ is the $i^{\text{th}}$ sample in class $k$, and $\mu_k$ is the mean of the samples in class $k$.

In `assignment2/qda_main.py`, use your implementation of QDA to fit the model to the training data and make predictions on the test data. Report the accuracy of your model on the test data. You should evaluate this on both RGB and Grayscale versions of the dataset.

# Gaussian Naive Bayes

Complete the class definition of `GaussianNaiveBayes` in `assignment2/models.py`. The `fit` method should estimate the class means and the class variances. The `predict` method should use these estimates to make predictions.

**Class Priors**

The class priors follow the same formula as in LDA and QDA:

$$
\pi_k = \frac{N_k}{N}
$$

where $N_k$ is the number of samples in class $k$, and $N$ is the total number of samples.

**Class Conditional Distributions**

For Gaussian Naive Bayes, the class conditional density follows a Gaussian distribution:

$$
p(x|y=k) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right)
$$

where $\mu_k$ is the mean of the samples in class $k$, and $\sigma_k^2$ is the variance of the samples in class $k$.

In `assignment2/gnb_main.py`, use your implementation of Gaussian Naive Bayes to fit the model to the training data and make predictions on the test data. Report the accuracy of your model on the test data. You should evaluate this on both RGB and Grayscale versions of the dataset.

# Questions

The following questions should be answered in a separate file. It is recommended that you typeset the answers in LaTeX, but this is not required. You may also write your answers on paper and scan them as a PDF.

1. State the maximum likelihood estimates for the parameters of Gaussian Naive Bayes. Do these estimates change if we use grayscale images instead of RGB images? Explain why or why not.

2. The accuracy of QDA using RGB images was lower than that of grayscale images. What assumptions does QDA make that might cause this difference in performance?

3. Both LDA and Gaussian Naive Bayes saw reduced test accuracy on grayscale images compared to RGB images. Why might this be the case (is it the data, the model, or something else)?

4. How many parameters are estimated for each model and each image type (RGB and grayscale)?