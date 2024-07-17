# Assignment 4

This assignment covers image classification with deep learning. You will train and evaluate three different categories of networks on the `Imagenette` dataset. For each model, it is recommended that you implement them using [PyTorch Lightning](http://www.pytorchlightning.ai). Feel free to adapt the code from the [demos we did in class](https://github.com/ajdillhoff/CSE6363/tree/main/deep_learning). A getting started codebase has been provided in this repository.

For all tasks, free CPU compute time and possible GPU time is available through [Google Colab](https://colab.research.google.com/).

# A Basic CNN

The first network will be a basic CNN. This network should include some number of convolutional layers followed by fully connected layers. There is no size requirement for this network nor is there a performance requirement. Train the model until convergence. Implement some form of early stopping in case the model begins to overfit.

In your report, describe the chosen architecture and include the training loss, validation loss, and final test accuracy of the model.

# ResNet 18

The second network will be a ResNet 18. This network is a popular choice for image classification tasks. Train the model until convergence. Implement some form of early stopping in case the model begins to overfit.

In your report, describe the chosen architecture and include the training loss, validation loss, and final test accuracy of the model.

# Regularization

Pick one of the models used in the previous two sections and add regularization in the form of data augmentation or dropout. Train the model until convergence.

In your report, describe your choice of data augmentation and provide a clear comparison of the model with and without regularization.

# Transfer Learning

Transfer learning is an effective way to leverage features learned from another task into a new task. For this part, use a model that was trained on the `Imagenette` dataset and fine-tune it using the CIFAR10 dataset. You can refer to the class demonstration of [transfer learning](https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/transfer_learning.ipynb) to help get started.

Using a model from a previous run, re-train it from scratch on the CIFAR10 dataset. Take the same model and initialize it with pre-trained weights from the Imagenette dataset. With the pre-trained model, fine-tune it on the CIFAR10 dataset.

In your report, describe the pre-trained model you chose to use and include the fine-tuning training plots along with the final model accuracy.

# Submission

Submit all your code, best model weights, and report via Canvas in a zip file.
