# Assignment 6

This assignment covers image classification with deep learning. You will train and evaluate three different categories of networks on the `Food101` dataset. For each model, it is recommended that you implement them using [PyTorch Lightning](http://www.pytorchlightning.ai). Feel free to adapt the code from the [demos we did in class](https://github.com/ajdillhoff/CSE6363/tree/main/deep_learning) or from any of the examples provided on [Pytorch Lightning's website](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/01-introduction-to-pytorch.html).

For all tasks, free CPU compute time and possible GPU time is available through [Google Colab](https://colab.research.google.com/).

# A Basic CNN

The first network will be a basic CNN. This network should include some number of convolutional layers followed by fully connected layers. There is no size requirement for this network nor is there a performance requirement. Train the model for up to 5 epochs. Implement some form of early stopping in case the model begins to overfit.

In your report, describe the chosen architecture and include the training loss, validation loss, and final test accuracy of the model.

# All Convolutional Net

Create an all convolutional model and train it on the `Food101` dataset. Compare the number of total parameters in this model versus the basic CNN used in the previous section. Train the model for up to 5 epochs. Use early stopping to prevent the network from overfitting (if applicable).

In your report, describe the chosen architecture and report the training loss, validation loss, and final test accuracy of the model.

# Regularization

Pick one of the models used in the previous two sections and add regularization in the form of data augmentation or dropout. Limit your training to 5 epochs on each model.

In your report, describe your choice of data augmentation and provide a clear comparison of the model with and without regularization.

# Transfer Learning

Transfer learning is an effective way to leverage features learned from another task into a new task. For this part, use a pre-trained model provided by `torchvision` and fine-tune it on the `Food101` dataset. You can refer to the class demonstration of [transfer learning](https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/transfer_learning.ipynb) to help get started.

After picking a model, train for up to 5 epochs from scratch. Evaluate the model performance on the test set. Take the same model and initialize it with pre-trained weights. Fine-tune the model for up to 5 epochs.

In your report, describe the pre-trained model you chose to use and include the fine-tuning training plots along with the final model accuracy.

# Submission

Submit all your code, best model weights, and report via Canvas in a zip file.
