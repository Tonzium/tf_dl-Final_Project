# tf_dl-Final_Project

This project involves a multiclass classification task to predict tree types from forestry-related data.
The dataset, which is highly imbalanced, goes through a data preparation process that includes initial exploration,
visualization, and resampling (using oversampling and undersampling) to address the class imbalance.

Following data splitting into training and test sets, further preprocessing includes resampling, standardization,
and one-hot encoding of the labels.

A TensorFlow-Keras neural network is implemented as the predictive model.
It's structured with several dense layers, dropout for regularization, and a softmax layer for multiclass prediction.
Early stopping based on validation loss is employed for training control.

The model's training and performance are evaluated using plots of accuracy and loss over epochs, along with a printed classification report.
Lastly, the model is used to predict classes for the first 10 rows of the dataset, and the results are compared to the actual values to validate the practical application of the model.
