# Perceptron-Learner

[Source](learner.py)

## Method

Each class gets its own perceptron, with the goal of distinguishing this class from the others. Documents are encoded as sparse, one-hot vectors.
The perceptron activation is the sigmoid function, and it is trained with stochastic gradient descent no binary cross-entropy loss as derived in class. In this way, it is similar to
logistic regression only without a bias term.

Right now, it only does one pass of the train dataset and stops. It is possible to extend this, but one pass was sufficient to show that the model found some signal. It isn't
particularly great, but I think this is mostly due to a somewhat naive one-hot word representation for the documents. When the gradient is flipped (negative -> positive), the
accuracies for each class go to zero (since we are maximizing loss instead of minimizing), so this provides some more evidence that this model is learning.

## Results
[Confusion Matrix](confusion.txt)

[Per-class accuracy](per-class-acc.txt)