# Building an Artificial Neural Network from scratch

This project is done by Javier Páez Franco for his GitHub repository, “Artificial Neural
Network (ANN)”. In this project, I discuss the process of the limitations of a single perceptron,
developing a multilayer perceptron, how it performs, and comparing its results to the Scikit-learn
solution. A detailed report was written, and is available in `ANN_Report.pdf`.

The network comprises multiple interconnected layers of neurons that are trained to classify
a 10-feature input into seven distinct classes. I discuss the use of various network architectures
and hyperparameters tuning to obtain the best performance on the test data set, which is studied
through cross-validation.

The purpose of this work was learning how neural networks work, by building one completely
from scratch, without using any libraries or frameworks (such as PyTorch or Scikit-learn), except
for NumPy (as we can really speed up calculations with this library). Additionally, more advanced
concepts such as L2 regularization or the Adam Optimization Algorithm were implemented too.
The final purpose was to test whether my work can be compared to other solutions, and more
specifically against Scikit-learn.

The data used for this project is licensed under an MIT license, so feel free to use it for your own projects.