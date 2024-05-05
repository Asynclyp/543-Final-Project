# 543-Final-Project

In this project, we implement Fast Gradient Sign Method and Projected Gradient Descent with the goal of perturbing images to cause misclassification. We incorporate our novel objective function aimed at minimizing the perturbation and maximizing misclassification rates, and run our algorithms on 16 images from the ImageNet dataset. This project was completed by Albert Ming, Yuanpeng Li, Aaron Luo, and Frank Wang of Washington University in St. Louis.

# Necessary Packages
Suggested IDE: Pycharm
Python version 3.10+

If using Pycharm for the first time, click on the <No Interpreter> on the bottom left corner -> Add New Interpreter -> Add Local Interpreter -> System Interpreter -> select installed Python. 

```plaintext
Please make sure to have the following packages installed:
- TensorFlow
- Matplotlib
- Numpy
- Scikit-Optimize
```

We recommend using pip to install these packages. You can install them by running the following command:
```plaintext
pip install tensorflow matplotlib numpy scikit-optimize
```

# How To Use
FGSM.py, PGD_BayesianOpt.py, and PGD_HillClimbing.py each corresponds to the implementation of FGSM, PDG with Bayesian Optimization, and PDG with Hill Climbing Optimization. Running each python program would generates the results detailed in our report.

For example, type  
```plaintext
python FGSM.py
```
to obtain the results of our FGSM implementation.
Similarly, type 
```plaintext
python PGD_BayesianOpt.py
```
```plaintext
python PGD_HillClimbing.py
```
to see those results as well.
