# Udacity Machine Learning Course: Self Driving Car 
This project is from the course for Machine Learning provided by Udacity. It consists of building a Gaussian Naive Bayesian classifier that tells whether the car should go slow or fast depending on certain defined features.
## Getting Started
### Prerequisites
To use this code, you should have an installed Python 2.7 or 3.x
I personnally installed the Anaconda 2.7 and I am using the provided Spyder IDE

### Theoritical Background
The data that is used for the classification is divided into features and Output

#### Features:
The vector of features X = [x1, x2] is composed of 2 elements where:
- x1 is the steepness of the terrain (flat, hill ...)
- x2 is the ruggness of the terrain (Smooth, Bumpy, Very Bumpy ...)

#### Output
The output y , we can call it label, tells if the speed is fast or slow
y is either 0 or 1.
- y = 0 Fast
- y = 1 Slow



