# ECE368
Probabilistic Reasoning | Machine Learning

<br>
<h3>Lab 1 Classification with Multinomial and Gaussian Models</h3>

The first part of this lab involved designing a **Naives Bayes classifier** to implement a spam-filtration system, flagging emails as "spam" or "ham (not-spam)" based on whether and how many times a certain list of words appeared in the email. Using a **multinomial distribution** for the feature vector and a **maximum likelihood estimate** with **Laplace smoothing** for the probabilities, expressions for the likelihood function and conditional and posterior probabilities were derived and coded from scratch in Python. While the final model had [...]% and [...]% as its **false negative** (Type 1 error) **false positive** (Type 2 error) rates, respectively, the lab also introduced the concept of adding an extra term to the decision rule to control the trade-off between the two types of errors. 

The second part of this lab also involved binary classification with one key difference from the first part being the involvement of feature vectors containing real values. This required the use of a **Gaussian Vector Model**, and through the use of **linear discriminant analysis (LDA)** and **quadratic discriminant analysis (QDA)**, height and weight data was used to predict whether an individual belonged to the "male" or "female" class. The classifications were then visualized on a 2D-plot showing the respective linear and quadratic boundaries.

<h3>Lab 2 Bayesian Linear Regression</h3>

This lab uses **Bayesian regression** to **fit a linear model** of the form ![img](http://latex.codecogs.com/svg.latex?z%3Dax%2Bw) where ![img](http://latex.codecogs.com/svg.latex?w) is a Gaussian noise with a known mean and variance. The parameter ![img](http://latex.codecogs.com/svg.latex?a) is modelled as a **zero mean isotropic Gaussian random vector** and estimated by deriving the posterior distribution. The effects of varying the size of the training dataset was also studied by first, limiting the number of samples to 1, 5 and then 100.

<h3>Lab 3 Inference on Hidden Markov Models</h3>

This lab utilizes a hidden Markov model to predict a rover’s movement over time across a region mapped as a grid. The rover's actions are quite predictable and depends on the rover's previous action as well as its current location. Algorithms, such as the **forward-backward algorithm** and the **Viterbi algorithm** are used to predict the most probable position of the robot at each time step and the most probable sequence of positions over time, respectively.
