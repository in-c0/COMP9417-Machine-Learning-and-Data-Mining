# COMP9417-Machine-Learning-and-Data-Mining

Machine learning is all about teaching computers learn patterns from existing data, so that it can make reliable predictions on unseen data.
The mathematical structure that is trained, evaluated, and differentiated to fit and learn from data, is called a model.
In the process of learning, it is crucial to adjust the model to minimize the error on the training data.


**General Structure of Machine Learning Problems**
Every supervised learning problem in machine learning can typically be broken down into three main steps:
 
1. Select a model and an error (loss) function
2. Fit the model to the training data by minimizing the loss
3. Use the trained model to predict for new, unseen data


**Step 1: Selecting a Model**

The first step, selecting a model, is about defining a mathematical formula or function that maps inputs to outputs.

For example, if we want to predict house prices (y), with the input features (x_1, x_2, x_3, ...) like:
- number of bedrooms  
- square footage  
- location  
...

A simple **linear regression** model captures the relationship between these input features and the output (house price) as such:
`price = w_1 * num_bed  +  w_2 * sqr_ft + w_3 * location + b`
... where w_1, w_2, w_3 is the weight controlling how much influence the corresponding input feature has on the final prediction,
     and b is the bias term (intercept) setting the base output, i.e. what the model predicts if all inputs are zero.

This model assumes a linear relationship between input features and output, and predicts continuous values. It's simple and fast but may not capture complex relationships. Other linear models include logistic regression for binary classification, and lasso regression for linear regularization.

Later on, we'll learn about neural networks that stack layers of interconnected nodes (i.e. artificial neurons) to model complex non-linear patterns.



**Step 2: Fitting a Model**

To train* a model, we measure how wrong its predictions are on the training data and try to reduce that error. This process is called **Empirical Risk Minimization (ERM)**.

First, we define a loss function to measure prediction error. Then, we calculate the average loss on the training data — this is called **empirical risk**. (*Note the definition: _Training_ a model means adjusting it to minimize this average loss. **empirical risk** simply means the **average loss on the training data**.)

We define a **loss function** $\ell(y_i, \hat{y}_i)$ that measures how wrong the model’s prediction $\hat{y}_i$ is compared to the true label $y_i$.

The **empirical risk** is calculated as:

$$
\mathcal{R}_{\text{emp}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, \hat{y}_i)
$$

Where:
- $n$ is the number of training examples  
- $y_i$ is the true output (label) for the $i$-th example  
- $\hat{y}_i$ is the predicted output from the model  
- $\theta$ represents the model parameters

The goal of training is to **find the parameters** $\theta$ that **minimize** this empirical risk.

This process is called **Empirical Risk Minimization (ERM)** — and it's at the heart of how most supervised learning algorithms work.


![image](https://github.com/user-attachments/assets/cae34d96-07c0-4974-bbdb-6f1b7db1c545)

Read more: https://en.wikipedia.org/wiki/Empirical_risk_minimization


### Covariance and Correlation
![image](https://github.com/user-attachments/assets/3f7ab23f-2394-49a5-b685-32d8987b780e)

### Univariate Linear Regression
### Multivariate Regression
### Vectors and Matrices



The second step, picking an error function (aka Loss function), is about choosing how we want the model to behave when it makes mistakes. 

You know that the chosen error function is good if minimizing it leads to better performance (i.e. fewer wrong predictions, higher accuracy, lower test error). Choosing the best error function is often a matter of trial and error.  
For example, MSE (Mean Squared Error) is sensitive to big errors. MAE (Mean Absolute Error) treats all errors equally. and Cross-entropy encourages confident classification, but may overfit if data is noisy.



