# COMP9417-Machine-Learning-and-Data-Mining

Machine learning is all about teaching computers learn patterns from existing data, so that it can make reliable predictions on unseen data.
The mathematical structure that is trained, evaluated, and differentiated to fit and learn from data, is called a model.
In the process of learning, it is crucial to adjust the model to minimize the error on the training data.


## General Structure of Machine Learning Problems

Every supervised learning problem in machine learning can typically be broken down into three main steps:
 
1. Select a model and an error (loss) function
2. Fit the model to the training data by minimizing the loss
3. Use the trained model to predict for new, unseen data


### Step 1: Selecting a Model

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


Selecting an error function (aka Loss function), on the other hand, is about choosing how we want the model to behave when it makes mistakes. 

You know that the chosen error function is good if minimizing it leads to better performance (i.e. fewer wrong predictions, higher accuracy, lower test error). Choosing the best error function is often a matter of trial and error.  
For example, MSE (Mean Squared Error) is sensitive to big errors. MAE (Mean Absolute Error) treats all errors equally. and Cross-entropy encourages confident classification, but may overfit if data is noisy.



### Step 2: Fitting a Model

To train* a model, we measure how wrong its predictions are on the training data and try to reduce that error. This process is called **Empirical Risk Minimization (ERM)**.

First, we define a loss function to measure prediction error. Then, we calculate the average loss on the training data — this is called **empirical risk**. (*Note the definition: _Training_ a model means adjusting it to minimize this average loss. **empirical risk** simply means the **average loss on the training data**.)

We define a **loss function** $\ell(y_i, \hat{y}_i)$ that measures how wrong the model’s prediction $\hat{y}_i$ is compared to the true label $y_i$.

The **empirical risk** is calculated as:

![image](https://github.com/user-attachments/assets/596cb984-ccc5-495f-b780-9c2e928dbc84)

Where:
- $n$ is the number of training examples  
- $y_i$ is the true output (label) for the $i$-th example  
- $\hat{y}_i$ is the predicted output from the model  
- $\theta$ represents the model parameters

The goal of training is to **find the parameters** $\theta$ that **minimize** this empirical risk.

This process is called **Empirical Risk Minimization (ERM)** — and it's at the heart of how most supervised learning algorithms work.


![image](https://github.com/user-attachments/assets/cae34d96-07c0-4974-bbdb-6f1b7db1c545)

Read more: https://en.wikipedia.org/wiki/Empirical_risk_minimization

### Step 3: Making a prediction on unseen data


## Covariance and Correlation


### Covariance

Covariance measures how two variables change together.

- If both variables increase together, the covariance is positive.  
- If one increases while the other decreases, the covariance is negative.  
- If the variables do not show any consistent pattern together, the covariance is close to zero.  

![image](https://github.com/user-attachments/assets/b33471e9-2d95-4e81-9df5-12c186545331)


Correlation is the standardized version of covariance so it's always between -1 and +1.

### Correlation

The **correlation coefficient** is a number between **−1** and **+1** that measures how strongly two variables \(X\) and \(Y\) are related.

- **+1**: Strong positive linear association  
- **0**: No clear linear association (high scatter)  
- **−1**: Strong negative linear association (inverse)

> Interpretation:
- High \(X\) with high \(Y\), low \(X\) with low \(Y\): positive correlation  
- High \(X\) with low \(Y\): negative correlation  
- Mixed pattern: weak or no correlation

⚠️ **Note**: Correlation is meaningful **only if the relationship is linear**. It doesn't work well for **curved (non-linear)** associations.
“Correlation does not imply causation”. You cannot use correlation to infer that X causes Y , or the other way around

![image](https://github.com/user-attachments/assets/8cbd0b3b-baa0-471f-81fc-ee3f4ce93ef0)

> Why divide by n-1 not n for sample variance ?: Check https://en.wikipedia.org/wiki/Bessel's_correction#Source_of_bias  

Read more: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


## Univariate Linear Regression
## Multivariate Regression
## Vectors and Matrices

