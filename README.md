# COMP9417-Machine-Learning-and-Data-Mining

Machine learning is all about teaching computers learn patterns from existing data, so that it can make reliable predictions on unseen data.
The mathematical structure that is trained, evaluated, and differentiated to fit and learn from data, is called a model.
In the process of learning, it is crucial to adjust the model to minimize the error on the training data.


The first section will introduce you to machine learning approaches to the problem of numerical prediction. Following it you should be able to reproduce theoretical results, outline algorithmic techniques and describe practical applications for the topics:  

• the supervised learning task of numeric prediction  
• how linear regression solves the problem of numeric prediction  
• fitting linear regression by least squares error criterion  


Review `linalg-review (1).pdf` for assumed knowledge.


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

We want to minimize a loss function, so we use calculus to:

- Take its derivative, and  
- Find where it’s zero (analytical solution) or Use the gradient (partial derivatives) to step downhill (gradient descent)  

If we're using MSE, it's a smooth, convex, differentiable function. 
A squared function is convex; a sum of convex funcions is also convex, so the whole loss function is convex. Meaning anywhere the slope is zero is the global minimum. 
Thus, it is guaranteed that taking partial derivatives of MSE and setting it to zero will give us clues about the value of a and b at the global minimum.

If we're using MAE, the absolute value makes it non-differentiable at 0, so we can't solve it anlaytically, but we can use gradient-based methods which we will cover later.

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


### Variance

 Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value. It is the second central moment of a distribution, and the covariance of the random variable with itself. Variance is denoted by symbols: ![image](https://github.com/user-attachments/assets/98c07ebe-043e-43e6-9ec1-9d43ac36d7a7)

The other variance is a characteristic of a set of observations. When variance is calculated from observations, those observations are typically measured from a real-world system. If all possible observations of the system are present, then the calculated variance is called the population variance. Normally, however, only a subset is available, and the variance calculated from this is called the sample variance. The variance calculated from a sample is considered an estimate of the full population variance. 

### 📊 Population vs. Sample Variance

- **Population values** are theoretical and fixed.  
- **Sample values** are calculated from data and used to estimate the population.

![image](https://github.com/user-attachments/assets/76225407-d745-44ca-bd04-8ac77308f679)


> If you're **not using all data**, treat it as a sample and apply **Bessel’s correction**: divide by \( n - 1 \), not \( n \).
> Why divide by n-1 not n for sample variance ?: Check https://en.wikipedia.org/wiki/Bessel's_correction#Source_of_bias  

![image](https://github.com/user-attachments/assets/e13a5bb0-1dfe-4dcc-bc13-4b0ea1ddd148)
![image](https://github.com/user-attachments/assets/8cbd0b3b-baa0-471f-81fc-ee3f4ce93ef0)



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


Read more: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


## Univariate Linear Regression

![image](https://github.com/user-attachments/assets/4c6a986f-a084-4e7a-bad2-d5c596a3e8b1)
![image](https://github.com/user-attachments/assets/e443ce46-ef73-4e7a-b88b-748dfed1be40)

**Quick linear algebra recap:**

1. [Chain rule](https://en.wikipedia.org/wiki/Chain_rule) says, if you have a function defined as a nested function (i.e. function inside of another function, f(x) = [g(x)]^2 for example) and you're trying to differentiate it - then first take the derivative of the outer square/power (2 * g(x)), then multiply it by the derivative of the inner function (g'(x)) to get the final derivative (2*g(x)*g'(x)).
  
2. Whenever you take the derivative (or partial derivative), any part that doesn't contain the variable you're differentiating is treated as a constant — and its derivative is 0.

3. The derivative of a sum is the sum of the derivatives.
![image](https://github.com/user-attachments/assets/4fca031c-edff-4d48-bff9-432b72bb9a51)


![image](https://github.com/user-attachments/assets/cbdb7c30-acf7-424a-88f5-7663e3eadbf9)
![image](https://github.com/user-attachments/assets/06cc2c4c-3166-4401-8074-48cfde6adfd3)
![image](https://github.com/user-attachments/assets/6f703d44-efd7-4fbe-888e-eeaf5fabd894)

![image](https://github.com/user-attachments/assets/6d489754-5194-471c-a38a-097943db14ba)
![image](https://github.com/user-attachments/assets/e36ab150-6cd6-4674-a2b6-b90f05d7383b)
![image](https://github.com/user-attachments/assets/bddaa17c-d83f-4779-9e92-c17f0de101ab)
![image](https://github.com/user-attachments/assets/4115b926-0b25-42b3-bb00-fecfdc962225)


The regression line always passes through the point (x-mean, y-mean) i.e. the mean of the data, because it is mathematically the point that balances the squared errors.
 
Shifting all data left/right (x) or up/down (y) just moves the line — it doesn’t change its tilt (slope). Subtracting the mean of X from X (or subtracting the mean of Y from Y) makes the intercept zero, so the regression line passes through (0, y-mean) (or (0,0) if subtracted both the mean of Y and the mean of X), instead of (x-mean, y-mean).  

![image](https://github.com/user-attachments/assets/df496203-2a0e-4b5c-9d23-fe3642a16435)


![image](https://github.com/user-attachments/assets/2e96ab46-9d6a-438b-8206-25ecfe8c6a0a)


The sum of the residuals (the differences between the actual and predicted values) is always zero in least-squares linear regression. The regression line balances around the mean, and the residuals cancel out like positive and negative weights.


## Multivariate Linear Regression

Multivariate linear regression is just a linear regression with multiple input variables (features) (x_1, x_2, ... , x_p).

![image](https://github.com/user-attachments/assets/48b18eab-7098-47fe-af53-95797dbba6d4)


We have p features, plus 1 bias term (b_o), so we need to learn p + 1 parameters (weights) ... note that x_o is always 1, because x_o is just an extra term introduced to add the bias term b_o into the summation (which helps in converting to matrix form)

![image](https://github.com/user-attachments/assets/e8fdadb8-5cef-4375-a774-db514aa380db)

![image](https://github.com/user-attachments/assets/609cc6d1-3dba-41ca-ab9f-911287e20744)


## Vectors and Matrices

Scalar
Vector
Transpose

Vector Addition
Vector Multiplication

Systems of linear equations
![image](https://github.com/user-attachments/assets/20ff096d-2913-4ba6-9892-90113b366ac1)
![image](https://github.com/user-attachments/assets/15fae65b-effe-4262-b546-918d569120f7)
![image](https://github.com/user-attachments/assets/ca70e3e9-d1c5-4999-a5f9-7e79b9316737)

Matrices
![image](https://github.com/user-attachments/assets/970339af-3048-453b-b365-c8a36ae3fd5e)
![image](https://github.com/user-attachments/assets/6e06acc0-423d-4bfb-be70-2bb444b8ac9a)
![image](https://github.com/user-attachments/assets/5555cd46-3d70-418a-bdc9-f2402a775756)
![image](https://github.com/user-attachments/assets/4f3b6c2e-2e54-46b3-a718-c2565e7f8ea6)
![image](https://github.com/user-attachments/assets/9ad42b1c-1c65-4083-945f-00eebc799dbb)
![image](https://github.com/user-attachments/assets/8d5691bb-d1f1-4402-a519-942c24dd6912)
![image](https://github.com/user-attachments/assets/9387f2f2-da27-4b8c-8eba-3f1e36e4afc5)
![image](https://github.com/user-attachments/assets/f36c11cc-18a5-4009-a3f7-b5a849f605f8)


Going back to the Linear regression, we can now represent the solution in terms of Matrices:
![image](https://github.com/user-attachments/assets/3a4af207-5699-45ba-a141-551cbae90485)

![image](https://github.com/user-attachments/assets/53e3fbe8-cdf3-4712-a0bb-be76b992d298)


---

This section will introduce you to machine learning approaches to the problem of numerical prediction. Following it you should be able to reproduce theoretical results, outline algorithmic techniques and describe practical applications for the topics:
• non-linear regression via linear-in-the-parameters models  
• gradient descent to estimate parameters for regression  


Bias-Variance Decomposition

### Var(X) in statistics vs probability
![image](https://github.com/user-attachments/assets/40aa1d66-9ff7-4bfb-b075-f2d1d298dbdd)
![image](https://github.com/user-attachments/assets/c55ac88b-ca19-4f35-9544-9e2b65b47839)

### Linearity of Expectation
![image](https://github.com/user-attachments/assets/a19e775a-7373-4bc4-ac9f-a782f9ecbb9d)

### Binomial Distribution
![image](https://github.com/user-attachments/assets/d513019f-0dbb-4248-81b6-dd33ccbe8f94)

### MLE (Maximum Likelihood Estimate)
https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

### Continuous Distribution

Bias in Multivariate Linear Regression

### Regularisation

### Optimization by gradient descent






