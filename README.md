# Resume-Project-List-Summary

[Linear Regression](#Predicting-Customer-Tenure-Using-Linear-Regression)

[Logistic Classification](#Logistic-Classification-Modeling-on-ISP-Customer-Churn-Data)

# Predicting Customer Tenure Using Linear-Regression

https://github.com/MaxSydow/Linear-Regression-Predicting-Customer-Tenure/blob/main/MultReg-checkpoint.ipynb

It is a common adage in the telecom industry that it costs more to acquire new customers than it does to keep existing ones. It would be worthwhile to look at which factors affect how long a customer keeps their services with an ISP, an attribute called Tenure. The Customer Churn data set contains 50 descriptive columns of numerical and categorical data which describe characteristics of customers, their services, and sentiments regarding the ISP. Some of this data may be useful in predicting Tenure. Multiple linear regression can be used to examine how columns, as explanatory variables, affect the Tenure outcome variable.

## Objectives and Goals:
Performance characteristics of such a model can be examined and used to find which explanatory variables make for better predictions. There are so many variables to start with, and surely using all 49 to predict just one would not be the best idea. Issues such as overfitting and co-linearity can hinder model performance. A good predictive model will be able to make predictions on non-existing data. Overfitting occurs when there are too many data points making the predictions overly specific to the variables that the model was created with. An entire column could result in overfitting too, and P-values for columns in a model can be used to look at which suggests changes in the predicted variable that are not associated with changes in the explanatory variables. Collinearity occurs when there is a strong linear correlation between explanatory variables. Regression coefficients and residuals can be used as quantifiable measures of a model’s accuracy. There may also be some explanatory variables that can just be eliminated using domain specific knowledge and narrowing the scope of the model. The goal, then is to use all of these techniques to end up with a model with an optimum collection of explanatory variables that gives the best predictions for Tenure.

## Method Justification and Assumptions
Summary of Assumptions The outcome of multiple linear regression is a linear equation of the form

Y = b + a1x1 + a2x2 + … + anxn,

where Y is the target, or response variable, b is an intercept, the xi’s are the explanatory variables, and the ai’s are coefficients. The equation is essentially a multi-variable analog of the common equation for a line in 2-dimensions:

y = mx + b.

Each of these elements should be numeric. More specifically the predictive variable needs to be numeric and continuous, but the explanatory variables can be discrete and numeric. The equation can be found by minimizing the differences between actual data points and a theorized best prediction or expectation. Such expected values can be expressed in the same form as the linear equation above, where E(Y) is used to denote the expected value of the response variable. Residuals are the differences between actual data points and the expected values. The regression model depends on minimizing the set of residuals.

Tool Benefits In a single variable model finding the smallest residuals can be found by hand using calculus, but the calculations become unwieldy with a larger number of variables. Programming languages come in handy here. Python has several pre-coded packages that contain useful functions for model creation and exploration. Data sets can be stored in objects like data frames and arrays and used for input and output with these functions. Jupyter Notebooks is a commonly used Python platform that allows for execution of smaller chunks of code in cells. This makes it easier to navigate the code of a project, and to trouble shoot smaller pieces of it. Each cell can be used to accomplish a specific objective of a project and present it in a clear sequential manner.

Appropriate Technique The Churn dataset contains columns with categorical data. Some of these columns might be useful in making predictions but can’t be treated mathematically. When columns consist of Yes or No values, they can easily be transformed to 1’s and 0’s. Likewise, columns with a higher number of categories can be mapped to numbers counting up to how many discrete values there are in them. Once this is done a collection of numeric explanatory variables can be stored in a data frame along with a continuous response variable. The Tenure column contains numeric continuous data, so multiple linear regression using Python can be used to create predictive models.




# Logistic Classification Modeling on ISP Customer Churn Data

https://github.com/MaxSydow/Logistic-Classification-Predicting-Customer-Churn/blob/main/LogisticRegression-checkpoint.ipynb

## About
Any company that provides a service wants to maintain their existing customer base. Churn is a commonly used term related to customers discontinuing their terms of service. For an ISP a customer who has “churned” has ceased to continue to use the services for which they have subscribed. It is more costly to attain new customers than it is to acquire new ones, so minimizing churn is a crucial aspect of maintaining profitability and providing good service. The Customer Churn data set contains a Churn column that is defined by whether a customer has ended their services within the last month, indicated by a Yes or No value.

The specific products and services offered may play an influential role in a customer’s decision to stop their subscriptions, but ultimately it is their choice. An individual’s sentiment and behavior may also influence decision making. Outside demographic influences, such as where a customer lives or how much money they make may play a role in such choices. A customers gauged perception of themselves and the ISP also seem like possibilities to sway influence. Other aspects, such as how they choose to pay their bill or number of times, they have called in may be seen as such behavioral indicators. In short, it seems worthwhile to get some deeper insight into customer-controlled aspects for marketing and customer service interaction purposes. If fields in the data set which pertain to products and services offered were set aside, which other customer-controlled aspects could be used to predict the likelihood of churn.

## Objective
A logistic predictive model can be applied to make predictions for an outcome with only 2 possible values. Beginning with a set of attributes that describe customer charateristics an initial model can be made. Model performance attributes can be examined to make improvements, which may require eliminating some predictive variables. The goal of arriving at a best model is to use it's parameters to reveal business decision insight.

## Data Goals and Assumptions
Using computational quantitative modelling can be used to aid in data driven insight. Churn is a binary valued field, and such modelling can be used to predict the likelihood or probability of a yes or no churn decision occurring. Probability can be calculated but require numerical input. There are several categorical fields in the data set that may be useful to make a prediction. If such fields could be ascribed to numerical values they might be of use.

To that end it would be interesting to explore such features which can be ranked. Consider that there are 50 states and 2 US territories included as possible values for the State column, but should one state or territory be ascribed a higher number than another? Perhaps population, GDP, number of congressional seats, date of statehood or any number of other factors could be used to devise such a ranking. To accomplish such a ranking criterion seems worthy of a completely separate line of investigation in and of itself. A similar argument could be made for the Jobs field; with several occupations involving education included is it ethical to rate one specialty of teaching above another? It seems reasonable that a simple and more objective ranking scheme should be applied to avoid tangential debate on which categories can be used in an initial multivariate predictive model exploration effort.

For the reasons outlined certain columns can be ruled out from inclusion of the model. Such columns include City, State, County, Timezone, Job, Employment, Marital and Gender. Zip, Lat and Lng columns contain numerical values, but cannot be aligned with any sort of easy to describe spectrum. Other categorical columns like Area and Contract can have a simple ranking applied to them. Rural, Suburban, and Urban Areas can be ranked in this order according to population density. A month-to-month contract is shorter than a year-long contract, which is shorter than a 2-year contract. PaperlessBilling consists of yes/no values which can be mapped to 1s and 0s, but PaymentMethod has 4 distinct values that can’t be assigned numerical values with such unambiguity. Techie is a very sentiment-oriented column but consists of yes/no values and thus will be included.

This leaves 23 categorical and numeric features to be explored as explanatory variables in the model. The numerical columns can be further subdivided into discrete and continuous.

Categorical: Area, Techie, Contract, PaperlessBilling

Discrete numerical: Children, Age, Contacts, Email, Yearly_equip_failure, items 1- 8

Continuous: Population, Tenure, Bandwidth_GB_Year, MonthlyCharge, Outage_sec_perweek, Income

Model Assumptions Transforming all predictive variables to numerical form allows the application of the logistic regression model to make predictions for the binary target variable. Linear regression attempts to fit data to an equation of the form:

Y = b + a1x1 + a2x2 + … + anxn, (1)

Where b is an intercept, the xi’s are the predictive variables, and ai’s are coefficients. If there were only one predictor y would take the shape of a line.

The shape of a non-continuous, binary variable would look a lot different. There are really only 2 levels possible: 1 for yes, and 0 for no. Instead of trying to model a 2-valued stepwise function, a more continuous interpretation can be made. The probabilities of the target being 0 or 1 would lie in a continuous range. The sigmoid function approaches 0 or 1 on either end and smoothly increases in an ‘S’ shape within the domain of most of the explanatory variable’s values, and has the form:

P(y) = 1/(1+e^(-y)), (2)

It looks like this:

image

Equation (1) can be substituted for y into equation (2), thereby transforming an otherwise linear correlation model with direct predictions of the target values into an exponential model that predicts outcome probabilities. The linear combination of explanatory variable can be solved for by taking the natural logarithm of both sides of equation (2), hence the name logistic.

ln(y/(1-y)) = b + a1x1 + a2x2 + … + anxn, (3)

This is the form an equation of fit would take to describe a logistic model.

Tool Benefits and Technique
Python has several packages that can make the computations and obtain get these equations much faster. In addition, there are other pre-coded functions that aid in determining the accuracy of the model and choose which explanatory variables are best. The sklearn library contains a vast number of such functions. Beyond just finding a good fit, the explanatory and predicted variables can be split into training and testing sets. Half of the 10,000 rows of data can be used to compute the model, and the predictions it makes can be verified against the rest. This allows for computations of True Positive (TP), True Negative (FN), False Positive (FP), and False Negative (FN) outcomes. These 4 values are typically summarized in a confusion matrix.

The 4 categories of predictions can be used to compute accuracy metrics. True Positive Rate (TPR) and False Positive Rate (FPR) use these categories. A plot of TPR vs. FPR gives an ROC (receiver operating chatacteristic) curve. The area under the curve (auc) provides a measure of how well a variable contributes the prediction; 0 being weakest to 1 being strongest. The auc can be computed as explanatory variables are added to the model in a process called forward stepwise variable selection. If too many features are used in a model the predictions on the test data may grow further away from the data, it was trained on. This would indicate overfitting, so using auc with stepwise selection can provide a way to obtain a good collection of explanatory features to keep in a final model.
