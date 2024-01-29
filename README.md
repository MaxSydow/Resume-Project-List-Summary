# Resume-Project-List-Summary

[Linear Regression](#Predicting-Customer-Tenure-Using-Linear-Regression)

[Logistic Classification](#Logistic-Classification-Modeling-on-ISP-Customer-Churn-Data)

[Logistic Classification 2](#Towards-Automated-and-Customized-Loan-Approval-Criteria-Using-Segmented-Logistic-Classification-Models)

[Principle Component Analysis (PCA)](#PCA-Variance-Thresholds-and-Dimensional-Reduction)

[K-Means Clustering](#Grouping-Customer-Characteristics-by-Bandwidth-Usage-With-K-Means-Clustering)

[K-Nearest Neighbors (KNN) Classification](#Optimizing-K-Nearest-Neighbors-Classifier-to-Predict-Customer-Churn)

[Random Forest Regression](#Predicting-Bandwidth-Usage-With-Random-Forest-Regression)

[Market Basket Analysis](#Market-Basket-Analysis-and-Product-Sales-Suggestions)



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

[Back to top](#Resume-Project-List-Summary)




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

[Back to top](#Resume-Project-List-Summary)

# Towards Automated and Customized Loan Approval Criteria Using Segmented Logistic Classification Models

## Introduction, Research Question and Hypothesis.
Credit score, income, previous debt amount and other numerically continuous attributes are typically used when determining the amount to approve for a loan. Instead of using a standard criterion in which a hierarchy of factors are applied for all loan approval decisions it may be more accurate to tailor differing orders of such factors based on certain groupings of other borrower traits. For example, years of credit history may be the most influential indicator of whether a loan will be repaid on time for a homeowner who has been at their current job for over 10 years. Income may be the biggest driver for someone who rents and is seeking debt consolidation. Establishing more nuanced approval requirements may help lenders provide better individualized service to their customers and provide loan officers with better means of determining approval.

Almost any data set that has at least one continuous, one binary and one categorical column can be subsetted according to the unique values in the categorical column. Individual logistic classigication (logit) models can be fitted for each subset. If the models accuracy is reasonably good then the slope of the coefficient for the predictor variable gives an indication of strength of influence. When separate multiple logistic classification models are fitted to grouped subsets of a larger data set by unique values in categorical columns, will there be more than 1 statistically significant predictor variable amongst more than 1 categorical column so that the slope coeffiecents of log-transformed equations of fit can be used to summarize at least one difference in variable impact across segments?

A hypothesis can be constructed from this question. Amongst several multiple logistic classification models fitted from this data grouped according to unique values in categorical columns, at least 2 models can be found with at least 2 predictor variables that have different coefficients, and statistically significant predictive accuracy as measured using AUC scores in recursive feature elimination.

Accuracy could be used instead of AUC scores in the RFE process to get the best fitting models. (Brownlee, 2020). ROC_AUC (reciever operating characteristic area under curve, or AUC for short) scores are measured from the ratio of true positive to false negative predictions and so give an additional indication of how a model compares to random guessing. Accuracy is only measured by comparing predictions from a trained model to test data, regardless of how randomly distributed the test data may actually be. Nevertheless, accuracy will be used when examining the results and implications of the models obtained.

## Data Collection and Summary
A Bank Loan Status Data Set was found on kaggle, with the objective of having participants predict future loan status using classification models. The site posting did not indicate if this is real or mock data, but the kinds of columns included seem like realistic attributes that a bank or lender would track. It would not be too much of a stretch of the imagination to see how the methods of analysis used in this treatment could be applied to real data. The csv data sets were downloadable as test and train sets with 100,000 rows in train, and 10,000 rows in test. They will be combined and split using a different proportion later. This data was already mostly prepared and fairly easy to load and prepare. Real data in an organization may be stored in a database and require some SQL or other form of ETL method to assemble such a table.

[Back to top](#Resume-Project-List-Summary)

# PCA Variance Thresholds and Dimensional Reduction

## Investigative Question and Goal
Without customers an internet service provider (ISP) could not stay in business. Like many other businesses it is important to retain existing customers, especially in areas where competing ISPs are an option. Churn can be described as customer turnover, while tenure refers to how long a customer keeps their services.

Thoughts and perceptions can influence how long a customer stays, or even whether or not they leave. The Customer Churn data set contains 8 fields that describe customer sentiment. This set of columns consists of 8 prompts for customers to rate in terms of importance, and are labeled item1 through item8. The responses in these columns include numerical data. A response of 1 indicates most important, with 8 on the other end as least important. The items themselves are:

• item1: Timely response

• item2: Timely fixes

• item3: Timely replacements

• item4: Reliability

• item5: Options

• item6: Respectful response

• item7: Courteous exchange

• item8: Evidence of active listening

Predictive models can shed some light on the likelihood of churn, and forcast trends with tenure. Churn is a binary value, so a classification model would be appropriate. Tenure is continuous, and a regression model would be suited to make predictions. It might be worthwhile to find out if 8 prompts are actually necessary or even that useful when used in such models. Some people probably don’t mind responding to this many questions, but it may be a deterrent to others. Could it be possible to engage more customers if there were less aspects to rate? If so, which of these prompts can be justified in being removed, or perhaps consolidated? Principal component analysis (PCA) provides a mathematical and algorithmic means of grouping seemingly independent sets of values by variability. If certain items vary in a similar way this would indicate redundancy. Why have 2 or more items that are only going to give you the same trends?

## Method Justification and Assumptions
The item data frame is ordered by a unique indexed column, and contains no missing values or outliers. Essentially, we want to transform the way these columns are viewed. Orthogonality and linear independence are key concepts in linear algebra. In a plain 2-dimensional Euclidean space a line can be represented as a vector, and 2 lines are said to be orthogonal if they are perpendicular to each other. In this case the vectors are also linearly independent. (Anton, Ch. 5.3) Looking at 2 of the item columns as vectors, we can say that their respective values are independent of each other if they are orthogonal. In order to look at all 8 item columns, or any number of vectorizable columns like this a linear equation consisting of transformed vectors each having a multiplying coefficient can be calculated. (Anton, Ch. 7.2) The transformed vectors are called eigenvectors while the coefficients are called eigenvalues. In the context of PCA the eigenvectors are the principle components. The magnitude of the eigenvalues can be viewed as a weight for variability amongst the components. That is to say the larger the eigenvalue, the greater the variability of it’s corresponding principal component. Furthermore this allows us to look at the strength of variability of each original vector relative to it’s orthogonal transformation. This means we can quantitatively tell which item aligns the strongest with which principle component. Such a task would be mathematically arduous to perform by hand, but Python contains libraries that allow such analysis to be performed quickly.

Python's sklearn.decomposition package allows for fitting of a PCA model to original data. It also provides easy to use functions to compute component variances in an similar way that eigenvalues are computed. This allows for the use of these variances when reducing dimensions by componenets. In general the ranges of all original variables may not be the same. This can result in skewed variances and misalignment to principal components. For this reason it is best to scale each variable before fitting the PCA model. Sklearn has a standardscaler function that will transform each value so that the column means are 0 and standard deviation is 1. So, the assumptions of the data to be used for PCA are that each column be continuous, numeric and scaled. It does not make any predictions or clustering, but collapses the dimensionality of the dataset according to similarity of each column's variances.

[Back to top](#Resume-Project-List-Summary)

# Grouping Customer Characteristics by Bandwidth Usage With K-Means Clustering

## About
In general, higher bandwidth subscriptions cost more and therefore bring more revenue for an ISP. The Customer Churn data set contains a field named Bandwidth_GB_Year, which is measured in GB used per year. Histograms can be used to identify high or low range users by other categorical fields like Area or PaymentMethod. Furthermore, classification algorithms can be applied to predict which bandwidth use ranges correlate with such categories. Such classifications may not be so apparent when compared to other continuous attributes. Perhaps there is a distinct difference in bandwidth use for customers by Tenure, Income, or other continuous fields included in this data set. Being able to make distinctions like this may then be worthwhile for marketing purposes.

A customer's income, for instance, could be arbitrarily subsetted into discrete intervals so that bandwidth usage could be fitted to a classification model to make predictions. This kind of approach may risk neglecting to detect a clustering of bandwidth usages that correspond with a less arbitrary range of incomes. Attempts to apply classification models when comparing the relationships between continous variables seems to fail in this regard. Regression can provide a means of predicting individual outcomes, but give no indication on how to group either predictor or target variables. An unsupervised machine learning algorithm that groups data points according to distance measures may be better at classifying relational correspondences through clustering. Such techniques provide means of discovering patterns and natural clusters amongst data. (DataCamp)

Given that high bandwidth users generate more revenue for an ISP it would then seem worthwhile to get the most out of existing data to identify segments of customers by continuously defined characteristics. To that end is it possible to identify a distinguishable range of incomes that correspond with high bandwidth users? What about other continuous factors such as the population of customer locale, outage time experienced, bill amount, age, or other possible features? Is there a measurable way to determine which relationship produces the most distinguishable segmentation in order to prioritize marketing efforts?

## Clustering Techniques and Assumtions
One way of clustering data begins with a single point then finds the closest neighbor via Euclidean distance. A centroid of these 2 points can be computed as the mean distance between the 2 points, and used as a reference to find the nearest distance of a 3rd data point. The inclusion of more and more points in this manner form a cluster. Centriods of clusters can then be used as endpoints from which distances to other clusters are computed. Such a bottom up algorithm describes hierarchical clustering. A set of random points would need to be specified to start with before carrying out the rest of the subsequent calculations, and the results of clustering after a certain number of iterations may differ depending on which original random points where initially selected. This may lead to a computationally expensive process.

Another way of forming clusters begins with computing the centroid of a randomly selected number of points. Distances between such collections are computed between respective centroids. A certain number of clusters can be specified, so that the distance between respecitve clusters is maximized. This gives a more computationally efficient means of clustering data points since more points are considered at each iteration than with the hierarchical method. This way of categorization is called K-Means Clustering, and is particularly appropriate for large data sets, especially if there is a need to cluster amongst multiple features.

Hierarchical clustering can produce a visual representation of the formations via a dendrogram which represents groupings in a sort-of upside down binary tree diagram where higher level cluster distances are indicated along the y-axis. (Pai, 2021.). When applying K-Means clustering in Python a pair of cluster centers and distances between clusters is implicit to the syntax of returnable objects. This makes it possible to create an "elbow plot" of cluster distances against number of clusters. As the number of clusters increases there should be a relatively discernable point on this plot where the rate of decrease in distance becomes more apparent. In terms of calculus, a second derivative would have a higher magnitude and imply a greater degree of concavity at such a point. In other words, looking at such a plot gives an erzats quantifiable indication of optimal number of clusters. The severity of the bend in this kind of plot provides insight into how distinct the clusters are.

There are several continuous candidate target variables for which to attempt clustering bandwidth usage by in this data set. Unfortunately, looking at binary plots of each it becomes rather noticeable that there are generally 2 clusters for each. The goal of this project is then to use K-Means clustering and elbow plots to determine which continuous relationship with bandwidth usage results in the most pronounced clustering.

## Packages Used
The data was processed, modelled and analyzed using Python. The following packages and libraries were used:

Pandas: general dataframe handling

warnings: ignore uneccesary warnings that don't affect outcomes of data operations

sklearn

preprocessing StandardScaler - normalizing/standardizing scipy - usupervized machine learning package

cluster - clustering kmeans - compute centroids and distances (distortions) vq - labelling clusters matplotlib.pyplot: plotting and graphics

seaborn extension of matplotlib, emphasis on ease of using dataframes

[Back to top](#Resume-Project-List-Summary)

# Optimizing K-Nearest Neighbors Classifier to Predict Customer Churn

## Research Question
Any company that provides a service wants to maintain their existing customer base. Churn is a commonly used term related to customers discontinuing their terms of service. For an ISP a customer who has “churned” has ceased to continue to use the services for which they have subscribed. It is more costly to attain new customers than it is to acquire new ones, so minimizing churn is a crucial aspect of maintaining profitability and providing good service. The Customer Churn data set contains a Churn column that is defined by whether a customer has ended their services within the last month, indicated by a Yes or No value.

The specific products and services offered may play an influential role in a customer’s decision to stop their subscriptions. Such a collection of ISP controlled offerings may serve as input for a predictive model for customer churn. This may beg the question of how a saturated collection of ranges of input variables may be used to formulate such predictions. A logistic classification model dependent on a linear combination of predictor variables could be applied in this situation, but there may exist further subtleties. In a logistic model a predictor variable is ascribed a static coefficient of variability in its contribution to prediticting the likelihood of the outcome.

There may be ranges of a predictor variable which conrtibute more weight on the outcome than others, and there may be more than one of these such sets of ranges. Perhaps the longer a customer experiences an outage on average may steadily impact the likelihood of churn overall, but what if there were a concentration of outage times that have more of an impact than others? Perhaps customers in a mid-to-high range of outage times were more likely to churn than those who experience a more middle of the range outage times. It would seem that, if this were the case, then such ranges would have differing impacts on predicting churn. This is the central notion of the K-Nearest Neighbors (KNN) classifier. Logistic classifiers draw a single smooth decision boundary on its prediction, whereas KNN may divy up several complex boundaries. (Supervised Learning with scikit-learn , Ch. 1, lesson 1). Does such a classifier predict churn better?

## Objective
A logistic predictive model can be applied to make predictions for an outcome with only 2 possible values, but in order to capture the kinds of subtleties mentioned above K-Nearest Neighbors classification may provide more accuracy. Beginning with a set of attributes that describe customer charateristics an initial model can be made. Model performance attributes can be examined to make improvements, which may require certain parameters to be adjusted. The objective is to optimize parameters to obtain a KNN model that most accurately predicts churn. Furthermore, it would be interesting to see if it can outperform logistic classification.

## Data Goals
Using computational quantitative modelling can be used to aid in data driven insight. Churn is a binary valued field, and such modelling can be used to predict the likelihood or probability of a yes or no churn decision occurring. Probability can be calculated but require numerical input. There are several categorical fields in the data set that may be useful to make a prediction. If such fields could be ascribed to numerical values they can be of use.

The data set includes several fields that describe product and service offerings that customers can choose from. Examples of such fields include InternetService, DeviceProtection, StreamingTV, and Contract. These all have categorical values of Yes/No, or a small number of options. There are also some fields that may be descriptive of customer experience, but can be influenced to some extent by the ISP. These type of predictors generally have a continuous range of values. Email describes how many times the ISP has contacted a customer via email. MonthlyCharge is what the customer pays for their services, but can be raised or lowered at the discretion of high level management. If the outcome of predictive modelling shows that churn can be reduced by lowering bills, then perhaps promotions or discount offerings may be worth implementing. Bandwidth_GB_Year is an indicator of level of internet usage that occurs in a customer household, but marketing efforts can be tailored to target low or high level users. Outage_sec_perweek and Yearly_equip_failure can be indluenced by the ISP through increased focus on network maintenance and better device offerings.

Fields that are more descriptive of customer characteristics will not be considered in this treatment. These include demographic features that describe geographic location, personal characterisics, and other opinion driven aspects. Honing in on the features that are more controllable with a detailed model may provide more immediate value towards decision making.

This leaves 19 categorical and numeric features to be explored as explanatory variables in the model. The numerical columns can be further subdivided into discrete and continuous.

Categorical: Contract, PaperlessBilling, Port_modem, Tablet, InternetService, Phone, Multiple, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreaminMovies, PaymentMethod

Discrete numerical: Email, Yearly_equip_failure

Continuous: Bandwidth_GB_Year, MonthlyCharge, Outage_sec_perweek

## Assumptions and Method
The KNN classifier alogirthmically considers how close a certain number of surrounding data points are to each other based on a Euclidean distance measure. Even when all categorical predictor variables are transformed numerically the ranges may differ widely. For this reason it is recommended that all variables be normalized on the same scale. The set of predictive and target features can be split into training and testing sets. A certain percentage of data can be used to create the model, while the rest can be used to measure how well it performs. By comparing predictions to actual values in the test set accuracy measures can be computed. A variation of such an accuracy measure can also be used to determine the best number of neighbors to use.

## Tool Benefits and Technique
Python has several libraries containing pre-coded functions that can make model building, parameter optimization and computing probabilities and accuracy metrics very efficient. The sci-kit learn library was used heavily. The following list includes packages and functions used:

sklearn

model_selection

train_test_split - splitting the dataset

GridSearchCV - parameter optimization

neighbors

KNeighborsClassifier - KNN model creation

metrics

confusion_matrix - see below for explanation

classification_report - model performance metrics summary

roc_auc_score - see below for explanation

roc_curve - see below for explanation

preprocessing

StandardScalar - scaling/normalizing data

Pipeline - apply multiple operations on data

linear_model - logistic regression model

pandas - general dataframe handling

Splitting the data allows for computations of True Positive (TP), True Negative (FN), False Positive (FP), and False Negative (FN) outcomes. These 4 values are typically summarized in a confusion matrix, and used to cumpute various model performance metrics.

Model Accuracy is essentially the ratio of the number of correct predictions to total number of predictions.

Accuracy = TP + TN / (TP + TN + FP + FN)

True Positive Rate (TPR) and False Positive Rate (FPR) are also comuputed from these 4 values. A plot of TPR vs. FPR gives an ROC (receiver operating chatacteristic) curve. The area under the curve (auc) provides a measure of how well a variable contributes the prediction; 0 being weakest to 1 being strongest. (Machine Learning Crash Course)

TPR = TP / (TP + FN)

FPR = FP / (FP + TN)

from sklearn.metrics import roc_auc_score

Precision is the proportion of positive instances that were correctly identified. Recall is the proportion of actual positive cases that were correctly predicted. It is clear that a good model will have both high sensitivity and specificity. If these ratios are too low the model may be overfit to the test data.

Precision = TP / (TP + FP)

Recall = TP / (TP + TN)

The F1-score considers both, and is used in a process called grid search cross validation to determine the optimal number of neighbors.

F1 = 2 x (Precision x Recall) / (Precision + Recall) (Analytics Vidhya)

In KNN larger values of K generally leads to a smoother decision boundary, while smaller K has a more complex decision boundary which can lead to overfitting. The .score() function provides a simple means of measuring accuracy of model predictions on a sample measured against actual outcomes in the same sample. (scikit-learn.org). A plot of training and testing set accuracies can be made for varying K's. Such a plot is called a model complexity plot. (Data Camp - Supervised Learning with scikit-learn, Ch. 1) In some cases when K becomes too large the accuracies of training and testing samples diverge from each other, and this is where underfitting occurs. The sweet spot on the model complexity plot occurs when the accuracy measures are closest together.

[Back to top](#Resume-Project-List-Summary)

# Predicting Bandwidth Usage With Random Forest Regression

## About
In general, higher bandwidth subscriptions cost more and therefore bring more revenue for an ISP. The Customer Churn data set contains a field named InternetService, which describes the type of internet service a customer subscribes to and reveals that the provider offers both DSL and Fiber Optic based network options. A fiber optic network can provide much higher speeds than DSL, but requires more complex infrastructure. 20 years ago bandwidths on the order of 100MBps or even in excess of 1GBps were not very desirable, but today's consumers often require such higher speeds. Holding on to higher bandwidth using customers and even expanding a fiber optic infrastructure footprint is therefore a worthwhile effort.

The churn data set provides several fields that describe the types of services a customer subribes to, some information on specific devices they use, demographic data and measures of sentiment regarding customer experience. All of these fields may be used to give a profile of high bandwidth using customers. The ability to identify them can be useful to orient marketing and customer service interaction priorities to promote greater retention. Demographic and characteristic data may be available from other sources for geographic regions beyond the current footprint so that areas of potential network expansion may be better identified. Is it possible to create such a predictive customer bandwidth usage profile, and if so which fields have the most influence?

## Predictive Method
Bandwidth usage, as a feature in the Customer Churn ISP data set includes a wide range of continuous numerical values. A classification model would not be appropriate to make predictions, as such a model indicates the likelihood of an outcome going one way or another. To make predictions on a continuous outcome a regression model would be more suitable. Linear regression could be used, but it may turn out that there is not a linear relationship between predictor and target variables. A decision tree based algorithm can capture non-linear subtleties by learning from rules applied to features. (Li, 2019).

A tree structure begins with a root node containing the entire sample of training data that splits into child nodes based on a decision made according to an information gain metric. These child nodes then get split into more child nodes of thier own based on another decision. This process repeats until the final collection of subsets result in a prediction on the target variable. (Gurucharan, 2020). Decision trees can be used for either classification or regression, and the number of of subsequent node splits is referred to as tree depth.

As the depth of a tree gets larger the size of each node's subsets gets smaller. If the tree were allowed to continue to the point of irreducible subset size then predictions may become too specific to individual data points, which poses the risk of overfitting. Two trees applied to the same data may still have differing predictions at the end. Instead of running several trees with different parameters a collection of randomly composed trees can be computed in a single algorithm called a random forest. The random forest considers many permutations of subsets that individual trees use, then returns a majority vote of each of the trees' leaves' predictions.

The random forest regressor needs to know how many trees to use ("n_estimators"), and each tree should have the number of levels specified ("max_depth"). These are parameters of the model that can be tuned through cross validation. Multiple options of these paramaters can be input into a search algorithm to find which ones result in the greatest accuracy. Some other parameters include, but are not limited to, number of features considered in the split decision ("max_features") and the minimum number of samples required in a leaf node ("min_samples_split"). (scikit-learn, 2007-2021)

## Packages Used
The data was processed, modelled and analyzed using Python. The following packages and libraries were used:

Pandas: general dataframe handling

warnings: ignore uneccesary warnings that don't affect outcomes of data operations

sklearn

model_selection

train_test_split: split data into training and testing sets RandomizedSearchCV: cross validation using random grid elements ensemble

RandomForestRegressor: create and fit random forest model metrics

make_scorer: define a specific accuracy score mean_squared_error: computes mean squared error in predictions matplotlib.pyplot: plotting and graphics

[Back to top](#Resume-Project-List-Summary)

# Market Basket Analysis and Product Sales Suggestions

## Research Question and Goal
By analysing purchase history of customers, can certain products be recommended given that other products have been purchased?

The main goal would then be to identify 3 top if then relationships between one product purchased and high likelihood of another.

The data set used is an itemized breakdown of technical product purchases for 7,501 telco customers, with each row considered as a history of transactions.

## Background
Proportions of single products can be calculated by summing up all instances of each then dividing by the total. This would indicate a probability of an item being purchased. The probability of 1 or more items being purchased is also known as a metric called Support. Furthermore, the conditional probability of purchasing one product given that another had been purchased can also be calculated. Another name for this metric is Confidence. A third useful number is called Lift, which indicates the degree of randomness in Confidence. (Datacamp)

Support(X) = freq(X) / N

Confidence(X, Y) = Support(X, Y) / Support(X)

Lift(X, Y) = Support(X, Y) / Support(X)Support(Y)

The essential idea behind Market Basket Analysis (MBA) is to use these 3 metrics to form Association Rules. These rules take an if then form of Antecedent => Consequent. In other words the 3 metrics above allow for the likelihood of a consequent product being purchased given that an antecedent was purchased. These rules can only be obtained from existing data, so they don't necessarily indicate causality. Nevertheless, history can be valuable in attempting to predict future patterns and the purchase history in this data set covers 2 years.

There are 20 unique items in this data set, and the number of combinations of products purchased increases astronomically as more items are considered. Thankfully, there is an implimentable algorithm that can be used to narrow down the number of combinations. According to Datacamp "The Apriori algorithm is structured around the idea that we should retain items that are frequent -- that is, exceed some minimal level of support. The Apriori principle states that subsets of frequent sets must also be frequent. The Apriori algorithm uses this principle to retain frequent sets and prune those that cannot be said to be frequent."

In general confidence, which ranges from 0 to 1, should be high. Also, a lift greater than 1 implies that 2 items in a transaction exceeds their random occurance together. Python also has an association_rules function that ranks association rules according to support, confidence and lift.

[Back to top](#Resume-Project-List-Summary)
