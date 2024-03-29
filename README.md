# Resume-Project-List-Summary

Each of the projects listed below were motivated by specific business objectives that were thoughtfully determined according to the context of the data used.  Some utilized the same data set, and may not always contain thorough exploratory data analysis (EDA) or data cleaning sections.  For those that do, numerical and graphical distributions were explored and outliers and/or missing values were mitigated using IQR or Z-score criteria.  Many include a discussion of the advanced mathematics behind the ML models used.  

[Linear Regression 1: Predecting Customer Tenure](#Predicting-Customer-Tenure-Using-Linear-Regression)

[Linear Regression 2: Predicting Catalog Demand](#Predicting-Catalog-Demand)

[Logistic Classification 1: Predicting Customer Churn](#Logistic-Classification-Modeling-on-ISP-Customer-Churn-Data)

[Logistic Classification 2: Identifying Loan Borrower Features Associated With Timely Payback](#Towards-Automated-and-Customized-Loan-Approval-Criteria-Using-Segmented-Logistic-Classification-Models)

[Principle Component Analysis (PCA)](#PCA-Variance-Thresholds-and-Dimensional-Reduction)

[K-Means Clustering](#Grouping-Customer-Characteristics-by-Bandwidth-Usage-With-K-Means-Clustering)

[Supervised Naive Bayes, SVC, Decision Tree, and AdaBoost Classifiers](#Using-Machine-Learning-to-Identify-Persons-of-Interests-from-Enron-Emails)

[K-Nearest Neighbors (KNN) Classification](#Optimizing-K-Nearest-Neighbors-Classifier-to-Predict-Customer-Churn)

[Random Forest Regression](#Predicting-Bandwidth-Usage-With-Random-Forest-Regression)

[Market Basket Analysis](#Market-Basket-Analysis-and-Product-Sales-Suggestions)

[Time Series](#Time-Series-Analysis---Using-a-Seasonal-ARIMA-Model-to-Predict-ISP-Revenue)

[NLP 1: Sentiment Classification Using RNN](#Using-Labeled-Customer-Reviews-to-Make-Binary-Sentiment-Predictions-With-NLP-and-Recurrent-Neural-Networks)

[NLP 2: Word Corpus Groupings Using Cosine-Similarity](#Identifying-Internet-Outages-Using-NLP-With-Twitter)

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

# Predicting Catalog Demand

## Business and Data Understanding
One way that a high-end home goods manufacturer generates revenue is through catalog purchases. The company has kept demographic and past sales data, and has used this to calculate likelihood for a customer to make a purchase whether or not they responded to the catalog. Another dataset for 250 new customers, who have not yet had a chance to respond to the catalog contains the same demographic data, and columns for the probability of them buying from the catalog as it would impossible to inlcude of them already having done so. Management requested a newly acquired analyst to help decide whether issuing catalogs to these new customers would be worthwhile, and has determined that an expected net profit of $10,000 would need to be exceded in order to make this decision.
The data from existing catalog receiving customers can be used to predict sales using other attributes in a linear model. Graphical exploration of trends for potential predictor variables, and mathematical indicators describing strength of fit and probability of impact on outcome can be used to determine the best model to make these predictions. Profit margin, the cost of printing catalogs, and the likelihood for catalog purchase can then be used to calculate the final overall expected profit.

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


# Using Machine Learning to Identify Persons of Interests from Enron Emails
## About
In 2001 the Enron Corporation filed for bankruptcy after it was found that several executives were involved with fraudulent financial activities. After a federal investigation data on these executives were made public, including emails, and salary and bonus amounts. Persons of interest in the investigation, POIs, are identified in the features of email data sets examined in this project.

The Enron email list data set has been used widely over the years, and can be accessed via multiple ways. The raw data can be found from: https://www.cs.cmu.edu/~enron/, then downloading the May 7, 2015 version as a tgz file. For this project it was preprocessed in the form of a data dictionary using Python 2.7. It was obtained from Udacity's repository for their machine learning course:

https://github.com/udacity/ud120-projects.git

Program files
final_project_dataset.pkl
        cloned data set

poi_id.py

        The following Python modules were used:

                numpy, pickle, sklearn

1. Select list of features, and update dictionary
2. Identify and remove outliers
3. Use K-best features to identify features most indicative of POI
4. Initialize classifiers: GuassianNB, SVC, DecisionTree, and AdaBoost were tried; SVC had the highest accuracy
5. GridSearch used on SVC model to optimize parameters
The following are saved, pickled files for the original modified dataset, classifiers, and feature lists:

        my_dataset.pkl

        my_classifier.pkl

        my_feature_list.pkl

## Features
Each key-value pair in the dictionary corresponds to one person. The key is the name, while the values are features that can be analyzed via machine learning algorithms. There is a total of 146 people in this set, 18 of which are designated as POI’s. There are 21 features for each person, they can be categorized as follows:

financial features:

['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features:

['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label:

[‘poi’] (boolean, represented as integer)

Persons of interest may email each other more frequently then others not involved with the scandal. Machine learning can be used to find evidence of correlation between POI and the features.

### Outliers
Reading through the dictionary keys I found one entry that doesn’t look like a name. I removed ‘THE TRAVEL AGENCY IN THE PARK’ from this list of keys. Plotting Bonus vs Salary, we see one outlier in the upper right corner. Running some code to loop through these 2 features reveals that the name corresponding to this point is ‘TOTAL’. There’s no sense in keeping this point for totals in the data set.

outliers_1

Bonus vs. Salary before removing outlier.

outliers_2

Bonus vs. Salary after removing outlier.

Also, looking at names/keys that have NaN above a threshold percentage as values it was found that ‘'LOCKHART EUGENE E' has all NaN for feature values. Others had over 85% NaN values, but some nonNaN values had strong K-Best-Features scores, so I decided to keep them.

### Updated Features
It stands to reason that emails sent to and from POIs may be an indicator of the person sending or receiving is also a POI. The raw number of such emails doesn’t give away much without a comparison. Two new features were created for proportions of emails sent/received to/from POI to total emails sent received to/from that person:

“prop_email_to_poi”, and “prop_email_from_poi”.

These new features were used along with the rest in the SelectKBest selection tool. The 'f_classif' score function was used, meaning that ANOVA F-statistics were used to measure significance. The top 5 highest scoring features are listed in the table below. Of note: the “prop_email_to_poi” feature is the 5th highest scoring feature.

Feature	Score
exercised_stock_options	24.82
total_stock_value	24.18
bonus	20.79
salary	18.29
prop_email_to_poi	16.41
Feature Score exercised_stock_options 24.82 total_stock_value 24.18 bonus 20.79 salary 18.29 prop_email_to_poi 16.41

## Algorithms
Default parameters were tried for each of the 4 algroithms used: Gaussian NB (Naïve Bayes), SVC (Support Vector Classification), Decision Tree, and AdaBoost. My ‘features_used’ were of the form [[‘poi’ , ‘feature’], where ‘feature’ is one of the 5 individual features above. To help choose which algorithm to focus on the accuracy was tabulated with and without scaling for each feature in each algorithm. The scaling used was the MinMaxScaler, which is the ratio of the distance of each value from the minimum to the range of values.
Feature Summary by Algorithm

### Naive Bayes

Accuracy	Bonus	Exercised_stock_options	Prop_email_to_poi	Salary	Total_stock_value	Avg
Scaled	0.84	0.951	0.852	0.763	0.84	0.851
Not Scaled	0.848	0.951	0.852	0.763	0.84	0.851
### SVC

Accuracy	Bonus	Exercised_stock_options	Prop_email_to_poi	Salary	Total_stock_value	Avg
Scaled	0.848	0.902	0.852	0.737	0.78	0.824
Not Scaled	0.697	0.878	0.852	0.737	0.76	0.781
### Decision Tree

Accuracy	Bonus	Exercised_stock_options	Prop_email_to_poi	Salary	Total_stock_value	Avg
Scaled	0.788	0.854	0.667	0.763	0.72	0.758
Not Scaled	0.788	0.854	0.667	0.763	0.72	0.758
### AdaBoost

Accuracy	Bonus	Exercised_stock_options	Prop_email_to_poi	Salary	Total_stock_value	Avg
Scaled	0.788	0.854	0.667	0.763	0.72	0.758
Not Scaled	0.788	0.854	0.667	0.763	0.72	0.758

The GaussianNB algorithm has the highest average metrics for all features used. It is clear to see that ‘excercised_stock_options’ has the highest accuracy feature in all algorithms used. The only algorithm affected by scaling was SVC, and I chose to focus on it for reasons that will be explained in the next section.

## Parameter Tuning
SVC has more parameters to adjust than Gaussian NB and trying just a few different kernels resulted in different accuracies. This is an example of parameter tuning. Instead of running the code and noting the accuracy with each parameter adjustment, grid search can be used to optimize accuracy over a range of parameters. I’d like to see if there are other parameters for the SVC algorithm that may increase accuracy. For this I used the following parameters: 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
A better accuracy of 0.927 was obtained.

## Validation
Validation refers to how accurate a model can make predictions on a testing data set based on its performance on a training set. One thing to be wary of is making sure that training and testing data sets don’t contain patterns that could distinguish them. For example, if there is a pattern in the training set it would be applied to the testing set, but if no such pattern exists in the testing set then those points wouldn’t be classified well. The data should be randomized to avoid this.
Overfitting is a common pitfall with validation. The extreme case is when the training and testing sets are the same – of course the same predictions will occur, but it’s as if you’re not really making a prediction at all. Another case is when an algorithm is too sensitive to small changes in the training set. Small changes generally should be attributed to noise, but the excess sensitivity in the model will look for similar patterns in the testing set when they probably should be ignored.
Adjusting the size of training and testing sets can lead to a balance model that avoids overfitting and maximizes performance. I started with a test size of 0.4, since that was used in the course mini projects. The accuracies for different test size proportions are summarized below:

Test Size	0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8
Accuracty	0.909	0.905	0.968	0.927	0.882	0.852	0.901	0.901
Here we see that accuracy increases as testing set proportion increases, then peaks at 0.3, but starts to increase again after 0.6. This increase where over half the data set is being used for testing is where overfitting starts to take hold. Using 3/10 of the data for testing gives the best accuracy.
A better method is K-fold Cross Validation which partitions the testing and training sets into k number of folds. It is better, because it iterates over all data using different partitions of the same size for training and testing. For example, a 3-fold validation starts with 2/3 of the data used for training, and the remaining 1/3 for testing. The next iteration uses another partition of 2/3 of the data for testing and so forth for 3 iterations. Validation scores for each iteration can be found using the cross_val_score function. Using 3 folds, I obtained an average accuracy of 0.88.

## Evaluation Metrics
Precision is the ratio of true positive to true positive plus false positive. In this context, it is the ratio of those that are truly POIs to the sum of true POIs and mistakenly identified POIs. Recall is the ratio of true positive to true positive plus false negative. In other words, it is the ratio of those that are truly POIs to the sum of true POIs and POIs that were mistakenly not identified.
The SVC algorithm using grid search yielded a precision of 1.0. This means that my model did not produce any false positives in its prediction. My recall was 0.667, so the model produced some false negatives.

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

# Time-Series-Analysis - Using a Seasonal ARIMA Model to Predict ISP Revenue

## Purpose
At a very high level, the upper-most tiers of leadership including the board of directors of an ISP may be interested in a quick summary of overall revenue trends. Seasonal fluctuations and possible predictions for future time periods should be available for such a broad measure after the first 2 years of operations.

An analyst or data scientist may begin their approach to such an insight driven task by asking themselves if there exists any periodic repetetiveness over time for overall revenue. During the first 2 years of operations is there a significant difference in the revenenue for Q2 from the other quarters or seasons? Perhaps a more interesting report would say whether revenue is expected to increase in future months. Crucial decisions regarding investor engagement and top-down reorganization of the company may impinge upon these results.

## Method_Justification
In addition to periodic fluctuation in revenue over time there may exist larger upward or downward trends over the 2 year span of available data. A smoothed out pattern of mean revenue could be plotted to help visualize this if that were the case. On the other hand, too much fluctuation on a smaller scale may indicate noise which could impeede efforts to make predictions. A machine learning model may place too much emphasis on this noise and result in overfitting on test data. A transformation to mitigate larger trends can ensure stationarity of time series data, while spectral decomposition can aid with reducing random noise. All these factors should be considered when building a predictive time series model.

Autocorrelation takes samples of times series data over the same time intervals and measures strength of correlation between them. An example of a single lag could be correlating one month of data with the previous month. When more previous months, or any lengths of time period are correlated the lag count increments by 1 for each successive interval. With more lag intervals there are more possibilities of interactions amongst sub-intervals, and autocorellation treats all of these. Partial autocorellation looks only at interactions between adjacent lags. (Abhishek, 2019) An Auto Regressive (AR) model uses partial auto-correlation, and yields an equation of the form:

Y(t) = B1 + M1Y(t-1) + M2Y(t-2) + ... + MpY(t-p),

where p is the lag order in the time series, and the Ms are the weights of lagged observations.

Moving Average (MA) treats errors in the lagged observations from AR. The resulting equation is of the same form as the one derived from partial autocorrelation.

Yt = B2 + w1E(t-1) + w2E(t-2) + ... + wqE(t-q) + Et.

Here the w's are weights of the error terms E, and q is the size of the moving window.

When AR and MA are combined the equations is:

Yt = (B1 + B2) + (M1Y(t-1) + ... + MpY(t-p)) + (w1E(t-1) + w2E(t-2) + ... + wqE(t-q) + Et)

Another important feature of a time series to consider is the behavior of means and standard deviations within lagged intervals. One way that means may differ is if there are seasonal trends in the data. While the means within fall and winter months may not differ too much, the larger spring and summer means might. This indicates seasonality within the time series. Distributions amongnst intervals may also exist, which would indicate the MA error term is not constant. For a time series to be stationary the means and standard deviations of lagged samples need to be constant, and there should be no seasonality.

An Integrated ARMA (ARIMA) model mitigates moving means. This can be accomplished by taking the difference of successive lagged intervals, or perhaps using a log or other form of transform. Before ensuring the stationarity assumption and performing any transformation, it is imperitive to check for any missing values and deal with outliers.

[Back to top](#Resume-Project-List-Summary)

# Using Labeled Customer Reviews to Make Binary Sentiment Predictions With NLP and Recurrent Neural Networks

## Research Question and Goals
Can natural language processing be used to predict negative or positive customer sentiment based on their verbal or writtent reviews? This would give companies a better idea of customer perceptions and may form a basis for finding areas of improvement.

Written customer reviews from 3 sources: Amazon, IMDB, and Yelp will be examined. Each review in these data sets are assigned a sentiment rating or 1 for positive, or 0 for negative. There are thusly 2 columns with each row describing a single review and sentiment pair. The number of words in each review may vary greatly, and so a model to make such predictions needs to be able to handle a wide range of input sizes. A neural network (NN) is such a model. Some preproccessing is required in order for it to work right though. The inputs to be numeric, so a method of assigning words to numbers needs to be employed. Counting similar words and even the same words typed using different case can affect predictions, so can the appearance of numbers or special characters. An algorithm can only work with what is fed to it, after all. With that in mind, meaningless words like 'an' or 'the' would not add much value. Maximum number of processed words and average word length are other factors that will come into consideration when building the network.

Python's natural language toolkit (nltk) will be useful for simplifying vocabulary. Sklearn has some useful functions for representing words and letters as numbers and splitting data into training and testing sets. Pandas and Numpy are involved with data handling, while matplotlib allows for graphing. The Tensorflow and Keras packages allow nueral networks to be constructed. In particular, Keras allows for high-level api creation of NNs using layers. Since a NN can be represented as sets of interconnected nodes divided into layers it makes sense to be able create them that way. Each connection between nodes applies a weight to the node value which is transformed via a mathematical function. Predictions are made by the changes that data goes through under these transformations. Along the way performance metrics are honed and optimized in both directions throughout the network. A NN that can do this is called Recurrent, or RNN.

[Back to top](#Resume-Project-List-Summary)

# Identifying-Internet-Outages-Using-NLP-With-Twitter

## Project Summary
The overall goal of this project was to provide alerts for possible internet outages by analyzing content from customers posted on Twitter. The result was a visual dashboard showing counts of observed word groupings related to possible outage types. This dashboard was incorporated into the ISPs existing outage reporting system, and was used to trigger email alerts to relevant work groups to aid in diagnosing and resolving a common cause issue.

Directed Tweets were already being responded to by technical support representatives, and this same set of Tweets was chosen to analyze. Python’s TwitterScraper package was used to extract and filter Tweet content. A Twitter Developer account was needed to be created in order to use this package. The remains were exported to comma separated files (CSVs). Spurious content such as stop words, articles, and names were filtered out. Different forms of the same words were also grouped together in a process known as lemmatization. For example, “talk”, “talking”, and “talked” are different verb forms of the same word. Once the content of Tweets was cleaned, counts of the most frequently occurring words and 2-3 word phrases were extracted. The sets of most frequent words and phrases were then organized into dataframes using Python’s Pandas data analytics package.

From here word groupings were identified using a form of machine learning called natural language processing (NLP). Lists of words that may describe internet outages and those that don’t were identified and used to test for similarity to form the word groupings from NLP. These lists were agreed upon during a brainstorming session involving all team members. The Word2Vec NLP model was used to obtain the groupings, and cosine similarity scores were computed to measure relatedness of groupings to the list of outage related terms. A t-SNE plot to visually represent the degrees of relatedness of words in Tweets to outage descriptors was made. Each word appears as a point on a 2-dimensional plane, color coded according to grouping. The larger the point, the stronger the word is related to an outage.

The ISPs existing outage reporting system was used as well. Outage categories, durations, and locations were extracted. The ISP uses an Oracle database to store the information from this system so SQL was used to make these queries. TwitterScraper was used again on each word grouping for the times that each outage actually occurred. If any frequency of word groupings coincided with the duration of an outage it was noted along with the number of words and phrases in each group. Since TwitterScraper does not provide exact locations, only cities that users identified were used to compare with real outage locations.

For some groupings it was clear that a higher spike in word groupings were related to outages, for others statistical T-tests could be performed. These T-tests allowed for rejection of a null hypothesis that the mean word frequencies for coincidence is less than or equal to the mean for non-coincidence. For coinciding groups the mean frequency minus standard deviations were computed to use as a threshold indicator of possible outage. This threshold was divided by the time of outage duration to obtain a scalable metric.

Now that word groupings and thresholds of occurrences were identified, their figures were summarized graphically in Tableau. Each outage related word grouping was color coded and the counts for outage durations were displayed in histogram form. This visual dashboard was then added to the existing outage reporting system, and functionality to adjust time intervals was added.

Twitter’s API provides geolocations, but can only be used to extract data going back one month. The same process of comparing with real outages carried out above was performed for the previous month. Deviations in corresponding word group frequencies were found to be negligible during that interval, so the metrics found from the longer duration analysis were kept.

Scripts to trigger the code to extract and analyze Tweets using the API were made. These scripts were set to run every hour. The output of word group counts were then updated to the dashboard with each iteration. The threshold of frequencies was indicated as a solid vertical line in each histogram column to visualize if/when the counts indicate a possible outage. If the height of a bin exceeded that threshold an alert was triggered. Scripts to monitor that were made, which then triggered sending emails to relevant work groups. The relevant work groups and content of emails were determined during an early brainstorming Scrum meeting.

A representative from each work group was chosen to respond to follow up to confirm that alerts and emails were received during the last day of work. Representatives from each organization that uses the dashboard were also consulted to make sure they were able to access it. After these meetings it was determined that further training on the use of these new features was needed.

## Code Overview
Python 3.7 was used in a Jupyter notebook for ETL and NLP. The code is included in InternetOutagesNLP.ipynb The following specific modules used include:

twitterscraper - used for extracting Tweets
pandas, numpy, matplotlib - data anlytics packages
sklearn, seaborn - machine learning, and statistics
nltk, word2vec, CountVectorizer - NLP packages
The following outlines what the code does
Functions, packages, and ML models are indicated in bold.

Data frames and lists are indicated using italics.

Tweets were extracted and saved to csvs from a list of southwestern cities including: Houston, San Antonio, Austin, Dallas, Fort Worth, Oklahoma City, Tulsa, Santa Fe, and Aluquerque.
The csvs were read into data frames with text, timestamp and location fields.
The text_clean() function used a regex to filter only words, and a custom list of stop words was used for further filtering.
The cleaned data frame tweets_df was then updated.
CountVecotrizer was instantiated and summaries of top 10 occuring words and phrases were found.
The nltk.corpus stopwords package and further use of regex's were applied to tweets_df, before the remains were composed into the final_tweet_list list to be tokenized.
The Word2Vec model was instantiated as tweets2vec and trained on final_tweet_list.
A t-SNE model from sklearn was applied to the tweets2vec model.
The vectorize_corpus() function was used to create a corpus of key words related to outages.
Lists of words related to outages, and opposites were initialized: internet_out and not_out.
Cosine similarity scores were computed using the cos_sim() function to quantify similarity of words from final_tweet_list to internet_out and not_out lists.
A t-SNE plot was created to show strength of relatedness according to size of circles in 2D plot, with blue relating to occurance of outage, and grey corresponding to not being related to outage.

## Project Methodology
The Scrum project implementation methodology was used to carry this project out. Scrum breaks a project down into sprints, which involve team members working in a focused manner on a specific objective. Scrum meetings are scheduled to communicate progress on achieving benchmarks. Little scope creep or deviation is allowed, which keeps teams focused on their goals defined by the sprints. The path from extracting Tweets to presenting a functional dashboard can be broken down into manageable tasks.

A team of analysts, developers, a data scientist, a product owner, and Scrum Master were assembled. The product owner and Scrum Master are roles specifically designated to manage a Scrum project. The product owner served as liaison between the project group and stakeholders. Stakeholders include the relevant workgroups identified above, as well as some senior management. This role is also responsible for scheduling and managing any backlog of objectives that may not have met the planned timeline. The Scrum Master serves as a team leader in the project. They work closely with team members to keep the project on schedule and communicate any needs of the team with the product owner. The scrum master also conducts sprint meetings and documents progress in achieving objectives.

The analysts were heavily involved with the earlier objectives of the project including. These objectives include:

Scraping data from Tweets by username going back 5 yrs.
Cleaning data – remove stop words, etc.
Preparing data frames and conduct NLP
Extracting data from existing outage reporting system going back 5 yrs.
The data scientist was then responsible for performing the machine learning aspects. This included:

Performing correlation study between frequently appearing words and actual outage occurrence
Grouping high frequency words into training and testing feature sets
Coding and running Word2Vec algorithm
Developers played a crucial role in the latter portions of the project including:

Managing locations of scripts and csv’s, and creating batch files for them
Scheduling execution of scripts, and email alerts Analysts were also responsible for creating the Tableau dashboard, while developers linked it to the existing outage reporting site.
## Vizualization Examples
### t-SNE Plot
![image](https://github.com/MaxSydow/Resume-Project-List-Summary/assets/56166497/84047099-eba3-4435-a347-785476a796f7)


Example bar chart (made with Tableau) showing key word counts related to outages pertaining to network congestion.
![image](https://github.com/MaxSydow/Resume-Project-List-Summary/assets/56166497/2f1d0fc7-47c5-4a36-920b-424e967d1d76)

[Back to top](#Resume-Project-List-Summary)
