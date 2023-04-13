# ML-For-Fraud-Transaction-Detection

The dataset provided in a CSV file contains transactions made by credit cards. It has a total of 30 different feature columns. Features P1, P2, ... P28 are principal components obtained by applying PCA on the original data set. The other two non PCA features are 'Time' and 'Dollar_amount'. ‘Time' is the seconds elapsed between each transaction and the first transaction in the dataset. 'Dollar_amount' is the transaction amount. 'Outcome' is the response variable, 1 in case of fraud and 0 otherwise.

Main goal of this tasks:

1. To gain an indepth insight on the dataset through data exploration
2.  Build a classifier that generalises to unseen fraudulent transaction


This notebook file contains detailed step-by-step analysis and implmentation of a Machine Learning classifier for Fraud Transaction Detection. Steps include following

1. Initial Data Exploration using Pandas: It is very important to first understand the dataset that you have before starting to actually building the models. From this analysis we gain many interesting insights about the dataset. For example, we find that the dataset is highly imbalanced with very negligibile amount of fraudulent class samples in contrast to non-fraudulent ones. 

This means if we build a classifier without taking this class imbalnce into consideration the model will be heavily biased towards detecting non-fraudulent class. However it should be noted that one misclassification of fraud transaction would be very very costly. Thus we need to ensure that such imbalance factor is taken into account while building the model.

2. Data Cleaning and transformation

Then we also perform data cleaning. There were missing values. We used pandas to perform this cleaning. Finally two features were in higher scales so we needed to apply feature standardization. We used Scikit-Learn StandardScaler function to normalise the data (zero mean and unit variance). This way we ensure that all our feature values are in similar scale having zero mean and unit variance (an important step in machine learning).

3. Feature selection
We perform initial experiments using various combination of features. For example taking all the 28 PCA components and two other features (Time and Dollar amount) and also tried other combinations. We chose the one that shows best performance on the validation set - as our goal is to make sure model do not overfit on the training data.

4. Model building and selection
We explore two different types of ML classifiers: Logistic Regression and Decision Tree in this work. Our goal here was not to apply complex ML algorithms like Deep Neural Networks, rather our main goal was to perform an understanding of the dataset and come up with a decent model that works well in such imbalalnce setting.

5. Final thoughts

When it comes to fraud detection, it is important to minimize false negatives, i.e., identifying a transaction as legitimate when it is actually fraudulent. This is because the cost of missing a fraudulent transaction can be high and can result in financial losses for the organization as well as damage to its reputation. Therefore, we need to ensure that our fraud detection model has a high recall for the fraud class, which means it should identify as many fraudulent transactions as possible. Thus we have used Recall as our focus in chosing the model, and we have not focussed on accuracy in judging model performance (and selection).

Following are the findings from this study

* Best optimal features would be to use the first 15 PCA component features P1 to P15.
* We get Fraud class Recall of 0.89 by training a LR model on first 15 PCA features.
* Adding the Time and Dollar_amount did not impact performance. Using them did not help
* Furthermore the Scikit-Learn method SelectKBest show comparable performance with the model trained on 15 PCA features.

Although we see that SelectKBest method returns a model giving 0.88 Recall with top 9 features. These features do not seem to include P1, P2 as they are the ones with higher variance that explains the underlying data distribution. Thus we tend to go for first 15 PCA feature based LR model that showed 0.89 Fraud Recall on the test data set.
