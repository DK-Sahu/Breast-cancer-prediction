# Breast-cancer-prediction
# Introduction
* Breast cancer (BC) is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths according to global statistics, making it a significant public health problem in today’s society.
* The early diagnosis of BC can improve the prognosis and chance of survival significantly, as it can promote timely clinical treatment to patients. Further accurate     classification of benign tumors can prevent patients undergoing unnecessary treatments. Thus, the correct diagnosis of BC and classification of patients into           malignant   or benign groups is the subject of much research. Because of its unique advantages in critical features detection from complex BC datasets, machine         learning (ML) is widely recognized as the methodology of choice in BC pattern classification and forecast modelling.
* Classification and data mining methods are an effective way to classify data. Especially in medical field, where those methods are widely used in diagnosis and         analysis to make decisions.

# Recommended Screening Guidelines:
* Mammography.The most important screening test for breast cancer is the mammogram. A mammogram is an X-ray of the breast. It can detect breast cancer up to two years   before the tumor can be felt by you or your doctor.
* Women age 40–45 or older who are at average risk of breast cancer should have a mammogram once a year.
* Women at high risk should have yearly mammograms along with an MRI starting at age 30.
* Some Risk Factors for Breast Cancer The following are some of the known risk factors for breast cancer. However, most cases of breast cancer cannot be linked to a     specific cause. Talk to your doctor about your specific risk.
* Age.The chance of getting breast cancer increases as women age. Nearly 80 percent of breast cancers are found in women over the age of 50.
* Personal history of breast cancer. A woman who has had breast cancer in one breast is at an increased risk of developing cancer in her other breast.
* Family history of breast cancer. A woman has a higher risk of breast cancer if her mother, sister or daughter had breast cancer, especially at a young age (before     40). Having other relatives with breast cancer may also raise the risk.
* Genetic factors. Women with certain genetic mutations, including changes to the BRCA1 and BRCA2 genes, are at higher risk of developing breast cancer during their     lifetime. Other gene changes may raise breast cancer risk as well.
* Childbearing and menstrual history. The older a woman is when she has her first child, the greater her risk of breast cancer. Also at higher risk are:
* Women who menstruate for the first time at an early age (before 12)
* Women who go through menopause late (after age 55)
* Women who’ve never had children

# step 1: Data Preparation
* We will use the UCI Machine Learning Repository for breast cancer dataset. 
* The dataset used in this story is publicly available and was created by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison,           Wisconsin, USA. To create the dataset Dr. Wolberg used fluid samples, taken from patients with solid breast masses and an easy-to-use graphical computer program       called Xcyt, which is capable of perform the analysis of cytological features based on a digital scan. The program uses a curve-fitting algorithm, to compute ten       features from each one of the cells in the sample, than it calculates the mean value, extreme value and standard error of each feature for the image, returning a 30   real-valuated vector
# Attribute Information:
* ID number
* Diagnosis (M = malignant, B = benign)
# Ten real-valued features are computed for each cell nucleus:
* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter² / area — 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension (“coastline approximation” — 1)
* The mean, standard error and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For       instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

# Objectives:
* This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection     and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification   methods to fit a function that can predict the discrete class of new input.
# step 2: Data Exploration
* We will be using jupyter notebook to work on this dataset. We will first go with importing the necessary libraries and import our dataset to Notebook :
* importing the libraries
* We can examine the data set using the pandas’ head() method.
* We can find the dimensions of the data set using the panda dataset ‘shape’ attribute.
* We can observe that the data set contain 569 rows and 32 columns. ‘Diagnosis’ is the column which we are going to predict , which says if the cancer is M = malignant   or B = benign. 1 means the cancer is malignant and 0 means benign.
* We can identify that out of the 569 persons, 357 are labeled as B (benign) and 212 as M (malignant).
# Visualization of Dataset:
* Visualization of data is an imperative aspect of data science. It helps to understand data and also to explain the data to another person. Python has several           interesting visualization libraries such as Matplotlib, Seaborn etc.
# Missing or Null Data points:
* We can find any missing or null data points of the data set (if there is any) using the following pandas function.
   dataset.isnull().sum()
   dataset.isna().sum()
# step 3 : Categorical Data
* Categorical data are variables that contain label values rather than numeric values.The number of possible values is often limited to a fixed set.
* For example, users are typically described by country, gender, age group etc.
* We will use Label Encoder to label the categorical data. Label Encoder is the part of SciKit Learn library in Python and used to convert categorical data, or text     data, into numbers, which our predictive models can better understand.
# Splitting the dataset:
* The data we use is usually split into training data and test data. The training set contains a known output and the model learns on this data in order to be           generalized to other data later on. We have the test dataset (or subset) in order to test our model’s prediction on this subset.
* We will do this using SciKit-Learn library in Python using the train_test_split method.

# step 4 : Feature Scaling
* Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian   distance between two data points in their computations. We need to bring all features to the same level of magnitudes. This can be achieved by scaling. This means     that you’re transforming your data so that it fits within a specific scale, like 0–100 or 0–1.
* We will use StandardScaler method from SciKit-Learn library.
# step 5 : Model selection
* It is also known as Algorithm selection for Predicting the best results.
* Usually Data Scientists use different kinds of Machine Learning algorithms to the large data sets. But, at high level all those different algorithms can be             classified in two groups : supervised learning and unsupervised learning.
* Supervised learning : Supervised learning is a type of system in which both input and desired output data are provided. Input and output data are labelled for         classification to provide a learning basis for future data processing. Supervised learning problems can be further grouped into Regression and Classification           problems.
* A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”.
* A classification problem is when the output variable is a category like filtering emails “spam” or “not spam”
* Unsupervised Learning : Unsupervised learning is the algorithm using information that is neither classified nor labeled and allowing the algorithm to act on that       information without guidance.
* In our dataset we have the outcome variable or Dependent variable i.e Y having only two set of values, either M (Malign) or B(Benign). So we will use Classification   algorithm of supervised learning.
* We have different types of classification algorithms in Machine Learning :-
* Logistic Regression
* Gaussian Naive Bayes
* Support Vector Machines
* Stochastic gradient descent Classifier
* Gradient Boosting Classifier
* Decision Tree Algorithm
* Random Forest Classification
# 1. Logistic Regression:
* It is a classification not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of         independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit   regression. Since, it predicts the probability, its output values lies between 0 and 1 (as expected).
# 2. Gaussian Naive Bayes:
* It is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. In simple terms, a Naive Bayes classifier assumes       that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it     is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier       would consider all of these properties to independently contribute to the probability that this fruit is an apple.
* Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly           sophisticated classification methods.
# 3. Support Vector Machines:
* It is a classification method. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of   each feature being the value of a particular coordinate.
# 4. Stochastic gradient descent Classifier:
* Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear)     Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable     amount of attention just recently in the context of large-scale learning.
# 5. Gradient Boosting Classifier:
* GBM is a boosting algorithm used when we deal with plenty of data to make a prediction with high prediction power. Boosting is actually an ensemble of learning         algorithms which combines the prediction of several base estimators in order to improve robustness over a single estimator. It combines multiple weak or average       predictors to a build strong predictor. These boosting algorithms always work well in data science competitions like Kaggle, AV Hackathon, CrowdAnalytix.
# 6. Decision Tree Algorithm:
* It is a type of supervised learning algorithm that is mostly used for classification problems. Surprisingly, it works for both categorical and continuous dependent     variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to   make as distinct groups as possible.
# 7. Random Forest Classification:
* Random Forest is a trademark term for an ensemble of decision trees. In Random Forest, we’ve collection of decision trees (so known as “Forest”). To classify a new     object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes   (over all the trees in the forest).
* Lets start applying the algorithms :

* We will use sklearn library to import all the methods of classification algorithms.
* We will use LogisticRegression method of model selection to use Logistic Regression Algorithm,
* We will also predict the test set results and check the accuracy with each of our model:

* To check the accuracy we need to import confusion_matrix method of metrics class. The confusion matrix is a way of tabulating the number of mis-classifications, i.e., the number of predicted classes which ended up in a wrong classification bin based on the true classes.

* We will use Classification Accuracy method to find the accuracy of our models. Classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of input samples.

* finally we have built our classification model.


