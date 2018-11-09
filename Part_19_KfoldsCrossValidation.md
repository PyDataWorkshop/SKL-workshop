 
Want help with statistics? Take the FREE Mini-Course
 


Start Here
Blog
Topics
Ebooks
FAQ
About
Contact

A Gentle Introduction to k-fold Cross-Validation
By Jason Brownlee on May 23, 2018 in Statistical Methods 
Tweet  Share 
   
Share
 
Cross-validation is a statistical method used to estimate the skill of machine learning models.
It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.
In this tutorial, you will discover a gentle introduction to the k-fold cross-validation procedure for estimating the skill of machine learning models.
After completing this tutorial, you will know:
That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
There are common tactics that you can use to select the value of k for your dataset.
There are commonly used variations on cross-validation such as stratified and repeated that are available in scikit-learn.
Let’s get started.

A Gentle Introduction to k-fold Cross-Validation
Photo by Jon Baldock, some rights reserved.
Tutorial Overview
This tutorial is divided into 5 parts; they are:
k-Fold Cross-Validation
Configuration of k
Worked Example
Cross-Validation API
Variations on Cross-Validation



Need help with Statistics for Machine Learning?
Take my free 7-day email crash course now (with sample code).
Click to sign-up and also get a free PDF Ebook version of the course.
Download Your FREE Mini-Course



k-Fold Cross-Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.
Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.
It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.
The general procedure is as follows:
Shuffle the dataset randomly.
Split the dataset into k groups
For each unique group: 
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
Summarize the skill of the model using the sample of model evaluation scores
Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.
This approach involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k − 1 folds.
— Page 181, An Introduction to Statistical Learning, 2013.
It is also important that any preparation of the data prior to fitting the model occur on the CV-assigned training dataset within the loop rather than on the broader data set. This also applies to any tuning of hyperparameters. A failure to perform these operations within the loop may result in data leakage and an optimistic estimate of the model skill.
Despite the best efforts of statistical methodologists, users frequently invalidate their results by inadvertently peeking at the test data.
— Page 708, Artificial Intelligence: A Modern Approach (3rd Edition), 2009.
The results of a k-fold cross-validation run are often summarized with the mean of the model skill scores. It is also good practice to include a measure of the variance of the skill scores, such as the standard deviation or standard error.
Configuration of k
The k value must be chosen carefully for your data sample.
A poorly chosen value for k may result in a mis-representative idea of the skill of the model, such as a score with a high variance (that may change a lot based on the data used to fit the model), or a high bias, (such as an overestimate of the skill of the model).
Three common tactics for choosing a value for k are as follows:
Representative: The value for k is chosen such that each train/test group of data samples is large enough to be statistically representative of the broader dataset.
k=10: The value for k is fixed to 10, a value that has been found through experimentation to generally result in a model skill estimate with low bias a modest variance.
k=n: The value for k is fixed to n, where n is the size of the dataset to give each test sample an opportunity to be used in the hold out dataset. This approach is called leave-one-out cross-validation.
The choice of k is usually 5 or 10, but there is no formal rule. As k gets larger, the difference in size between the training set and the resampling subsets gets smaller. As this difference decreases, the bias of the technique becomes smaller
— Page 70, Applied Predictive Modeling, 2013.
A value of k=10 is very common in the field of applied machine learning, and is recommend if you are struggling to choose a value for your dataset.
To summarize, there is a bias-variance trade-off associated with the choice of k in k-fold cross-validation. Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.
— Page 184, An Introduction to Statistical Learning, 2013.
If a value for k is chosen that does not evenly split the data sample, then one group will contain a remainder of the examples. It is preferable to split the data sample into k groups with the same number of samples, such that the sample of model skill scores are all equivalent.
Worked Example
To make the cross-validation procedure concrete, let’s look at a worked example.
Imagine we have a data sample with 6 observations:

1
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
The first step is to pick a value for k in order to determine the number of folds used to split the data. Here, we will use a value of k=3. That means we will shuffle the data and then split the data into 3 groups. Because we have 6 observations, each group will have an equal number of 2 observations.
For example:

1
2
3
Fold1: [0.5, 0.2]
Fold2: [0.1, 0.3]
Fold3: [0.4, 0.6]
We can then make use of the sample, such as to evaluate the skill of a machine learning algorithm.
Three models are trained and evaluated with each fold given a chance to be the held out test set.
For example:
Model1: Trained on Fold1 + Fold2, Tested on Fold3
Model2: Trained on Fold2 + Fold3, Tested on Fold1
Model3: Trained on Fold1 + Fold3, Tested on Fold2
The models are then discarded after they are evaluated as they have served their purpose.
The skill scores are collected for each model and summarized for use.
Cross-Validation API
We do not have to implement k-fold cross-validation manually. The scikit-learn library provides an implementation that will split a given data sample up.
The KFold() scikit-learn class can be used. It takes as arguments the number of splits, whether or not to shuffle the sample, and the seed for the pseudorandom number generator used prior to the shuffle.
For example, we can create an instance that splits a dataset into 3 folds, shuffles prior to the split, and uses a value of 1 for the pseudorandom number generator.

1
kfold = KFold(3, True, 1)
The split() function can then be called on the class where the data sample is provided as an argument. Called repeatedly, the split will return each group of train and test sets. Specifically, arrays are returned containing the indexes into the original data sample of observations to use for train and test sets on each iteration.
For example, we can enumerate the splits of the indices for a data sample using the created KFold instance as follows:

1
2
3
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (train, test))
We can tie all of this together with our small dataset used in the worked example of the prior section.

1
2
3
4
5
6
7
8
9
10
# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))
Running the example prints the specific observations chosen for each train and test set. The indices are used directly on the original data array to retrieve the observation values.

1
2
3
train: [0.1 0.4 0.5 0.6], test: [0.2 0.3]
train: [0.2 0.3 0.4 0.6], test: [0.1 0.5]
train: [0.1 0.2 0.3 0.5], test: [0.4 0.6]
Usefully, the k-fold cross validation implementation in scikit-learn is provided as a component operation within broader methods, such as grid-searching model hyperparameters and scoring a model on a dataset.
Nevertheless, the KFold class can be used directly in order to split up a dataset prior to modeling such that all models will use the same data splits. This is especially helpful if you are working with very large data samples. The use of the same splits across algorithms can have benefits for statistical tests that you may wish to perform on the data later.
Variations on Cross-Validation
There are a number of variations on the k-fold cross validation procedure.
Three commonly used variations are as follows:
Train/Test Split: Taken to one extreme, k may be set to 1 such that a single train/test split is created to evaluate the model.
LOOCV: Taken to another extreme, k may be set to the total number of observations in the dataset such that each observation is given a chance to be the held out of the dataset. This is called leave-one-out cross-validation, or LOOCV for short.
Stratified: The splitting of data into folds may be governed by criteria such as ensuring that each fold has the same proportion of observations with a given categorical value, such as the class outcome value. This is called stratified cross-validation.
Repeated: This is where the k-fold cross-validation procedure is repeated n times, where importantly, the data sample is shuffled prior to each repetition, which results in a different split of the sample.
The scikit-learn library provides a suite of cross-validation implementation. You can see the full list in the Model Selection API.
Extensions
This section lists some ideas for extending the tutorial that you may wish to explore.
Find 3 machine learning research papers that use a value of 10 for k-fold cross-validation.
Write your own function to split a data sample using k-fold cross-validation.
Develop examples to demonstrate each of the main types of cross-validation supported by scikit-learn.
If you explore any of these extensions, I’d love to know.
Further Reading
This section provides more resources on the topic if you are looking to go deeper.
Posts
How to Implement Resampling Methods From Scratch In Python
Evaluate the Performance of Machine Learning Algorithms in Python using Resampling
What is the Difference Between Test and Validation Datasets?
Data Leakage in Machine Learning
Books
Applied Predictive Modeling, 2013.
An Introduction to Statistical Learning, 2013.
Artificial Intelligence: A Modern Approach (3rd Edition), 2009.
API
sklearn.model_selection.KFold() API
sklearn.model_selection: Model Selection API
Articles
Resampling (statistics) on Wikipedia
Cross-validation (statistics) on Wikipedia
Summary
In this tutorial, you discovered a gentle introduction to the k-fold cross-validation procedure for estimating the skill of machine learning models.
Specifically, you learned:
That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
There are common tactics that you can use to select the value of k for your dataset.
There are commonly used variations on cross-validation, such as stratified and repeated, that are available in scikit-learn.
Do you have any questions?
Ask your questions in the comments below and I will do my best to answer.



Get a Handle on Statistics for Machine Learning!

Develop a working understanding of statistics
…by writing lines of code in python
Discover how in my new Ebook:
Statistical Methods for Machine Learning
It provides self-study tutorials on topics like:
Hypothesis Tests, Correlation, Nonparametric Stats, Resampling, and much more…
Discover how to Transform Data into Knowledge
Skip the Academics. Just Results.
Click to learn more.


Tweet  Share 
   
Share
 


About Jason Brownlee
Jason Brownlee, PhD is a machine learning specialist who teaches developers how to get results with modern machine learning methods via hands-on tutorials. 
View all posts by Jason Brownlee → 


 How to Transform Data to Better Fit The Normal Distribution
A Gentle Introduction to the Bootstrap Method 

36 Responses to A Gentle Introduction to k-fold Cross-Validation

Kristian Lunow Nielsen May 25, 2018 at 4:30 pm # 
Hi Jason
Nice gentle tutorial you have made there!
I have a more technical question; Can you comment on why the error estimate obtained through k-fold-cross-validation is almost unbiased? with an emphasis on why. 
I have had a hard time finding literature describing why.
It is my understanding that everyone comments on the bias/variance trade-off when asked about the almost unbiased feature of k-fold-cross-validation.
Reply 

Jason Brownlee May 26, 2018 at 5:48 am # 
Thanks.
Good question. 
We repeat the model evaluation process multiple times (instead of one time) and calculate the mean skill. The mean estimate of any parameter is less biased than a one-shot estimate. There is still some bias though.
The cost is we get variance on this estimate, so it’s good to report both mean and variance or mean and stdev of the score.
Reply 

Vladislav Gladkikh May 25, 2018 at 7:25 pm # 
Another possible extension: stratified cross-validation for regression. It is not directly implemented in Scikit-learn, and there is discussion if it worth implementing or not: https://github.com/scikit-learn/scikit-learn/issues/4757 but this is exactly what I need in my work. I do it like this:


1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
from sklearn.model_selection import StratifiedKFold
import numpy as np
 
n_splits = 3
 
X = np.ones(10)
y = np.arange(1,11,dtype=float)
 
# binning to make StratifiedKFold work
yc = np.outer(y[::n_splits],np.ones(n_splits)).flatten()[:len(y)]
yc[-n_splits:]=yc[-n_splits]*np.ones(n_splits)
 
skf = StratifiedKFold(n_splits=n_splits)
for train, test in skf.split(X, yc):
    print("train: %s test: %s" % (train, test))

Reply 

Vladislav Gladkikh May 25, 2018 at 7:26 pm # 
How to make code formatting here?
Reply 

Jason Brownlee May 26, 2018 at 5:53 am # 
You can use PRE HTML tags. I formatted your code for you.
Reply 

Jason Brownlee May 26, 2018 at 5:53 am # 
Thanks for sharing!
Reply 

hayet May 29, 2018 at 10:46 pm # 
Should be used k cross-validation in deep learning?
Reply 

Jason Brownlee May 30, 2018 at 6:44 am # 
It can be for small networks/datasets.
Often it is too slow.
Reply 

Chan June 8, 2018 at 9:45 pm # 
Dear Jason,
Thanks for this insight ,especially the worked example section. It’s very helpful to understand the fundamentals. However, I have a basic question which I didn’t understand completely.
If we throw away all the models that we learn from every group (3 models in your example shown), what would be the final model to predict unseen /test data? 
Is it something like:
We are using cross-validation only to choose the right hyper-parameter for a model? say K for KNN.
1. We fix a value of K;train and cross-validate to get three different models with different parameters (/coefficients like Y=3x+2; Y=2x+3; Y=2.5X+3 = just some random values)
2. Every model has its own error rate. Average them out to get a mean error rate for that hyper-parameter setup / values
3. Try with other values of Hyper-parameters (step 1 and 2 repetitively for all set of hyper-parameter values)
4. Choose the hyper-parameter set with the least average error
5. Train the whole training data set (without any validation split this time) with new value of hyper-parameter and get the new model [Y=2.75X+2.5 for eg.,]
6. Use this as a model to predict the new / unseen / test data. Loss value would be the final error from this model
Is this the way? or May be I understood it completely wrong.
Sorry for this naive question as I’m quite new or just a started. Thanks for your understanding 
Reply 

Jason Brownlee June 9, 2018 at 6:52 am # 
I explain how to develop a final model here:
https://machinelearningmastery.com/train-final-machine-learning-model/
Reply 

teja_chebrole June 21, 2018 at 9:40 pm # 
awesome article..very useful…
Reply 

Jason Brownlee June 22, 2018 at 6:06 am # 
I’m glad it helped.
Reply 

M.sarat chandra July 7, 2018 at 5:32 pm # 
if loocv is done it increase the size of k as datasets increase size .what would u say abt this.
when to use loocv on data. what is use of pseudo random number generator.
Reply 

Jason Brownlee July 8, 2018 at 6:17 am # 
In turn it increases the number of models to fit and the time it will take to evaluate.
The choice of random numbers does not matter as long as you are consistent in your experiment.
Reply 

marison July 10, 2018 at 4:20 pm # 
hi,
1. can u plz provide me a code for implementing the k-fold cross validation in R ?
2. do we have to do cross validation on complete data set or only on the training dataset after splitting into training and testing dataset?
Reply 

Jason Brownlee July 11, 2018 at 5:52 am # 
Here is an example in R:
http://machinelearningmastery.com/evaluate-machine-learning-algorithms-with-r/
It is your choice how to estimate the skill of a model on unseen data.
Reply 

Zhian July 16, 2018 at 7:36 pm # 
Hello, 
Thank you for the great tutorial. I have one question regarding the cross validation for the data sets of dynamic processes. How one could do cross validation in this case? Assume we have 10 experiments where the state of the system is the quantity which is changing in time (initial value problem). I am not sure here one should shuffle the data or not. Shall I take the whole one experiment as a set for cross validation or choose a part of every experiment for that purpose? every experiment contain different features which control the state of the system. When I want to validate I would like to to take the initial state of the system and with the vector of features to propagate the state in time. This is exactly what I need in practice. 
Could you please provide me your comments on that. I hope I am clear abot my issue.
Thanks.
Reply 

Jason Brownlee July 17, 2018 at 6:16 am # 
You could use walk-forward validation:
https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
Reply 

Tamara August 8, 2018 at 5:29 am # 
Hi Jason,
Firstly, your tutorials are excellent and very helpful. Thank you so much!
I have a question related to the use of k-fold cross-validation (k-fold CV) in testing the validity of a neural network model (how well it performs for new data). I’m afraid there is some confusion in this field as k-fold CV appears to be required for justifying any results.
So far I understand we can use k-fold CV to find optimal parameters while defining the network (as accuracy for train and test data will tell when it is over or under fitting) and we can make the choices that ensure good performance. Once we made these choices we can run the algorithm for the entire training data and we generate a model. This model has to be then tested for new data (validation set and training set). My question is: on how many new data sets has this model to be tested din order to be considered useful?
Since we have a model, using again k-fold CV does not help (we do not look for a new model). I my understanding the k-fold CV testing is mainly for the algorithm/method optimization while the final model should be only tested on new data. Is this correct? if so, should I split the test data into smaller sets, and use these as multiple tests, or using just the one test data set is enough?
Many thanks,
Tamara
Reply 

Jason Brownlee August 8, 2018 at 6:25 am # 
Often we split the training dataset into train and validation and use the validation to tune hyperparameters.
Perhaps this post will help:
https://machinelearningmastery.com/difference-test-validation-datasets/
Reply 

ashish August 14, 2018 at 7:21 pm # 
Hi jason , thanks for a nice blog 
my dataset size is 6000 (image data). how do we know which type of cross validation should use (simply train test split or k- fold cross validation) .
Reply 

Jason Brownlee August 15, 2018 at 5:58 am # 
Start with 10-folds.
Reply 

Carlos August 16, 2018 at 2:46 am # 
Good morning!
I am an Economics Student at University of São Paulo and I am researching about Backtesting, Stress Test and Validation Models to Credit Risk. Thus, would you help me answering some questions? I researching how to create a good procedure to validate prediction models that tries to forecast default behavior of the agents. Thereby, suppose a log-odds logit model of Default Probability that uses some explanatory variables as GDP, Official Interest Rates, etc. In order to evaluate it, I calculate the stability and the backtesting, using part of my data not used in the estimation with this purpose. In the backtesting case, I use a forecast, based on the regression of relevant variables to perceive if my model is corresponding to the forecast that has interval of confidence to evaluate if they are in or out. Furthermore, I evaluate the signal of the parameters to verify if it is beavering according to the economic sense.
After reading some papers, including your publication here and a Basel one (“Sound Practices for Backtesting Counterparty Credit Risk Models”), I have some doubts.
1) Do a pattern backtesting procedure lead completely about the overfitting issue? If not, which the recommendations to solve it?
2) What are the issues not covered by a pattern backtesting procedure and we should pay attention using another metrics to lead with them?
3) Could you indicate some paper or document that explains about Back-pricing, conception introduced by “Sound Practices for Backtesting Counterparty Credit Risk Models”? I have not found another document and I had not understood their explanation.
“A bank can carry out additional validation work to support the quality of its models by carrying out back-pricing. Back-pricing, which is similar to backtesting, is a quantitative comparison of model predictions with realizations, but based on re-running current models on historical market data. In order to make meaningful statements about the performance of the model, the historical data need to be divided into distinct calibration and verification data sets for each initialization date, with the model calibrated using the calibration data set before the initialization date and the forecasts after initialization tested on the verification data sets. This type of analysis helps to inform the effectiveness of model remediation, ie by demonstrating that a change to the model made in light of recent experience would have improved past and present performance. An appropriate back-pricing allows extending the backtesting data set into the past.” 
Thus, I appreciate your attention and help.
The best regards.
Reply 

Jason Brownlee August 16, 2018 at 6:12 am # 
Too much for one comment, sorry. One small question at a time please.
You can get started with back-testing procedures for time series forecasting here:
https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
Reply 

Scott Miller September 6, 2018 at 11:48 pm # 
Hi Jason, I’m using k-fold with regularized linear regression (Ridge) with the objective to determine the optimial regularization parameter. 
For each regularization parameter, I do k-fold CV to compute the CV error.
I then select the regularization parmeter that achieves the lowest CV error. 
However, in k-fold when I use ‘shuffle=True’ AND no ‘random_state’ in k-fold, the optimal regularization parameter changes each time I run the program.
kf=KFold(n_splits=n_kfolds, shuffle=True)
If I use a random state or ‘shuffle = False’, the results are always the same. 
Question: Do you feel this is normal behavior and any recommendations.
note: Predictions are really good, just looking for general discussion.
Thanks.
Reply 

Jason Brownlee September 7, 2018 at 8:06 am # 
Yes, it might be a good idea to repeat each experiment to counter the variance of the model.
Going even one step further, you might even want to use statistical tests to help determine whether “better” is real or noise. I have tutorials on this under the topic of statistics I believe.
Reply 

Pascal Schmidt October 4, 2018 at 1:35 pm # 
Hi Jason, 
thank you for the great tutorial. It helped me a lot to understand cross-validation better.
There is one concept I am still unsure about and I was hoping you could answer this for me please.
When I do feature selection before cross validation then my error will be biased because I chose the features based on training and testing set (data leakage). Therefore, I believe I have to do feature selection inside the cross validation loop with only the training data and then test my model on the test data. 
So my question is when I end up with different predictors for the different folds, should I choose the predictors that occured the majority of the time? And after that, should I do cross validation for this model with the same predictors? So, do k-fold cv with my final model where every predictor is the same for the different folds? And then use this estimate to be my cv error? 
It would be really great if you could help me out. Thanks again for the article and keep up the great work.
Reply 

Jason Brownlee October 4, 2018 at 3:30 pm # 
Thanks.
Correct. Yes, you will get different features, and perhaps you can take the average across the findings from each fold.
Alternately, you can use one hold out dataset to choose features, and a separate set for estimating model performance/tuning.
It comes down to how much data you have to “spend” and how much leakage/bias you can handle. We almost never have enough data to be pure.
Reply 

Pascal Schmidt October 6, 2018 at 3:32 am # 
Thanks, Jason. I guess statistics is not as black and white as a discipline like mathematics. A lot of different ways to deal with problems and no one best solution exists. This makes it so challenging I feel. A lot of experience is required to deal with all these unique data sets.
Reply 

Jason Brownlee October 6, 2018 at 5:50 am # 
Yes, the best way to get good is to practice, like programming, driving, and everything else we want to do in life.
Reply 

Bilal October 16, 2018 at 6:16 pm # 
for which purpose we calculate the standard deviation from any data set.
Reply 

Jason Brownlee October 17, 2018 at 6:47 am # 
In what context?
Reply 

Leontine Ham October 16, 2018 at 9:21 pm # 
Thank you for explaining the fundamentals of CV.
I am working with repeated (50x) 5-fold cross validation, but I am trying to figure out which statistical test I can use in order to compare two datasets. Can you help me? Or is that out of the scope of this blog?
Reply 

Jason Brownlee October 17, 2018 at 6:50 am # 
Yes, see this post:
https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
Reply 

kingshuk October 22, 2018 at 1:27 am # 
Hi Jason ,
What is the difference between Kfold and Stratified K fold?
Reply 

Jason Brownlee October 22, 2018 at 6:21 am # 
Kfold uses random split of the into k folds.
Stratified tries to maintain the same distribution of the target variable when randomly selecting examples for each fold.
Reply 
Leave a Reply 


Name (required) 
Email (will not be published) (required) 
Website
 
Welcome to Machine Learning Mastery!

Hi, I'm Jason Brownlee, PhD 
I write tutorials to help developers (like you) get results with machine learning.
Read More



Statistics for Machine Learning
Understand statistics by writing code in Python.

Click to Get Started Now!


Popular
How to Develop a Deep Learning Photo Caption Generator from Scratch 
November 27, 2017

How to Develop a Neural Machine Translation System from Scratch 
January 10, 2018

Difference Between Classification and Regression in Machine Learning 
December 11, 2017

How to Develop a Word-Level Neural Language Model and Use it to Generate Text 
November 10, 2017

How to Develop an N-gram Multichannel Convolutional Neural Network for Sentiment Analysis 
January 12, 2018

So, You are Working on a Machine Learning Problem… 
April 4, 2018

Encoder-Decoder Models for Text Summarization in Keras 
December 8, 2017

You might also like…
How to Install Python for Machine Learning
Your First Machine Learning Project in Python
Your First Neural Network in Python
Your First Classifier in Weka
Your First Time Series Forecasting Project
© 2018 Machine Learning Mastery. All Rights Reserved. 

Privacy | Disclaimer | Terms | Contact 


Your Start in Machine Learning
 
×
Your Start in Machine Learning
You can master applied Machine Learning 
without the math or fancy degree.
Find out how in this free and practical email course.
 

 I consent to receive information about services and special offers by email. For more information, see the Privacy Policy. 
 
