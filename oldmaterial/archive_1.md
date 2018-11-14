

2.2. Linear model: from regression to sparsity

Diabetes dataset

The diabetes dataset consists of 10 physiological variables (age, sex, weight, blood pressure) measure on 442 patients, and an indication of disease progression after one year:

>>> diabetes = datasets.load_diabetes()
>>> diabetes_X_train = diabetes.data[:-20]
>>> diabetes_X_test  = diabetes.data[-20:]
>>> diabetes_y_train = diabetes.target[:-20]
>>> diabetes_y_test  = diabetes.target[-20:]
The task at hand is to predict disease prediction from physiological variables.

2.2.1. Linear regression

_images/plot_ols_1.png
Linear models: y = X\beta + \epsilon

X: data
y: target variable
\beta: Coefficients
\epsilon: Observation noise
>>> from scikits.learn import linear_model
>>> regr = linear_model.LinearRegression()
>>> regr.fit(diabetes_X_train, diabetes_y_train)
LinearRegression(fit_intercept=True)
>>> print regr.coef_
[  3.03499549e-01  -2.37639315e+02   5.10530605e+02   3.27736980e+02
  -8.14131709e+02   4.92814588e+02   1.02848452e+02   1.84606489e+02
   7.43519617e+02   7.60951722e+01]

>>> # The mean square error
>>> np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2)
2004.5676026898223

>>> # Explained variance score: 1 is perfect prediction
>>> regr.score(diabetes_X_test, diabetes_y_test)
0.58507530226905713



2.2.4.1. Excercice: classification of digits\documentclass[MASTER.tex]{subfiles} 
\begin{document} 

\textbf{Classification with scikit-learn}
\begin{itemize}
\item In this segment, we
 look into the problem of classification, a situation in which a response is a categorical variable. 
% \item We will build upon the techniques that we previously discussed in the context of regression and show how they can be transferred to classification problems. 
\item We will introduces a number of classification techniques, and 
convey their corresponding strengths and weaknesses by visually inspecting the decision boundaries for each model.
\item Here we will use \textbf{scikit-learn}, an easy-to-use, general-purpose toolbox for machine learning in Python
%	\item This is part of a series of blog posts showing how to do common statistical learning techniques with Python. 
%	\item We provide only a small amount of background on the concepts and techniques we cover, so if you’d like a more thorough explanation check out Introduction to Statistical Learning or sign up for the free online course run by the book’s authors here.
	\end{itemize}


%===========================================================%

%===========================================================%

	\Large
	\textbf{Scikit-learn}\\
	\begin{itemize}
	\item Scikit-learn is a library that provides a variety of both supervised and unsupervised machine learning techniques. 
	\item Supervised machine learning refers to the problem of inferring a function from labeled training data, and it comprises both regression and classification. 
	\end{itemize}



%===========================================================%

	\Large
	
\textbf{Unspervised Learning}
	\begin{itemize}
		\item	
	Unsupervised machine learning, on the other hand, refers to the problem of finding interesting patterns or structure in the data; it comprises techniques such as clustering and dimensionality reduction.
	\item  In addition to statistical learning techniques, scikit-learn provides utilities for common tasks such as model selection, feature extraction, and feature selection.
\end{itemize}

%===========================================================%

	\LARGE
	\textbf{Estimators}
\begin{itemize}
\item Scikit-learn provides an object-oriented interface centered around the concept of an Estimator. \item According to the scikit-learn tutorial “\textit{An estimator is any object that learns from data; it may be a classification, regression or clustering algorithm or a transformer that extracts/filters useful features from raw data.}” 	
\end{itemize}



%===========================================================%

	\Large
\begin{itemize}
\item The \texttt{Estimator.fit} method sets the state of the estimator based on the training data. 
\item Usually, the data is comprised of a two-dimensional numpy array X of shape \texttt{(n\_samples, n\_predictors) }that holds the so-called feature matrix and a one-dimensional numpy array y that holds the responses. 
\item Some estimators allow the user to control the fitting behavior. 
\end{itemize}



%===========================================================%

	\Large
	\begin{itemize}
		\item For example, the \texttt{sklearn.linear\_model.LinearRegression} estimator allows the user to specify whether or not to fit an intercept term. 
\item This is done by setting the corresponding constructor arguments of the estimator object:
\end{itemize}

%===========================================================%

	\begin{figure}
\centering
\includegraphics[width=1.1\linewidth]{sklcass/sklclass1a}

\end{figure}

\large
\begin{framed}
\begin{verbatim}
from sklearn.linear_model import LinearRegression
est = LinearRegression(fit_intercept=False)
\end{verbatim}
\end{framed}

%The docstring of the estimator shows you all available arguments – in IPython simply use LinearRegression? to view the docstring.

%===========================================================%

\Large
\begin{itemize}
\item During the fitting process, the state of the estimator is stored in instance attributes that have a trailing underscore ('\_'). 
\item For example, the coefficients of a LinearRegression estimator are stored in the attribute \texttt{coef\_}:
\end{itemize}


%===========================================================%

\begin{framed}
\begin{verbatim}
import numpy as np

# random training data
X = np.random.rand(10, 2)
y = np.random.randint(2, size=10)
est.fit(X, y)
est.coef_   # access coefficients

# Output : array([ 0.33176871,  0.34910639])
\end{verbatim}
\end{framed}

%===========================================================%

	\textbf{Estimators}
\Large
\begin{itemize}
\item Estimators that can generate predictions provide a Estimator.predict method.
\item In the case of regression, Estimator.predict will return the predicted regression values; it will return the corresponding class labels in the case of classification.
\item  Classifiers that can predict the probability of class membership have a method \texttt{Estimator.predict\_proba} that returns a two-dimensional numpy array of shape \texttt{(n\_samples, n\_classes)} where the classes are lexicographically ordered.
\end{itemize}



%%===========================================================%
%
%\textbf{Estimators: Transformers}
%\begin{itemize}
%\item Finally, there is a special type of Estimator called Transformer which transforms the input data — e.g. selects a subset of the features or extracts new features based on the original ones.
%%\item In addition to a fit method, a Transformer object provides the following methods:
%\end{itemize}
%
%

%%===========================================================%
%
%\Large
%\begin{framed}
%	\begin{verbatim}
%class Transformer(Estimator):
%
%def transform(self, X):
%"""Transforms the input data. """
%# transform ``X`` to ``X_prime``
%return X_prime
%\end{verbatim}
%\end{framed}
%
%%===========================================================%
%
%	\begin{itemize}
%	\item Usually, a Transformer does not provide a predict method, but in some cases it may.
%	\item One transformer that we will use in this posting is \texttt{sklearn.preprocessing.StandardScaler}. 
%	\item This transformer centers each predictor in X to have zero mean and unit variance:
%	\end{itemize}
%
%
%===========================================================%
%
%In [6]:
%from sklearn.preprocessing import StandardScaler
%scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
%X_centered = scaler.fit(X).transform(X)
%scaler.mean_  # mean that will be subtracted upon transform
%Out[6]:
%array([ 0.48261456,  0.48636312])
%For more information on scikit-learn please consult the detailed user guide or walk through the excellent tutorial.
%

	\huge
\[ \mbox{ Classification with Scikit-Learn} \]

%===========================================================%

\textbf{Understanding Classification}\\
Although regression and classification appear to be very different they are in fact similar problems.

\begin{itemize}
\item In regression our predictions for the response are real-valued numbers
\item on the other hand, in classification the response is a mutually exclusive class label 
\item Example ``\textit{Is the email spam?}" or ``\textit{Is the credit card transaction fraudulent?}".
\end{itemize}


%===========================================================%


 	\Large
 	\textbf{Binary Classsification Problems}\\
 	\begin{itemize}
\item If the number of classes is equal to two, then we call it a binary classification problem; if there are more than two classes, then we call it a multiclass classification problem.
\item  In the following we will assume binary classification because it’s the more general case, and — we can always represent a multiclass problem as a sequence of binary classification problems.
\end{itemize}


%===========================================================%

	\Large
\textbf{Credit Card Fraud}
\begin{itemize}
\item We can also think of classification as a function estimation problem where the function that we want to estimate separates the two classes. 
\item This is illustrated in the example below where our goal is to predict whether or not a credit card transaction is fraudulent
\item he dataset is provided by James et al., \textbf{Introduction to Statistical Learning}.
\end{itemize}



%===========================================================%

	\Large
\vspace{-1cm}
\textbf{Credit Card Fraud}
\begin{figure}
\centering
\includegraphics[width=1.2\linewidth]{sklcass/sklclass1}

\end{figure}


%===========================================================%

	\Large
\textbf{Credit Card Fraud}
\begin{figure}
\centering
\includegraphics[width=0.7\linewidth]{sklcass/sklclass2}

\end{figure}



%===========================================================%

	\Large
	\textbf{Credit Card Fraud}
	\begin{itemize}
\item 	On the left you can see a scatter plot where fraudulent cases are red dots and non-fraudulent cases are blue dots. 
\item A good separation seems to be a vertical line at around a balance of 1400 as indicated by the boxplots on the next slide.
	\end{itemize}
	



%===========================================================%

	
	\begin{figure}
\centering
\includegraphics[width=0.95\linewidth]{sklcass/sklclass3}

\end{figure}

	


%======================================================%

\begin{figure}
\centering
\includegraphics[width=0.9\linewidth]{sklcass/sklclass4}

\end{figure}

%======================================================%

	
	\Large
\textbf{Simple Approach - Linear Regression}
\begin{itemize}
\item A simple approach to binary classification is to simply encode default as a numeric variable with 'Yes' == 1 and 'No' == -1; fit an Ordinary Least Squares regression model and use this model to predict the response as 'Yes' if the regressed value is higher than 0.0 and 'No' otherwise. 
\item The points for which the regression model predicts 0.0 lie on the so-called decision surface — since we are using a linear regression model, the decision surface is linear as well.
\end{itemize}



%======================================================%
%
%The example below illustrates this. Note that we use the \texttt{sklearn.linear\_model.LinearRegression} class in scikit-learn instead of the statsmodels.api.OLS class in statsmodels – they both implement the same procedure.
%
%======================================================%

\begin{figure}
\centering
\includegraphics[width=0.99\linewidth]{sklcass/sklclass6}
\end{figure}


%======================================================%

\begin{figure}
\centering
\includegraphics[width=0.99\linewidth]{sklcass/sklclass7}

\end{figure}


%======================================================%

	\begin{itemize}
	\item Points that lie on the left side of the decision boundary will be classified as negative; 
	\item Points that lie on the right side, positive. 
	\end{itemize}

%The implementation of plot_surface can be found in the Appendix. 


%======================================================%

	\Large
\textbf{Confusion Matrix}
\begin{itemize}
\item We can assess the performance of the model by looking at the confusion matrix — a cross tabulation of the actual and the predicted class labels. 

\item The correct classifications are shown in the diagonal of the confusion matrix. The off-diagonal terms show you the \textbf{classification errors}. 
\item A condensed summary of the model performance is given by the \textbf{misclassification rate} determined simply by dividing the number of errors by the total number of cases.
\end{itemize}



%======================================================%

\textbf{Confusion Matrix}\begin{figure}
\centering
\includegraphics[width=0.95\linewidth]{sklcass/sklclass8}

\end{figure}

%======================================================%

\textbf{Confusion Matrix}
\begin{figure}
\centering
\includegraphics[width=0.7\linewidth]{sklcass/sklclass9}

\end{figure}



%======================================================%

	\large
	\textbf{Cross Validation}
\begin{itemize}
\item In this example we are assessing the model performance on the same data that we used to fit the model. 
\item This might be a biased estimate of the models performance, for a classifier that simply memorizes the training data has zero training error but would be totally useless to make predictions.
\item  It is much better to assess the model performance on a separate dataset called the test data.
\item  Scikit-learn provides a number of ways to compute such held-out estimates of the model performance. \item One way is to simply split the data into a \textbf{training set} and \textbf{testing set}.
\end{itemize}


%======================================================%

\begin{figure}
\centering
\includegraphics[width=0.99\linewidth]{sklcass/sklclass10}

\end{figure}


%======================================================%

\begin{figure}
\centering
\includegraphics[width=0.7\linewidth]{sklcass/sklclass11}

\end{figure}


%======================================================%

\Large
\textbf{Classification Techniques}
\begin{itemize}
\item Different classification techniques can often be compared using the type of decision surface they can learn. \item The decision surfaces describe for what values of the predictors the model changes its predictions and it can take several different shapes: piece-wise constant, linear, quadratic, vornoi tessellation, \ldots
\end{itemize}


%======================================================%

\Large
This next part will introduce three popular classification techniques: 
\begin{itemize}
\item[1] Logistic Regression, 
\item[2] Discriminant Analysis, 
\item[3] Nearest Neighbor.
\end{itemize} We will investigate what their strengths and weaknesses are by looking at the decision boundaries they can model. In the following we will use three synthetic datasets that we adopted from this scikit-learn example.

%======================================================%

	\textbf{Synthetic Data Sets}
\begin{figure}
\centering
\includegraphics[width=0.99\linewidth]{sklcass/sklclass12}

\end{figure}

%======================================================%

\textbf{Synthetic Data Sets}
\begin{itemize}
\item The task in each of the above examples is to separate the red from the blue points. 
\item Testing data points are plotted in lighter color. 
\item The left example contains two intertwined moon sickles; the middle example is a circle of blues framed by a ring of reds; and the right example shows two linearly separable gaussian blobs.
\end{itemize}

\end{document}
