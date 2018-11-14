2.2.2. Shrinkage

If there are few data points per dimension, noise in the observations induces high variance:

_images/plot_ols_variance_1.png
>>> X = np.c_[ .5, 1].T
>>> y = [.5, 1]
>>> test = np.c_[ 0, 2].T
>>> regr = linear_model.LinearRegression()

>>> import pylab as pl
>>> pl.figure() 

>>> np.random.seed(0)
>>> for _ in range(6): 
...    this_X = .1*np.random.normal(size=(2, 1)) + X
...    regr.fit(X, y)
...    pl.plot(test, regr.predict(test))
...    pl.scatter(this_X, y, s=3)
A solution, in high-dimensional statistical learning, is to srhink the regression coefficients to zero: any two randomly chosen set of observations are likely to be uncorrelated. This is called ridge regression:

_images/plot_ridge_variance_1.png
>>> regr = linear_model.Ridge(alpha=.1)

>>> pl.figure() 

>>> np.random.seed(0)
>>> for _ in range(6): 
...    this_X = .1*np.random.normal(size=(2, 1)) + X
...    regr.fit(this_X, y)
...    pl.plot(test, regr.predict(test))
...    pl.scatter(this_X, y, s=3)
This is an example of bias/variance tradeoff: the larger the ridge alpha parameter, the higher the bias and the lower the variance.

We can choose alpha to minimize left out error, this time using the diabetes dataset, rather than our synthetic data:

>>> alphas = np.logspace(-4, -1, 6)
>>> print [regr.fit(diabetes_X_train, diabetes_y_train, alpha=alpha
...             ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
[0.58511106838835292, 0.58520730154446743, 0.58546775406984897, 0.58555120365039137, 0.58307170855541623, 0.570589994372801]
Note Capturing in the fitted parameters noise that prevents the model to generalize to new data is called overfitting. The bias introduced by the ridge regression is called a regularization.
2.2.3. Sparsity

Fitting only features 5 and 6

diabetes_ols_diag diabetes_ols_x2 diabetes_ols_x1

Note A representation of the full diabetes dataset would involve 11 dimensions (10 feature dimensions, and one of the target variable). It is hard to develop an intuition on such representation, but it may be useful to keep in mind that it would be a fairly empty space.
We can see that although feature 2 has a strong coefficient on the full model, it conveys little information on y when considered with feature 1.

To improve the conditioning of the problem (mitigate the curse of dimensionality), it would be interesting to select only the informative features and set non-informative ones, like feature 2 to 0. Ridge regression will decrease their contribution, but not set them to zero. Another penalization approach, called Lasso, can set some coefficients to zero. Such methods are called sparse method, and sparsity can be seen as an application of Occam’s razor: prefer simpler models.

>>> regr = linear_model.Lasso(alpha=.1)
>>> print [regr.fit(diabetes_X_train, diabetes_y_train, alpha=alpha
...             ).score(diabetes_X_test, diabetes_y_test)
...        for alpha in alphas]
[0.5851191069162196, 0.58524713649060311, 0.58571895391793782, 0.58730094854527282, 0.5887622418309254, 0.58284500296816755]

>>> best_alpha = alphas[4]
>>> regr.fit(diabetes_X_train, diabetes_y_train, alpha=best_alpha)
Lasso(precompute='auto', alpha=0.025118864315095794, max_iter=1000,
   tol=0.0001, fit_intercept=True)
>>> print regr.coef_
[   0.         -212.43764548  517.19478111  313.77959962 -160.8303982    -0.
 -187.19554705   69.38229038  508.66011217   71.84239008]
Different algorithms for a same problem

Different algorithms can be used to solve the same mathematical problem. For instance the Lasso object in the scikit-learn solves the lasso regression using a coordinate descent method, that is efficient on large datasets. However, the scikit-learn also provides the LassoLARS object, using the LARS which is very efficient for problems in which the weight vector estimated is very sparse, that is problems with very few observations.
