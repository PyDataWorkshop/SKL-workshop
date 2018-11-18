#### Estimators objects: Fitting data:

The core object of scikit-learn is the estimator object. 

All estimator objects expose a``fit`` method, that takes as input a dataset (2D array):

<pre><code>
estimator.fit(data)
<\code><\pre>

Suppose``LogReg`` and``KNN`` are (shorthand names for) scikit-learn estimators.

<pre><code>
# Supervised Learning Problem
LogReg.fit(SAheartFeat, SAheartTarget)

# Unsupervised Learning Problem
KNN.fit(IrisFeat)
<\code><\pre>

#### Estimator parameters:

All the parameters of an estimator can be set when it is instanciated, or by modifying the corresponding attribute:

<pre><code>
estimator = Estimator(param1=1, param2=2)
estimator.param1
<\code><\pre>

#### Retrieving Estimator parameters: 

* When data is fitted with an estimator, parameters are estimated from the data at hand.
* All the estimated parameters are attributes of the estimator object ending by an underscore:

<pre><code>
estimator.estimated_param_ 
<\code><\pre>
