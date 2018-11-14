
%=======================================================================================================%
% % Classifying Data with scikit-learn

### Implementation

Ok, so it took a bit longer than normal to get the data ready, but we're dealing with text data that isn't as quickly represented as a matrix as the data we're used to.
However, now that we're ready, we'll fire up the classifier and fit our model:
<pre><code>
from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
</code></pre>
Before we fit the model, let's split the dataset into a training and test set:
<pre><code>
mask = np.random.choice([True, False], len(bow))
clf.fit(bow[mask], newgroups.target[mask])
predictions = clf.predict(bow[~mask])
</code></pre>
Now that we fit a model on a test set, and then predicted the training set in an attempt to
determine which categories go with which articles, let's get a sense of the approximate
accuracy:
np.mean(predictions == newgroups.target[~mask])
0.92446043165467628
### Theoretical Background
The fundamental idea of how Naïve Bayes works is that we can estimate the probability of
some data point being a class, given the feature vector.
This can be rearranged via the Bayes formula to give the MAP estimate for the feature vector.
This MAP estimate chooses the class for which the feature vector's probability is maximized.
There's more…

We can also extend Naïve Bayes to do multiclass work. Instead of assuming a Gaussian
likelihood, we'll use a multinomial likelihood.
First, let's get a third category of data:
<pre><code>
from sklearn.datasets import fetch_20newsgroups
mn_categories = ["rec.autos", "rec.motorcycles",
"talk.politics.guns"]
mn_newgroups = fetch_20newsgroups(categories=mn_categories)
</code></pre>
%157
We'll need to vectorize this just like the class case:
<pre><code>

mn_bow = count_vec.fit_transform(mn_newgroups.data)
mn_bow = np.array(mn_bow.todense())
<\code><\pre>

Let's create a mask array to train and test:
<pre><code>

mn_mask = np.random.choice([True, False], len(mn_newgroups.data))
multinom = naive_bayes.MultinomialNB()
multinom.fit(mn_bow[mn_mask], mn_newgroups.target[mn_mask])
mn_predict = multinom.predict(mn_bow[~mn_mask])
np.mean(mn_predict == mn_newgroups.target[~mn_mask])
0.96594778660612934
<\code><\pre>

It's not completely surprising that we did well. We did fairly well in the dual class case, and
since one will guess that the ``talk.politics.guns`` category is fairly orthogonal to the other two, we should probably do pretty well.
