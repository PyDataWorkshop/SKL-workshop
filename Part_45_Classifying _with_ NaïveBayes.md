### Classifying documents with Naïve Bayes
Naïve Bayes is a really interesting model. It's somewhat similar to k-NN in the sense that it
makes some assumptions that might oversimplify reality, but still perform well in many cases.

#### Preparation} % %Getting ready
In this recipe, we'll use Naïve Bayes to do document classification with sklearn. An example
I have personal experience of is using the words that make up an account descriptor in
accounting, such as Accounts Payable, and determining if it belongs to Income Statement,
Cash Flow Statement, or Balance Sheet.

The basic idea is to use the word frequency from a labeled test corpus to learn the classifications of the documents. Then, we can turn this on a training set and attempt to
predict the label. 
We'll use the newgroups dataset within sklearn to play with the Naïve Bayes model. It's a
nontrivial amount of data, so we'll fetch it instead of loading it. We'll also limit the categories
to rec.autos and rec.motorcycles:
<pre><code>

from sklearn.datasets import fetch_20newsgroups
categories = ["rec.autos", "rec.motorcycles"]
newgroups = fetch_20newsgroups(categories=categories)
#take a look
print "\n".join(newgroups.data[:1])
From: gregl@zimmer.CSUFresno.EDU (Greg Lewis)
Subject: Re: WARNING.....(please read)...
Keywords: BRICK, TRUCK, DANGER
Nntp-Posting-Host: zimmer.csufresno.edu
Organization: CSU Fresno
Lines: 33

[…]
newgroups.target_names[newgroups.target[:1]]
'rec.autos'
<\code><\pre>
Now that we have newgroups, we'll need to represent each document as a bag of words. This representation is what gives Naïve Bayes its name. The model is "naive" because documents
are classified without regard for any intradocument word covariance. This might be considered a flaw, but Naïve Bayes has been shown to work reasonably well.
We need to preprocess the data into a bag-of-words matrix. This is a sparse matrix that has entries when the word is present in the document. This matrix can become quite large,
as illustrated:

<pre><code>

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
bow = count_vec.fit_transform(newgroups.data)
<\code><\pre>

This matrix is a sparse matrix, which is the length of the number of documents by each word.
The document and word value of the matrix are the frequency of the particular term:
<pre><code>

bow
<1192x19177 sparse matrix of type '<type 'numpy.int64'>'
with 164296 stored elements in Compressed Sparse Row format>
<\code><\pre>

We'll actually need the matrix as a dense array for the Naïve Bayes object. So, let's convert
it back:
<pre><code>
	
bow = np.array(bow.todense())
<\code><\pre>

Clearly, most of the entries are 0, but we might want to reconstruct the document counts as
a sanity check:
<pre><code>
	
words = np.array(count_vec.get_feature_names())
words[bow[0] > 0][:5]
array([u'10pm', u'1qh336innfl5', u'33', u'93740', u'_____________________
______________________________________________'],
dtype='<U79')
<\code><\pre>

Now, are these the examples in the first document? Let's check that using the
following command:
<pre><code>
	
'10pm' in newgroups.data[0].lower()
True
'1qh336innfl5' in newgroups.data[0].lower()
True
<\code><\pre>

%=======================================================================================================%

### Implementation
Ok, so it took a bit longer than normal to get the data ready, but we're dealing with text data
that isn't as quickly represented as a matrix as the data we're used to.

However, now that we're ready, we'll fire up the classifier and fit our model:

<pre><code>
from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()
Before we fit the model, let's split the dataset into a training and test set:
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

### There's more…
We can also extend Naïve Bayes to do multiclass work. Instead of assuming a Gaussian
likelihood, we'll use a multinomial likelihood.
First, let's get a third category of data:
from sklearn.datasets import fetch_20newsgroups
mn_categories = ["rec.autos", "rec.motorcycles",
"talk.politics.guns"]
mn_newgroups = fetch_20newsgroups(categories=mn_categories)

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
since one will guess that the talk.politics.guns category is fairly orthogonal to the
other two, we should probably do pretty well.
