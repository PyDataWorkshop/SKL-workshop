#### OneHot Encoder

Encode categorical integer features using a one-hot aka one-of-K scheme.
The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features. 
The output will be a sparse matrix where each column corresponds to one possible value of one feature. 
It is assumed that input features take on values in the range [0, n_values).
This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.


#### Examples

Given a dataset with three features and two samples, we let the encoder find the maximum value per feature and transform the data to a binary one-hot encoding.

<pre><code>
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  

OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
       handle_unknown='error', n_values='auto', sparse=True)
enc.n_values_
array([2, 3, 4])

enc.feature_indices_
array([0, 2, 5, 9])

enc.transform([[0, 1, 1]]).toarray()
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
</code></pre>
