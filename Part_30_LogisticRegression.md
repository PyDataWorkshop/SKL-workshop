### Logistic Regression
* Logistic regression can be viewed as an extension of linear regression to classification problems. 
* One of the limitations of linear regression is that it cannot provide class probability estimates. 
* This is often useful, for example, when we want to inspect manually the most fraudulent cases. 
* Basically, we would like to constrain the predictions of the model to the range $[0,1]$ so that we can interpret them as probability estimates. 
* In Logistic Regression, we use the logit function to clamp predictions from the range $[−infty,infty]$ to $[0,1]$. 


\includegraphics[width=0.7\linewidth]{sklcass/sklclass13}

\end{figure}



* Logistic regression is available in scikit-learn via the class \texttt{sklearn.linear\_model.LogisticRegression}. 
* It uses liblinear, so it can be used for problems involving millions of samples and hundred of thousands of predictors. 
* Lets see how Logistic Regression does on our three toy datasets.

<pre><code>
from sklearn.linear_model import LogisticRegression

est = LogisticRegression()
plot_datasets(est)
</code></pre>

* As we can see, a linear decision boundary is not a poor approximation for the moon datasets, although we fail to separate the two tips of the sickles in the center. 
* The cicles dataset, on the other hand, is not well suited for a linear decision boundary. 
* The error rate of 0.68 is in fact worse than random guessing. 
* For the linear dataset we picked in fact the correct model class — the error rate of 10\% is due to the noise component in our data. 
* The gradient shows you the probability of class membership — white shows you that the model is very uncertain about its prediction.



