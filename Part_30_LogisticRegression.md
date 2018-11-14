


%======================================================%

		\frametitle{Method 1: Logistic Regression}
\textbf{Logistic Regression}
\begin{itemize}
\item Logistic regression can be viewed as an extension of linear regression to classification problems. \item One of the limitations of linear regression is that it cannot provide class probability estimates. 
\item This is often useful, for example, when we want to inspect manually the most fraudulent cases. 
\item Basically, we would like to constrain the predictions of the model to the range $[0,1]$ so that we can interpret them as probability estimates. 
\item In Logistic Regression, we use the logit function to clamp predictions from the range $[−infty,infty]$ to $[0,1]$. 
\end{itemize}


%======================================================%

		\frametitle{Method 1: Logistic Regression}
	\textbf{Logistic Transformation}
	\begin{figure}
\centering
\includegraphics[width=0.7\linewidth]{sklcass/sklclass13}

\end{figure}


%======================================================%

		\frametitle{Method 1: Logistic Regression}

\begin{itemize}
\item Logistic regression is available in scikit-learn via the class \texttt{sklearn.linear\_model.LogisticRegression}. 
%\item It uses liblinear, so it can be used for problems involving millions of samples and hundred of thousands of predictors. 
\item Lets see how Logistic Regression does on our three toy datasets.
\end{itemize}



		\begin{verbatim}
from sklearn.linear_model import LogisticRegression

est = LogisticRegression()
plot_datasets(est)
	\end{verbatim}

	\textbf{Model Appraisal}
\begin{itemize}
\item	As we can see, a linear decision boundary is not a poor approximation for the moon datasets, although we fail to separate the two tips of the sickles in the center. 
\item The cicles dataset, on the other hand, is not well suited for a linear decision boundary. 

		
		\item The error rate of 0.68 is in fact worse than random guessing. \item For the linear dataset we picked in fact the correct model class — the error rate of 10\% is due to the noise component in our data. 
\item The gradient shows you the probability of class membership — white shows you that the model is very uncertain about its prediction.
\end{itemize}



%======================================================%
