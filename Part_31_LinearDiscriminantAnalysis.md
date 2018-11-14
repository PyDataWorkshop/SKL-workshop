

\frametitle{Method 2: Linear Discriminant Analysis}
\Large
\textbf{Linear Discriminant Analysis}
\begin{itemize}
\item Linear discriminant Analysis (LDA) is another popular technique which shares some similarities with Logistic Regression. 
\item LDA too finds linear boundary between the two classes where points on side are classified as one class and those on the other as classified as the other class.
\end{itemize}


\begin{verbatim}

from sklearn.lda import LDA

est = LDA()
plot_datasets(est)
\end{verbatim}

\frametitle{Method 2: Linear Discriminant Analysis}
\textbf{Model Appraisal}
\begin{figure}
\centering
\includegraphics[width=0.95\linewidth]{sklcass/sklclass15}

\end{figure}
(Remark - almost same as logistic regression)


\item The major difference between LDA and Logistic Regression is the way both techniques picks the linear decision boundary.
\item  Linear Discriminant Analysis models the decision boundary by making distributional assumptions about the data generating process 
\item Logistic Regression models the probability of a sample being member of a class given its feature values.
\end{itemize}

