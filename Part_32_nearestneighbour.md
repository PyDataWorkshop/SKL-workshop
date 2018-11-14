
%======================================================%

	\frametitle{Method 3: Nearest Neighbor}
\Large
\textbf{Nearest Neighbor}
\begin{itemize}
\item Nearest Neighbor uses the notion of similarity to assign class labels; it is based on the smoothness assumption that points which are nearby in input space should have similar outputs.
\item  It does this by specifying a similarity (or distance) metric, and at prediction time it simply searches for the k most similar among the training examples to a given test example.
\end{itemize}


%======================================================%

		\frametitle{Method 3: Nearest Neighbor}
	\Large
	\textbf{Nearest Neighbor}
	\begin{itemize} 
		\item The prediction is then either a majority vote of those k training examples or a vote weighted by similarity. \item The parameter k specifies the smoothness of the decision surface.\item  The decision surface of a k-nearest neighbor classifier can be illustrated by the \textbf{Voronoi tesselation} of the training data, that show you the regions of constant respones.
\end{itemize}

%======================================================%

		\frametitle{Method 3: Nearest Neighbor}
	\begin{figure}
		\centering
		\includegraphics[width=1.1\linewidth]{sklcass/sklclass16}
	
	\end{figure}
	

%======================================================%

		\frametitle{Method 3: Nearest Neighbor}
		\Large
	\textbf{Nearest Neighbours}
	\begin{itemize}
\item 	Yet Nearest Neighbor differs fundamentally from the above models in that it is a so-called non-parametric technique: the number of parameters of the model can grow infinitely as the size of the training data grows. 
\item Furthermore, it can model non-linear decision boundaries, something that is important for the first two datasets: moons and circles.
	\end{itemize}





%======================================================%

	\frametitle{Method 3: Nearest Neighbor}
	\large
	\begin{framed}
		\begin{verbatim}
	from sklearn.neighbors import KNeighborsClassifier
	
	est = KNeighborsClassifier(n_neighbors=1)
	plot_datasets(est)
		\end{verbatim}
	\end{framed}


	\begin{figure}
		\centering
		\includegraphics[width=1.1\linewidth]{sklcass/sklclass16}
		

	\begin{itemize}
\item If we increase k we enforce the smoothness assumption. 
\item This can be seen by comparing the decision boundaries in the plots below where k=5 to those above where k=1.
	\end{itemize}
	{
		\large
	\begin{framed}
	\begin{verbatim}
	est = KNeighborsClassifier(n_neighbors=5)
	plot_datasets(est)
	
	\end{verbatim}


\begin{figure}
\centering
\includegraphics[width=1.1\linewidth]{sklcass/sklclass20}

\end{figure}


