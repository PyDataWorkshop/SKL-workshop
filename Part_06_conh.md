 
 Confusion Matrix in Machine Learning

 
Make the Confusion Matrix Less Confusing.
A confusion matrix is a technique for summarizing the performance of a classification algorithm.
Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.
Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.
In this post, you will discover the confusion matrix for use in machine learning.
After reading this post you will know:
What the confusion matrix is and why you need to use it.
How to calculate a confusion matrix for a 2-class classification problem from scratch.
How create a confusion matrix in Weka, Python and R.
Let’s get started.
Update Oct/2017: Fixed a small bug in the worked example (thanks Raktim).
Update Dec/2017: Fixed a small bug in accuracy calculation (thanks Robson Pastor Alexandre)

What is a Confusion Matrix in Machine Learning
Photo by Maximiliano Kolus, some rights reserved
Classification Accuracy and its Limitations
Classification accuracy is the ratio of correct predictions to total predictions made.

1
classification accuracy = correct predictions / total predictions
It is often presented as a percentage by multiplying the result by 100.

1
classification accuracy = correct predictions / total predictions * 100
Classification accuracy can also easily be turned into a misclassification rate or error rate by inverting the value, such as:

1
error rate = (1 - (correct predictions / total predictions)) * 100
Classification accuracy is a great place to start, but often encounters problems in practice.
The main problem with classification accuracy is that it hides the detail you need to better understand the performance of your classification model. There are two examples where you are most likely to encounter this problem:
When you are data has more than 2 classes. With 3 or more classes you may get a classification accuracy of 80%, but you don’t know if that is because all classes are being predicted equally well or whether one or two classes are being neglected by the model.
When your data does not have an even number of classes. You may achieve accuracy of 90% or more, but this is not a good score if 90 records for every 100 belong to one class and you can achieve this score by always predicting the most common class value.
Classification accuracy can hide the detail you need to diagnose the performance of your model. But thankfully we can tease apart this detail by using a confusion matrix.
What is a Confusion Matrix?
A confusion matrix is a summary of prediction results on a classification problem.
The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix.
The confusion matrix shows the ways in which your classification model
is confused when it makes predictions.
It gives you insight not only into the errors being made by your classifier but more importantly the types of errors that are being made.
It is this breakdown that overcomes the limitation of using classification accuracy alone.
How to Calculate a Confusion Matrix
Below is the process for calculating a confusion Matrix.
You need a test dataset or a validation dataset with expected outcome values.
Make a prediction for each row in your test dataset.
From the expected outcomes and predictions count: 
The number of correct predictions for each class.
The number of incorrect predictions for each class, organized by the class that was predicted.
These numbers are then organized into a table, or a matrix as follows:
Expected down the side: Each row of the matrix corresponds to a predicted class.
Predicted across the top: Each column of the matrix corresponds to an actual class.
The counts of correct and incorrect classification are then filled into the table.
The total number of correct predictions for a class go into the expected row for that class value and the predicted column for that class value.
In the same way, the total number of incorrect predictions for a class go into the expected row for that class value and the predicted column for that class value.
In practice, a binary classifier such as this one can make two types of errors: it can incorrectly assign an individual who defaults to the no default category, or it can incorrectly assign an individual who does not default to the default category. It is often of interest to determine which of these two types of errors are being made. A confusion matrix […] is a convenient way to display this information.
— Page 145, An Introduction to Statistical Learning: with Applications in R, 2014
This matrix can be used for 2-class problems where it is very easy to understand, but can easily be applied to problems with 3 or more class values, by adding more rows and columns to the confusion matrix.
Let’s make this explanation of creating a confusion matrix concrete with an example.
2-Class Confusion Matrix Case Study
Let’s pretend we have a two-class classification problem of predicting whether a photograph contains a man or a woman.
We have a test dataset of 10 records with expected outcomes and a set of predictions from our classification algorithm.

1
2
3
4
5
6
7
8
9
10
11
Expected, 	Predicted
man,		woman
man, 		man
woman,		woman
man,		man
woman,		man
woman, 		woman
woman, 		woman
man, 		man
man, 		woman
woman, 		woman
Let’s start off and calculate the classification accuracy for this set of predictions.
The algorithm made 7 of the 10 predictions correct with an accuracy of 70%.

1
2
accuracy = total correct predictions / total predictions made * 100
accuracy = 7 / 10 * 100
But what type of errors were made?
Let’s turn our results into a confusion matrix.
First, we must calculate the number of correct predictions for each class.

1
2
men classified as men: 3
women classified as women: 4
Now, we can calculate the number of incorrect predictions for each class, organized by the predicted value.

1
2
men classified as women: 2
woman classified as men: 1
We can now arrange these values into the 2-class confusion matrix:

1
2
3
		men	women
men		3	1
women	2	4
We can learn a lot from this table.
The total actual men in the dataset is the sum of the values on the men column (3 + 2)
The total actual women in the dataset is the sum of values in the women column (1 +4).
The correct values are organized in a diagonal line from top left to bottom-right of the matrix (3 + 4).
More errors were made by predicting men as women than predicting women as men.
Two-Class Problems Are Special
In a two-class problem, we are often looking to discriminate between observations with a specific outcome, from normal observations.
Such as a disease state or event from no disease state or no event.
In this way, we can assign the event row as “positive” and the no-event row as “negative“. We can then assign the event column of predictions as “true” and the no-event as “false“.
This gives us:
“true positive” for correctly predicted event values.
“false positive” for incorrectly predicted event values.
“true negative” for correctly predicted no-event values.
“false negative” for incorrectly predicted no-event values.
We can summarize this in the confusion matrix as follows:

1
2
3
  			event			no-event
event		true positive		false positive
no-event	false negative		true negative
This can help in calculating more advanced classification metrics such as precision, recall, specificity and sensitivity of our classifier.
For example, classification accuracy is calculated as true positives + true negatives.
Consider the case where there are two classes. […] The top row of the table corresponds to samples predicted to be events. Some are predicted correctly (the true positives, or TP) while others are inaccurately classified (false positives or FP). Similarly, the second row contains the predicted negatives with true negatives (TN) and false negatives (FN).
— Page 256, Applied Predictive Modeling, 2013
Now that we have worked through a simple 2-class confusion matrix case study, let’s see how we might calculate a confusion matrix in modern machine learning tools.
Code Examples of the Confusion Matrix
This section provides some example of confusion matrices using top machine learning platforms.
These examples will give you a context for what you have learned about the confusion matrix for when you use them in practice with real data and tools.
Example Confusion Matrix in Weka
The Weka machine learning workbench will display a confusion matrix automatically when estimating the skill of a model in the Explorer interface.
Below is a screenshot from the Weka Explorer interface after training a k-nearest neighbor algorithm on the Pima Indians Diabetes dataset.
The confusion matrix is listed at the bottom, and you can see that a wealth of classification statistics are also presented.
The confusion matrix assigns letters a and b to the class values and provides expected class values in rows and predicted class values (“classified as”) for each column.

Weka Confusion Matrix and Classification Statistics
You can learn more about the Weka Machine Learning Workbench here.
Example Confusion Matrix in Python with scikit-learn
The scikit-learn library for machine learning in Python can calculate a confusion matrix.
Given an array or list of expected values and a list of predictions from your machine learning model, the confusion_matrix() function will calculate a confusion matrix and return the result as an array. You can then print this array and interpret the results.

<pre><code>
# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix
 
expected = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
results = confusion_matrix(expected, predicted)
print(results)
</code></pre>

Running this example prints the confusion matrix array summarizing the results for the contrived 2 class problem.

1
2
[[4 2]
[1 3]]
Learn more about the confusion_matrix() function in the scikit-learn API documentation.

### Example Confusion Matrix in R with caret
The caret library for machine learning in R can calculate a confusion matrix.
Given a list of expected values and a list of predictions from your machine learning model, the confusionMatrix() function will calculate a confusion matrix and return the result as a detailed report. You can then print this report and interpret the results.

<pre><code>
# example of a confusion matrix in R
library(caret)
 
expected <- factor(c(1, 1, 0, 1, 0, 0, 1, 0, 0, 0))
predicted <- factor(c(1, 0, 0, 1, 0, 0, 1, 1, 1, 0))
results <- confusionMatrix(data=predicted, reference=expected)
print(results)
</code></pre>
Running this example calculates a confusion matrix report and related statistics and prints the results.

<pre><code>
Confusion Matrix and Statistics
 
          Reference
Prediction 0 1
         0 4 1
         1 2 3
 
               Accuracy : 0.7
                 95% CI : (0.3475, 0.9333)
    No Information Rate : 0.6
    P-Value [Acc > NIR] : 0.3823
 
                  Kappa : 0.4
 Mcnemar's Test P-Value : 1.0000
 
            Sensitivity : 0.6667
            Specificity : 0.7500
         Pos Pred Value : 0.8000
         Neg Pred Value : 0.6000
             Prevalence : 0.6000
         Detection Rate : 0.4000
   Detection Prevalence : 0.5000
      Balanced Accuracy : 0.7083
 
       'Positive' Class : 0
</code></pre>
There is a wealth of information in this report, not least the confusion matrix itself.
Learn more about the confusionMatrix() function in the caret API documentation [PDF].

#### Further Reading
There is not a lot written about the confusion matrix, but this section lists some additional resources that you may be interested in reading.
Confusion matrix on Wikipedia
Simple guide to confusion matrix terminology
Confusion matrix online calculator

#### Summary

In this post, you discovered the confusion matrix for machine learning.
Specifically, you learned about:
The limitations of classification accuracy and when it can hide important details.
The confusion matrix and how to calculate it from scratch and interpret the results.
How to calculate a confusion matrix with the Weka, Python scikit-learn and R caret libraries.
Do you have any questions?
Ask your question in the comments below and I will do my best to answer them.



