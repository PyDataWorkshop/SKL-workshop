Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are available in the scikit. A simple way to extend these algorithms to the multi-class classification case is to use the so-called one-vs-all scheme.
At learning time, this simply consists in learning one regressor or binary classifier per class. 

In doing so, one needs to convert multi-class labels to binary labels (belong or does not belong to the class). 

LabelBinarizer makes this process easy with the transform method.
At prediction time, one assigns the class for which the corresponding model gave the greatest confidence. 

LabelBinarizer makes this easy with the inverse_transform method.
