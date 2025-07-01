**The main goal is to:**
-Classify the species of Iris flowers using their petal and sepal measurements.
-Optimize the K value in KNN to achieve the best prediction accuracy.
-Visualize how the classifier separates the classes using a 2D decision boundary.

**What the Code Does:**
The code performs K-Nearest Neighbors (KNN) classification on the Iris flower dataset. It includes:
  -Loading and preprocessing the dataset.
  -Normalizing the features using StandardScaler.
  -Splitting the dataset into training and testing sets.
  -Training KNN models with multiple K values (from 1 to 20) to find the optimal number of neighbors.
  -Evaluating the best model using a confusion matrix and classification report.
  -Visualizing decision boundaries using the first two features.
  -Performing cross-validation to validate model consistency.

The Datset used here is **Iris Dataset**. It contains attributes like: id, sepal length, sepal width, petal length, petal width, species.


