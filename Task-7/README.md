
**The primary goal:**

Classify breast tumors as malignant or benign using machine learning (SVM) models and evaluate their performance.
It aims to compare different SVM kernels and find the best parameters to achieve high classification accuracy.


This code performs binary classification on a **breast cancer dataset** using **Support Vector Machines (SVMs)**. It includes:

  - Data loading and cleaning.
  - Feature scaling and dimensionality reduction using PCA.
  - Model training using SVM with both linear and RBF kernels.
  - Visualization of decision boundaries.
  - Hyperparameter tuning using GridSearchCV.
  - Model evaluation with cross-validation and classification report.

The Dataset used is **Breast Cancer Dataset** which contains 569 samples with 30 numerical features derived from digitized images of fine needle aspirate (FNA) of breast mass. Each sample is labeled as either:
  - 'M' = Malignant (cancerous)
  - 'B' = Benign (non-cancerous)

