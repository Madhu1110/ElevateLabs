**The Main Goal:** 

The primary goal of this code is to segment customers into distinct groups based on their behavior (specifically, annual income and spending score). This helps businesses or marketing teams to better understand their customer base and tailor strategies such as promotions, product recommendations, or loyalty programs for different customer segments.

**What this code does:**
This Python script performs customer segmentation using the K-Means clustering algorithm. It follows a complete unsupervised machine learning pipeline:
  - Loads and preprocesses the dataset.
  - Scales the data using StandardScaler for optimal clustering.
  - Applies the Elbow Method to determine the optimal number of clusters (k) by analyzing inertia (within-cluster sum of squares).
  - Fits the K-Means model with the chosen value of k and assigns each customer to a cluster.
  - Optionally reduces dimensionality using PCA to visualize the clusters in 2D.
  - Evaluates the clustering performance using the Silhouette Score.
  - Visualizes the clusters with color-coding to show how customers are grouped based on their annual income and spending behavior.

 The dataset used is **Mall Customers Dataset** .It contains customer data with the following columns:
   - CustomerID
   - Gender
   - Age
   - Annual Income (k$)
   - Spending Score (1-100)
