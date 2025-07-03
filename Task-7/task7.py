import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv("C:/Users/madhu/OneDrive/Documents/ElevateLabs/task7/archive (6)/breast-cancer.csv")  # Replace with your path if needed
df = df.drop(columns=[col for col in ['id'] if col in df.columns])

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # M = Malignant, B = Benign

df = df.dropna(subset=['diagnosis'])
# 2. Prepare the data
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Visualize function
def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

# 4. Train and visualize SVM with Linear Kernel
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
plot_decision_boundary(X_train, y_train, svm_linear, "SVM with Linear Kernel")

# 5. Train and visualize SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
plot_decision_boundary(X_train, y_train, svm_rbf, "SVM with RBF Kernel")

# 6. Hyperparameter Tuning (C and gamma)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
plot_decision_boundary(X_train, y_train, grid.best_estimator_, "Tuned SVM with RBF Kernel")

# 7. Cross-validation
scores = cross_val_score(grid.best_estimator_, X_pca, y, cv=5)
print("Cross-validation Accuracy: {:.2f}%".format(scores.mean() * 100))

# 8. Final Evaluation
y_pred = grid.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
