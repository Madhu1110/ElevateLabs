import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# Load  and preprocessing the dataset
df = pd.read_csv("C:/Users/madhu/OneDrive/Documents/ElevateLabs/task6/archive (6)/Iris.csv")
df.drop("Id", axis=1, inplace=True)

# Normalising the features
X = df.drop("Species", axis=1)
y = df["Species"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train KNearestNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Experiment with different values of K and evaluate
k_range = range(1, 21)
accuracies = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

# Plot accuracy vs. K
plt.figure(figsize=(8,5))
plt.plot(k_range, accuracies, marker='o', linestyle='dashed')
plt.title("Accuracy vs. K")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Evaluate using best K
best_k = k_range[np.argmax(accuracies)]
print(f"Best K: {best_k}")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix, accuracy
print("\nConfusion Matrix:",confusion_matrix(y_test, y_pred))
print("\nClassification Report:",classification_report(y_test, y_pred))

# Visualize decision boundaries (only using first 2 features for 2D plot)
def plot_decision_boundaries(X, y, model, title):
    # Encode string labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ["red", "green", "blue"]

    X = X[:, :2]  # Only use first 2 features for 2D plot

    h = .02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = le.transform(Z)  # Convert to numeric values
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    for idx, class_name in enumerate(np.unique(y)):
        plt.scatter(X[y == class_name][:, 0], 
                    X[y == class_name][:, 1],
                    label=class_name,
                    c=cmap_bold[idx])
        
    plt.xlabel("Feature 1 (Sepal Length)")
    plt.ylabel("Feature 2 (Sepal Width)")
    plt.title(title)
    plt.legend()
    plt.show()

# Train on first two features only for visualization
X_vis = X_scaled[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.2, random_state=42)
model_vis = KNeighborsClassifier(n_neighbors=best_k)
model_vis.fit(X_train_vis, y_train_vis)

plot_decision_boundaries(X_train_vis, y_train_vis, model_vis, f"Decision Boundaries (K={best_k})")
