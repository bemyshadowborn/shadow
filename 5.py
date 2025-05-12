import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generating random values and creating labels
values = np.random.rand(100)
print("\nValues:")
print(values)

labels = []
for i in values[:50]:
    if i <= 0.5:
        labels.append('Class1')
    else:
        labels.append('Class2')
labels += [None] * 50
print("\nLabels:")
print(labels)

# Creating DataFrame
data = {
    "point": [f"x {i+1}" for i in range(100)],
    "value": values,
    "Label": labels
}            

print("\nData Frame:")
df = pd.DataFrame(data)
print(df)

# Plotting histograms
num_col = df.select_dtypes(include=['int', 'float']).columns
for col in num_col:
    df[col].hist(figsize=(12,8), bins=10, edgecolor='black')
    plt.title(f"Histogram for {col}", fontsize=16)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Preparing labeled and unlabeled data
labeled_df = df[df["Label"].notna()]
x_train = labeled_df[["value"]]  # Use the correct column name for values
y_train = labeled_df["Label"]    # Correct label column reference

# Preparing true labels for the unlabeled part
true_labels = ["Class1" if x <= 0.5 else "Class2" for x in values[50:]]

# Define k values to test
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}
accuracies = {}

# KNN Classification
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    
    # Split unlabeled data for testing
    unlabeled_df = df[df["Label"].isna()]
    x_test = unlabeled_df[["value"]]  # Correct test data column reference
    
    predictions = knn.predict(x_test)
    results[k] = predictions
    accuracies[k] = accuracy_score(true_labels, predictions) * 100
    
    # Print accuracy for the given k
    print(f"Accuracy for k={k}: {accuracies[k]:.2f}%")
    
    unlabeled_df[f"Label_k{k}"] = predictions  # Assign the predicted labels to the unlabeled data

# Drop the original 'Label' column from unlabeled data
df1 = unlabeled_df.drop(columns=['Label'], axis=1)

# Display the data frame with predicted labels
print("\nData Frame with Label:")
print(df1)
