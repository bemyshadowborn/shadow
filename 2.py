import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt
# Step 1: Load the dataset
# Replace this with the path to your dataset
df = pd.read_csv('California Housing.csv') # Example: 'data.csv'
# Step 2: Check for non-numeric columns and handle them (convert or drop them)
df = df.select_dtypes(include=['number']) # Keep only numeric columns
# Step 3: Compute the correlation matrix
correlation_matrix = df.corr()
# Step 4: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
# Step 5: Create a pair plot to visualize pairwise relationships
sns.pairplot(df)
plt.suptitle('Pairwise Relationships Between Features', y=1.02)
plt.show()
