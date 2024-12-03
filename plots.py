import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('data/bert_kmeans.csv')

# Plot the results
plt.figure(figsize=(4.13, 2.91))  # Half the width of an A4 page (8.27 inches) and a reasonable height
plt.barh(data['K'], data['Accuracy'], color='b')

# Style the plot
plt.title('K-Means Clustering Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Number of Clusters')
plt.xlim(0, 1)  # Set x-axis range from 0 to 1
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('figures/kmeans_accuracy_plot.png', dpi=300)

#%%
# Load the CSV file
data = pd.read_csv('data/bert_rf.csv')

# Plot the results
plt.figure(figsize=(4.13, 2.91))  # Half the width of an A4 page (8.27 inches) and a reasonable height
plt.barh(data['n_estimators'], data['Accuracy'], color='b')

# Style the plot
plt.title('Random Forest Classifier Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Number of Estimators')
plt.xlim(0, 1)  # Set x-axis range from 0 to 1
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('figures/rf_accuracy_plot.png', dpi=300)