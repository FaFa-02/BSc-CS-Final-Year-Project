from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from ridge_regression import RidgeRegressionClassifier
import numpy as np
import matplotlib.pyplot as plt

# Importing linnerud dataset, seperate dataset into its features and labels(waist)
linnerud = load_linnerud()

X = linnerud['data']
y = linnerud['target'][:,1]


# Plot features against labels to visualise data
fig, ax = plt.subplots(3, figsize=(15, 15))
plt.suptitle("Linnerud_pairplot")

x = np.linspace(-2, 2, 100)
for i in range(3):
    print(i)
    ax[i].scatter(X[:,i], y, s=100)
    ax[i].set_xticks(())
    ax[i].set_yticks(())
    ax[i].set_xlabel(linnerud['feature_names'][i])
    ax[i].set_ylabel(linnerud['target_names'][1])

plt.show()

# Split dataset into training and test sets in preparation for the Ridge Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

ridge = RidgeRegressionClassifier(1)

ridge.fit(X_train, y_train)

y_hat = ridge.predict(X_test)

# Print the optimal coefficients found by the model and the predictions along with the real labels
print("betas:", ridge.beta_ridge_hat)
print("predictions:", y_hat)
print("true labels:", y_test)

# Plot features against targets to visualise data
fig, ax = plt.subplots(3, figsize=(15, 15))
plt.suptitle("Linnerud_pairplot")

x = np.linspace(-2, 2, 100)
for i in range(3):
    print(i)
    ax[i].scatter(X[:,i], y, s=100)
    ax[i].scatter(X_test[:,i], y_hat)
    ax[i].set_xticks(())
    ax[i].set_yticks(())
    ax[i].set_xlabel(linnerud['feature_names'][i])
    ax[i].set_ylabel(linnerud['target_names'][1])

plt.show()
