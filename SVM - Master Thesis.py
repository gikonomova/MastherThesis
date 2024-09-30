#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = '/Users/galyaikonomova/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)


# In[2]:


# Set 'Diabetes_binary' as the target variable and the rest are left as features
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[3]:


# First split to reduce the dataset size to 20%
X_reduced, _, y_reduced, _ = train_test_split(X, y, test_size=0.2)


# In[4]:


# Now split the reduced dataset into training and testing sets (e.g., 80% training, 20% testing of the reduced data)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.2)


# In[5]:


# Do a Grid Search
svm_estimator = SVC()
svm_param_grid = {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1, 10] }
svm_grid_searchCV = GridSearchCV(svm_estimator, svm_param_grid, scoring = "accuracy", verbose = True)


# In[13]:


# Do a Grid Search
# svm_estimator = SVC()
# svm_param_grid = {"kernel": ["rbf"], "C": [1], "gamma": [0.001, 0.01] }
# svm_grid_searchCV = GridSearchCV(svm_estimator, svm_param_grid, scoring = "accuracy", verbose = True)


# In[6]:


# Fir the SVM grid search model
svm_grid_searchCV.fit(X_train, y_train)


# In[7]:


svm_grid_searchCV.best_score_


# In[8]:


svm_grid_searchCV.best_params_


# In[11]:


# Train the SVM model
svm_model = SVC(C=10, gamma=0.001, kernel='rbf')
svm_model.fit(X_train, y_train)


# In[12]:


# Assuming you have already trained the model
# Make predictions on the training set
y_train_pred = svm_model.predict(X_train)

# Now, calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = svm_model.predict(X_test)

# Now, calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[13]:


# Classification report
test_class_report = classification_report(y_test, y_test_pred)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample confusion matrix for demonstration
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('Confusion Matrix with Color Bar')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[16]:


# Print results
print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Test Precision: {test_precision:.4f}")
print("\nTest Classification Report:\n", test_class_report)
print("Test Confusion Matrix:\n", test_conf_matrix)


# In[ ]:




