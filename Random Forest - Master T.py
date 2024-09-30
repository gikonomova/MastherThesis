#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix


# In[2]:


# Load the dataset
file_path = '/Users/galyaikonomova/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)


# In[3]:


# Define the independent variables (features) and the dependent variable (target)
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']


# In[4]:


# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


# Do a Grid Search
from sklearn.model_selection import GridSearchCV

rf_grid = RandomForestClassifier()
rf_param_grid = {'max_depth': [3,5,7,10], 'n_estimators': [100, 200, 300, 400, 500], 'max_features': [10, 20, 30 , 40], 'min_samples_leaf': [1, 2, 4]}
rf_grid_searchCV = GridSearchCV(rf_grid, rf_param_grid, cv = 3, scoring='accuracy', verbose = 3)


# In[6]:


rf_grid_searchCV.fit(X_train, y_train)


# In[7]:


rf_grid_searchCV.best_params_


# In[9]:


# Train the Decision Tree model
rf_model = RandomForestClassifier(max_depth=10, min_samples_leaf=4, max_features=10, n_estimators=200)
rf_model.fit(X_train, y_train)


# In[10]:


# Make predictions on the test set
y_test_pred = rf_model.predict(X_test)

# Make predictions on the training set
y_train_pred = rf_model.predict(X_train)


# In[11]:


# Evaluate the model - accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Evaluate the model - precision
train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)


# In[12]:


# Classification reports
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)


# In[13]:


# Print results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Test Precision: {test_precision:.4f}")

print("\nTraining Classification Report:\n", train_class_report)
print("Test Classification Report:\n", test_class_report)


# In[14]:


# Display confusion matrices
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nTraining Confusion Matrix:\n", train_conf_matrix)
print("Test Confusion Matrix:\n", test_conf_matrix)


# In[15]:


# Feature Importance
importance = rf_model.feature_importances_


# In[16]:


# Create a DataFrame for better visualization
importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
importance_rf = importance_rf.sort_values(by='Importance', ascending=False)


# In[18]:


print("\nFeature Importances:\n", importance_rf)


# In[21]:


import matplotlib.pyplot as plt
# Visualize the Feature Importance
plt.figure(figsize=(12, 8))
plt.barh(importance_rf['Feature'], importance_rf['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


# In[ ]:




