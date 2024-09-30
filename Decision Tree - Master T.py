#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# Load the dataset
file_path = '/Users/galyaikonomova/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)


# In[40]:


# Define the independent variables (features) and the dependent variable (target)
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']


# In[41]:


# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[46]:


# Do a Grid Search
dt_estimator = DecisionTreeClassifier()
dt_param_grid = {"max_depth": [6, 7, 8], "min_samples_leaf": [1, 2, 3, 4], "min_samples_split": [2, 4, 6]}
dt_grid_searchCV = GridSearchCV(dt_estimator, dt_param_grid, scoring = "accuracy", verbose = True)


# In[47]:


dt_grid_searchCV.fit(X_train, y_train)


# In[48]:


dt_grid_searchCV.best_score_


# In[49]:


dt_grid_searchCV.best_params_


# In[50]:


# Train the Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=3, min_samples_split=2)
dt_model.fit(X_train, y_train)


# In[51]:


dt_model.get_params()


# In[52]:


# Make predictions on the test set
y_test_pred = dt_model.predict(X_test)


# In[53]:


# Make predictions on the training set
y_train_pred = dt_model.predict(X_train)


# In[54]:


# Evaluate the model - accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[55]:


# Evaluate the model - confusion matrix
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)


# In[56]:


# Classification report
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)


# In[57]:


# My training accuracy is slightly more than my test accuracy which migh suggest some overfitting. This is the reason after visualizing the tree, I decided to applu cross-validation
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Training Precision: {train_precision:.4f}")
# print(f"Test Precision: {test_precision:.4f}")

print("\nTraining Classification Report:\n", train_class_report)
print("Test Classification Report:\n", test_class_report)


# In[58]:


# Plot the confusion matrix
plot_confusion_matrix(dt_model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# In[59]:


# Feature Importance
importance = dt_model.feature_importances_


# In[60]:


# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


# In[61]:


print("\nFeature Importances:\n", importance_df)


# In[62]:


# Visualize the Feature Importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


# In[63]:


# Store feature names for later use
feature_names = X.columns


# In[64]:


fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(dt_model,
                   feature_names=feature_names,
                   class_names={0: 'Non-diabetes', 1: 'Diabetes'},
                   filled=True,
                   fontsize=12)
plt.show()

