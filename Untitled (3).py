#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
# Load the dataset
data = pd.read_excel('embeddingsdatasheet-1.xlsx')
 
# 'embed_0' and 'embed_1' are the features and 'Label' is the target variable
features = data[['embed_0', 'embed_1']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
 
# Initialize and train the Support Vector Machine (SVM) model
clf = SVC()
clf.fit(X_train, y_train)
 
# Get the support vectors
support_vectors = clf.support_vectors_
 
# Print the support vectors
print(f'Support Vectors ={support_vectors}')


# In[2]:


# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['embed_0', 'embed_1']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['embed_0', 'embed_1']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[3]:


decision_values = clf.decision_function(X_test[['embed_0', 'embed_1']])
 
# Relate the decision values to the class values
predictions = clf.predict(X_test[['embed_0', 'embed_1']])
 
# Test the accuracy using your own logic for class determination
# Here, we'll simply compare decision values against zero for binary classification
# Adjust this logic based on the specifics of your classification problem
 
correct_predictions = 0
for i in range(len(predictions)):
    if (predictions[i] == 1 and decision_values[i] > 0) or (predictions[i] == 0 and decision_values[i] < 0):
        correct_predictions += 1
 
accuracy = correct_predictions / len(y_test)
print(f"Accuracy using decision values: {accuracy}")
 


# In[ ]:
import numpy as np
import pandas as pd
from sklearn import svm

# Read data from the Excel file
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

# Extract data from the columns 'embed_0' and 'embed_1' for training
X_train = df[['embed_0', 'embed_1']]  # Features
y_train = df['Label']  

# Create an SVM classifier
clf = svm.SVC()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Get the support vectors
support_vectors = clf.support_vectors_

# Now you can study the support vectors
print("Support Vectors:")
print(support_vectors)



# Use the predict() function to get predicted class values for the test set
predicted_classes = clf.predict(X_train)

# Calculate accuracy by comparing predicted_classes to the actual y_test labels
correct_predictions = (predicted_classes == y_train)
accuracy = np.mean(correct_predictions)

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}")
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data from the Excel file
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

# Extract data from the columns 'embed_0' and 'embed_1' for training
X = df[['embed_0', 'embed_1']]  # Features
y = df['Label']  
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create an SVM classifier with the current kernel function
    clf = svm.SVC(kernel=kernel)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Use the classifier to make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate and print accuracy for the current kernel
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with '{kernel}' kernel: {accuracy:.2f}")





