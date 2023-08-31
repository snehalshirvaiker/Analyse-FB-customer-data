#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# A start-up is running targetted marketing ads on facebook.
# Our objective is to anaylze customer behaviour by predicting which customer clicks on the advertisement. Customer data is as follows: 
# 
# Inputs: 
# - Name 
# - e-mail 
# - Country 
# - Time on Facebook 
# - Estimated Salary (derived from other parameters)
# 
# Outputs:
# - Click (1: customer clicked on Ad, 0: Customer did not click on the Ad)

# # LIBRARIES IMPORT
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # IMPORT DATASET

# In[3]:


# reading the data using pandas dataframe
training_set = pd.read_csv("C:/Users/sneha/Downloads/Facebook_Ads_2.csv", encoding='ISO-8859-1')


# In[4]:


# Show the data head
training_set.head()


# In[5]:


# Show the data tail
training_set.tail()


# # EXPLORE/VISUALIZE DATASET

# In[6]:


click    = training_set[training_set['Clicked']==1]
no_click = training_set[training_set['Clicked']==0]


# In[7]:


print("Total =", len(training_set))

print("Number of customers who clicked on Ad =", len(click))
print("Percentage Clicked =", 1.*len(click)/len(training_set)*100.0, "%")
 
print("Did not Click =", len(no_click))
print("Percentage who did not Click =", 1.*len(no_click)/len(training_set)*100.0, "%")
 
        


# In[8]:


sns.scatterplot(training_set['Time Spent on Site'], training_set['Salary'], hue = training_set['Clicked'])


# In[9]:


plt.figure(figsize=(5, 5))
sns.boxplot(x='Clicked', y='Salary',data=training_set)


# In[10]:


plt.figure(figsize=(5, 5))
sns.boxplot(x='Clicked', y='Time Spent on Site',data=training_set)


# In[11]:


training_set['Salary'].hist(bins = 40)


# In[12]:


training_set['Time Spent on Site'].hist(bins = 20)


# # PREPARING THE DATA 

# In[13]:


training_set


# In[14]:


#Let's drop the emails, country and names (we can make use of the country later!)
training_set.drop(['Names', 'emails', 'Country'],axis=1,inplace=True)


# In[15]:


training_set


# In[16]:


#Let's drop the target coloumn before we do train test split
X = training_set.drop('Clicked',axis=1).values
y = training_set['Clicked'].values


# In[17]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# # MODEL TRAINING

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


X_train


# In[21]:


y_train


# In[22]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# # MODEL TESTING

# In[23]:


y_predict_train = classifier.predict(X_train)
y_predict_train


# In[24]:


y_train


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True, fmt="d")


# In[26]:


y_predict_test = classifier.predict(X_test)
y_predict_test


# In[27]:


cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True, fmt="d")


# In[28]:


#This is the confusion matrix from the test data.


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_test))


# In[30]:


#We got 86% accuracy


# In[31]:


from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_predict_test)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_predict_test)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# # VISUALIZING OUR RESULTS

# In[32]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# Creating a meshgrid ranging from the minimum to maximum value for both features

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


# In[33]:


y_train.shape


# In[34]:


X_train.shape


# In[35]:


X1.shape


# In[36]:


# plotting the boundary using the trained classifier
# Running the classifier to predict the outcome on all pixels with resolution of 0.01
# Colouring the pixels with 0 or 1
# If classified as 0 it will be red, and if it is classified as 1 it will be shown in green 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


# In[40]:


# plot all the actual training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[43]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


#The model is accurately classifying more than 80 percent of the customers of the training as seen above


# In[42]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


#The model is accurately classifying more than 80 percent of the customers of the test as seen above

