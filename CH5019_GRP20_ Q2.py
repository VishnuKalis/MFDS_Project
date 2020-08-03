#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[92]:


re  = pd.read_csv('Data set question2.csv')
re.head()
re.info()
re['Test'].value_counts()


# In[93]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
re['Test']=label_encoder.fit_transform(re['Test'])

corr_matrix = re.corr()
corr_matrix['Test'].sort_values(ascending=False)


# In[94]:


X1 = np.array(re.iloc[:,:5])
Y = np.array(re.iloc[:,5:])
X0 = np.ones((len(re),1),dtype=float)
X = np.concatenate((X0,X1),axis=1)


from sklearn.model_selection import *
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


# In[95]:


from sklearn.base import ClassifierMixin

class Logistic(ClassifierMixin):
    def fit(self,X=None,Y=None):
        #B = np.random.rand(np.shape(X)[1],1)
        B = [   [ 812.25329167],
                [ 392.31241846],
                [ 496.96732017],
                [1009.05307749],
                [1001.37857587],
                [ 788.45853768]     ]

        B = np.array(B)
        
        
        print('Initial parameter values:',B)
        
        def derivative(B):
            sig = lambda a : 1/(1+np.exp(-a))
            z = np.shape(X)[1]
            grad = np.zeros((z,1),dtype=float)
            m = np.shape(X)[0]

            for j in range(len(grad)):
                su = 0
                for i in range(len(X)):
                    a = X[i,:].dot(B)
                    su = su + ((sig(a) - Y[i])*X[i,j])

                grad[j] = -(su/m)
            return grad

        def gradient_descent(W,alpha):
            grad = derivative(W)
            for i in range(75000):
                W = W - alpha*grad

            print('Parameters values :', W)
            return W
        self.y_bar = gradient_descent(B,0.01)
        
    def predict(self,X=None):
        Y_bar = np.zeros((len(X),1),dtype = float)
        sig = lambda a : 1/(1+np.exp(-a))
        for i in range(len(X)-1):
            a = X[i,:].dot(self.y_bar)
            if sig(a) >= 0.5:
                Y_bar[i] = 1
            elif sig(a) < 0.5:
                Y_bar[i] = 0
            else:
                Y_bar[i] = None
                
        return Y_bar
 

# In[96]:


model = Logistic()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import *
print('Confusion_matrix :',confusion_matrix(Y_test,Y_pred))
print('Accuracy_score :',accuracy_score(Y_test,Y_pred))
print('Report :',classification_report(Y_test,Y_pred))


