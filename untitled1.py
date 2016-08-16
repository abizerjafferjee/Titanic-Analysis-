# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:28:40 2016

@author: admin
"""

from sklearn.datasets import load_iris 

iris = load_iris()

x = iris.data
y = iris.target 



from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x,y)
a_new = [[3, 5, 4, 2],[5, 4, 3, 2]]
knn.predict(a_new)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x,y)
logreg.predict(a_new)
