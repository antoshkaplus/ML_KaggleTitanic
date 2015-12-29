
# coding: utf-8

# In[29]:

# do reload as top level
import research
reload(research)
from research import *


# In[30]:

import random 
from sklearn.tree import DecisionTreeClassifier as DeciTree


train, mission = research.prepare_matrices()
# have to split train 80/20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)

# criterion: gini (Gini impurity), entropy
# somehow we want a lot of forests!!!
solver = DeciTree()
solver.fit(X_train, y_train)

# have to look at feature importance

res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/random_forest.csv", index=False)
print solver.score(X_test, y_test)


# In[31]:

# let's start with simple decision tree and make it work


# In[39]:

import random

train, mission = research.prepare_matrices()
# have to split train 80/20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)

# criterion: gini (Gini impurity), entropy
# somehow we want a lot of forests!!!
solver = RandForest(n_jobs=-1, n_estimators=100)
solver.fit(X_train, y_train)

# have to look at feature importance

res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/random_forest.csv", index=False)
print solver.score(X_test, y_test)
print solver.feature_importances_ 


# In[33]:

# let's find out which passagers suck dick on the bottom of the sea
y_test_pred = solver.predict(X_test)


# In[38]:

import random
from sklearn.grid_search import GridSearchCV


train, mission = research.prepare_matrices()
# have to split train 80/20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)

# criterion: gini (Gini impurity), entropy
# somehow we want a lot of forests!!!
solver = RandForest(n_jobs=-1)

# i could probably try to change scoring function
param_grid = {
    'n_estimators': [100, 900],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_depth': [3,4,5]
}

cv_solver = GridSearchCV(estimator=solver, param_grid=param_grid, cv=10)
cv_solver.fit(train, data_train.Survived)

print cv_solver.best_params_


# In[40]:

import random
from sklearn.grid_search import GridSearchCV


train, mission = research.prepare_matrices()
# criterion: gini (Gini impurity), entropy
# somehow we want a lot of forests!!!
solver = RandForest(n_jobs=-1)

# i could probably try to change scoring function
param_grid = {
    'n_estimators': [10, 50, 100],
}

cv_solver = GridSearchCV(estimator=solver, param_grid=param_grid, cv=10)
cv_solver.fit(train, data_train.Survived)

print cv_solver.best_params_


# In[43]:

import random
from sklearn.cross_validation import cross_val_score

train, mission = research.prepare_matrices()
solver = RandForest(n_jobs=-1, n_estimators=1000)
solver.fit(train, data_train.Survived)

scores = cross_val_score(solver, train, data_train.Survived, cv=10)
print min(scores)

res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/random_forest.csv", index=False)


# In[ ]:



