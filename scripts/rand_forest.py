
# coding: utf-8

# In[26]:

# do reload as top level
import research
reload(research)
from research import *


# In[7]:

# lets put here all imports that we need
import random 
from sklearn.tree import DecisionTreeClassifier as DeciTree
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# In[19]:




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


# In[9]:

# let's start with simple decision tree and make it work


# In[20]:

train, mission = research.prepare_matrices()
# have to split train 80/20
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


# In[11]:

# let's find out which passagers suck dick on the bottom of the sea
y_test_pred = solver.predict(X_test)


# In[12]:



train, mission = research.prepare_matrices()
# have to split train 80/20
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


# In[ ]:


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


# In[27]:

train, mission = research.prepare_matrices()
#solver = RandForest(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)
solver = RandForest(n_jobs=-1, n_estimators = 80, max_features='auto', criterion='entropy',max_depth=4)
solver.fit(train, data_train.Survived)

scores = cross_val_score(solver, train, data_train.Survived, cv=10)
print min(scores)

res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/random_forest.csv", index=False)


# In[ ]:



