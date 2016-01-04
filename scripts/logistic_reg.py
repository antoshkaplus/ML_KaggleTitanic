
# coding: utf-8

# In[24]:

import research
reload(research)
from research import *


# In[25]:

# lets put here all imports that we need
import random 
from sklearn.tree import DecisionTreeClassifier as DeciTree
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# In[26]:

solver = None
print all_cols
def solve(cols=all_cols):
    train, mission = prepare_matrices(cols)
    # have to split train 80/20
    # classification reports asks for predicates....
    global solver
    solver = LogRegCV(n_jobs=-1, cv=10)
    solver.fit(train, data_train.Survived)
    res = solver.predict(mission)
    res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
    res.to_csv("../output/logic_0.csv", index=False)
    print solver.C_
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)
    solver = LogRegCV(n_jobs=-1, cv=3)
    solver.fit(X_train, y_train)
    res = solver.predict(mission)
    res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
    res.to_csv("../output/logic_0.csv", index=False)
    
    print solver.score(X_test, y_test)

    #y_test_pred = solver.predict(X_test)
    #print metrics.classification_report(y_test, y_test_pred)
    #print all_cols
    #print solver.coef_
    """
    groups = data_train.groupby('Sex').groups 
    males = groups['male']
    females = groups['female']

    tt = np.matrix(train)
    p = solver.predict(tt[males])
    print class_report(data_train.Survived[males], p)
    
    scores = cross_val_score(solver, train, data_train.Survived, cv=10)
    #print min(scores)

    tt = np.matrix(train)
    p = solver.predict(tt[females])
    print class_report(data_train.Survived[females], p)
    """
    
    scores = cross_val_score(solver, train, data_train.Survived, cv=10)
    print scores
    print np.mean(scores)

    
#solve()
noIsFamily = list(all_cols)
noIsFamily.remove('IsFamily')
solve(noIsFamily)
    
print solver 
"""
solver = LogRegCV(n_jobs=-1, scoring='roc_auc')
solver.fit(train, data_train.Survived)
res = solver.predict(test)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_1.csv", index=False)
print solver.score(train, data_train.Survived)


solver = LogRegCV(n_jobs=-1, scoring='average_precision')
solver.fit(train, data_train.Survived)
res = solver.predict(test)

res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_2.csv", index=False)
print solver.score(train, data_train.Survived)
"""

"""
solver = LogRegCV(n_jobs=-1)
solver.fit(train, data_train.Survived)
model = SelectFromModel(solver, prefit=True)
print model.get_support()


t_new = model.transform(train)
print t_new.shape

res = solver.predict(test)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_1.csv", index=False)
print solver.score(train, data_train.Survived)

"""


# In[ ]:




# In[ ]:



