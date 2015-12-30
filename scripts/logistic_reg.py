
# coding: utf-8

# In[36]:

get_ipython().magic(u'run research.py')


# In[ ]:

# lets put here all imports that we need
import random 
from sklearn.tree import DecisionTreeClassifier as DeciTree
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# In[ ]:

train, mission = prepare_matrices()
# have to split train 80/20
# classification reports asks for predicates....
solver = LogRegCV(n_jobs=-1)
solver.fit(train, data_train.Survived)
res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_0.csv", index=False)
#print solver.score(X_test, y_test)

#y_test_pred = solver.predict(X_test)
#print metrics.classification_report(y_test, y_test_pred)
#print all_cols
#print solver.coef_

scores = cross_val_score(solver, train, data_train.Survived, cv=10)
print min(scores)

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



