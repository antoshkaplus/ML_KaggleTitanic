
# coding: utf-8

# In[8]:

get_ipython().magic(u'run research.py')


# In[9]:

train, mission = prepare_matrices()
# have to split train 80/20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)

# classification reports asks for predicates....
solver = LogRegCV(n_jobs=-1)
solver.fit(X_train, y_train)
res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_0.csv", index=False)
print solver.score(X_test, y_test)

y_test_pred = solver.predict(X_test)
print metrics.classification_report(y_test, y_test_pred)
print all_cols
print solver.coef_

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



