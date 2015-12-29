
# coding: utf-8

# In[30]:

# do reload as top level
import research
reload(research)
from research import *


# In[31]:

import random
from sklearn.svm import SVC
from sklearn.preprocessing import normalize


train, mission = prepare_matrices()

train = normalize(train, axis=1)
mission = normalize(mission, axis=1)

#train[]

# have to split train 80/20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_train.Survived, test_size=0.2, random_state=0)



# criterion: gini (Gini impurity), entropy
solver = SVC(kernel='rbf', verbose=True) # rbf is best option
solver.fit(X_train, y_train)
res = solver.predict(mission)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/svn.csv")
print solver.score(X_test, y_test)


# In[ ]:




# In[ ]:



