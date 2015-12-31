
# coding: utf-8

# In[ ]:

import logistic_reg
reload(logistic_reg)
from logistic_reg import *


# In[ ]:

# lets divide group on sex groups
# how good we can do?
groups = data_train.groupby('Sex').groups 
males = groups['male']
females = groups['female']

p = solver.predict(data_train[males])
class_report(data_train[males].Survived, p)

print "hello"


# In[ ]:



