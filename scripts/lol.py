
import pandas as pd

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")

print data_train
print data_test

print "Missing data:"
print len(data_train.isnull())
