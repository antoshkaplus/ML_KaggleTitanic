#for ms we have to check if they are married or not
# 
s = list(data_all[data_all.Title == "Ms"].LastName)
print s
print data_all[(data_all['LastName'] == s[0]) | (data_all['LastName'] == s[1])]

