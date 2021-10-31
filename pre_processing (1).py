import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#date et heure
"""
data0=pd.read_csv("test_ml.csv")
date=data0[['date']].values

data=data0
dic={'day':[],'day_number':[],'month':[],'year':[],'hour':[],'step':[]}


for k in range(date.shape[0]):
    word=date[k,0]
    word=word.split(' ')
    i=0
    if len(word)<6:
        for j in dic.keys():
            dic[j].append(None)
            i+=1
        continue
    for j in dic.keys():
        dic[j].append(word[i])
        i+=1

n=len(dic['hour'])

for k in range(n):
    if dic['step'][k]!=None and (len(dic['step'][k])<3 or dic['step'][k][2] not in [str(u) for u in range(10)]):
        v=random.randint(0,10)
        dic['step'][k]='00'+str(v)
    if dic['hour'][k]!=None and dic['step'][k]!=None:
        hour=int(dic['hour'][k][0:2])
        hour_step=int(dic['step'][k][2])
        hour=(hour+hour_step)%24
        dic['hour'][k]=hour
        dic['step'][k]='0'



data = data.drop(['date'], axis=1)
#data_dic=data.to_dict()

for elm in ['day','day_number','month','year','hour']:
    data[elm]=dic[elm]
    
data.head()

data.to_csv('new_test_set.csv',index=False)
"""

# split for a coupled mail_type
try : 
    data1 = pd.read_csv("test_set.csv")
    mail_type = data1[["mail_type"]].values
    data = data1

    for k in range(mail_type.shape[0]) : 
        data1[k] = data1[k].split("/")

    data = data1.drop(["mail_type"], axis = 1)
    data.head()
    data.to_csv('new_test_set2.csv', index=False)

except : 
    None