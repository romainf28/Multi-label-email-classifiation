#######################################################################
## Import all modules
#######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


#######################################################################
## Read csvs
#######################################################################


train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)

train_y = train_df[['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social']]


#######################################################################
## Get meaningful org and tld
#######################################################################


list_org = {}
for ind,org in enumerate(np.array(train_df[['org']])):
    if str(org) not in list_org.keys():
        list_org[str(org)] = np.array(train_y)[ind]
    else:
        list_org[str(org)] += np.array(train_y)[ind]
meaningful_org = [key for key in list_org.keys() if max(list_org[key]) >= 20] #justifier le 20

list_tld = {}
for ind,tld in enumerate(np.array(train_df[['tld']])):
    if str(tld) not in list_tld.keys():
        list_tld[str(tld)] = np.array(train_y)[ind]
    else:
        list_tld[str(tld)] += np.array(train_y)[ind]
meaningful_tld = [key for key in list_tld.keys() if max(list_tld[key]) >= 10] #justifier le 10

L_org = []
for org in train_df['org']:
    if (str([org]) not in meaningful_org) and (org not in L_org):
        L_org.append(org)
        
L_tld = []
for tld in train_df['tld']:
    if (str([tld]) not in meaningful_tld) and (tld not in L_tld):
        L_tld.append(tld)

for org in L_org:
    train_df.replace(to_replace=org, value="None", inplace=True)
    test_df.replace(to_replace=org, value="None", inplace=True)
for tld in L_tld:
    train_df.replace(to_replace=tld, value="None", inplace=True)
    test_df.replace(to_replace=tld, value="None", inplace=True)
    

#######################################################################
## Preprocess the date
#######################################################################


date=train_df[['date']]
dic={'day':[],'day_number':[],'month':[],'year':[],'hour':[],'step':[]}
for k in range(date.shape[0]):
    word=np.array(date)[k,0]
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

for key in ['day_number', 'year', 'hour']:
    L = []
    for elem in dic[key]:
        try:
            L.append(float(elem))
        except:
            L.append(np.nan)
        dic[key] = np.array(L)

dates_train = pd.DataFrame.from_dict(dic)
dates_train.drop(['step'], axis=1, inplace=True)

date=test_df[['date']]
dic={'day':[],'day_number':[],'month':[],'year':[],'hour':[],'step':[]}
for k in range(date.shape[0]):
    word=np.array(date)[k,0]
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

for key in ['day_number', 'year', 'hour']:
    L = []
    for elem in dic[key]:
        try:
            L.append(float(elem))
        except:
            L.append(np.nan)
        dic[key] = np.array(L)
        
dates_test = pd.DataFrame.from_dict(dic)
dates_test.drop(['step'], axis=1, inplace=True)


#######################################################################
## Filtering columns
#######################################################################


train_x_str = pd.concat([train_df[['org', 'tld', 'mail_type']], dates_train[['day', 'month']]], axis=1)
train_x_str = train_x_str.fillna(value='None')
train_x_numbers = pd.concat([dates_train[['day_number', 'year', 'hour']],train_df[['ccs', 'bcced', 'images', 'urls', 'salutations', 'designation', 'chars_in_subject', 'chars_in_body']]], axis=1)
train_x_numbers = train_x_numbers.fillna(value=np.nan)

test_x_str = pd.concat([test_df[['org', 'tld', 'mail_type']], dates_test[['day', 'month']]], axis=1)
test_x_str = test_x_str.fillna(value='None')
test_x_numbers = pd.concat([dates_test[['day_number', 'year', 'hour']],test_df[['ccs', 'bcced', 'images', 'urls', 'salutations', 'designation', 'chars_in_subject', 'chars_in_body']]], axis=1)
test_x_numbers = test_x_numbers.fillna(value=np.nan)


#######################################################################
## Impute mean for the missing values
#######################################################################


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(train_x_numbers)
train_x_numbers = imp.transform(train_x_numbers)
test_x_numbers = imp.transform(test_x_numbers) #must be same imputer

train_x_numbers = pd.DataFrame(train_x_numbers)
test_x_numbers = pd.DataFrame(test_x_numbers)


#######################################################################
## One hot encoding on str data
#######################################################################


feat_enc = OneHotEncoder(handle_unknown ='ignore')
feat_enc.fit(np.vstack([train_x_str, test_x_str]))
train_x_featurized_str = feat_enc.transform(train_x_str)
test_x_featurized_str = feat_enc.transform(test_x_str)
train_x_str = pd.DataFrame(train_x_featurized_str.toarray())
test_x_str = pd.DataFrame(test_x_featurized_str.toarray())


#######################################################################
## Concatenate all data into one file
#######################################################################


train_x = pd.concat([train_x_str, train_x_numbers], axis=1)
test_x = pd.concat([test_x_str, test_x_numbers], axis=1)


#######################################################################
## Normalize
#######################################################################


scaler = StandardScaler(with_mean=False).fit(train_x) #breaks sparsity if with_mean=True
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x) #must be same scaling

#######################################################################
## Split dataset
#######################################################################

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)


#######################################################################
## Use classification models
#######################################################################


#Random forest
clf = RandomForestClassifier(n_estimators=150, criterion="gini", bootstrap=True, warm_start=False)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(accuracy_score(y_test, pred, normalize=True))


"""
#Neural network
hidden_layer_size = tuple([20 for i in range(50)])
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_layer_size, random_state=1, learning_rate='adaptive')
clf.fit(train_x, train_y)
pred=clf.predict(train_x)

print(accuracy_score(train_y, pred, normalize=True))
"""


#######################################################################
## Export results into a csv file
#######################################################################
"""
pred_df = pd.DataFrame(clf.predict(test_x), columns=['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social'])
pred_df.to_csv("knn_sample_submission_ml_team.csv", index=True, index_label='Id')
"""


"""
# A priori useless
#Show how correlated is our data
colormap = plt.cm.RdBu
plt.figure(figsize=(32,10))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(train_x.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
"""