'''
0       Andik Vermansyah   27       87         90  Non-Target
1     Awan Setho Raharjo   22       75         83  Non-Target
2      Bambang Pamungkas   38       85         75  Non-Target
3      Cristian Gonzales   43       90         85  Non-Target
4      Egy Maulana Vikri   18       88         90      Target
5             Evan Dimas   24       85         87      Target
6         Febri Hariyadi   23       77         80  Non-Target
7   Hansamu Yama Pranata   24       82         85      Target
8  Septian David Maulana   22       83         80      Target
9       Stefano Lilipaly   29       88         86  Non-Target
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv('data.csv')
# print(list(df))
df_n=df[['Name','Age','Overall','Potential']] #buat df lebih mudah dilihat


xao=df_n[(df_n['Age']<=25)&(df_n['Overall']>=80)&(df_n['Potential']>=80)]
ind_xao=xao.index.tolist() #index yang target
df_n['Status']=['Target' if i in ind_xao else 'Non_Target' for i in range(len(df_n.index))]

# print(df_n.head(10))
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df_new=df_n #mau di drop kolom statusnya
df_new['Status_en']=lab.fit_transform(df_n['Status'])
df_new=df_new.drop(['Status'],axis='columns')
# print(df_new.head(10))
untukx=df_new[['Age','Overall','Potential']]
untuky=df_new['Status_en']

##SPLIT
from sklearn.model_selection import train_test_split,KFold
x_trainset, x_testset, y_trainset, y_testset = train_test_split(untukx, untuky, test_size=0.1, random_state=3) #test_size sudah diganti2, tetep decision tree paling baik
kf=KFold(n_splits=3,random_state=1) ##buat cross_validation

## TREE
from sklearn import tree
#Train Model and Predict  
fifaTree=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
fifaTree.fit(x_trainset,y_trainset)

##### PREDICT
dfpred=pd.read_csv('prediksisoal2.csv')
print(dfpred.head())
x=dfpred.iloc[:,1:]
prediksi=fifaTree.predict(x)

li=[]
for i in prediksi:
    if i ==1:
        li.append('Target')
    else:
        li.append('Non-Target')
dfpred['Status']=li
print(dfpred)


