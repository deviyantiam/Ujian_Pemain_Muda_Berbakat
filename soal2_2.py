'''
Berdasarkan model yang saya buat, paling bagus decision tree dengan rincian dibawah
DecisionTrees's Accuracy:  1.0
Logistic's Accuracy:  0.9939593629873695
KNN's Accuracy:  0.9989017023613399
Cross_Val_Score untuk Decision Tree 1.0
Cross_Val_Score untuk Linear Regression 0.9924935920908092
Cross_Val_Score untuk KNN 0.9989625289881606
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv('data.csv')
# print(list(df))
df_n=df[['Name','Age','Overall','Potential']] #buat df lebih mudah dilihat
#Check if there's any missing values in the dataset
df_n=df_n.replace(['-','n.a'],np.nan)
df_n=df_n.fillna(0)
# print('Jumlah data yang bernilai 0 :')
# print(df_n.isnull().sum())
# print(len(df))
# print(df.head())

xao=df_n[(df_n['Age']<=25)&(df_n['Overall']>=80)&(df_n['Potential']>=80)]
ind_xao=xao.index.tolist() #index yang target
df_n['Status']=['Target' if i in ind_xao else 'Non_Target' for i in range(len(df_n.index))]
# print(df_n.head(10))

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df_new=df_n #mau di drop kolom statusnya
df_new['Status_en']=lab.fit_transform(df_new['Status']) ####0 bukan target, 1 target
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
predTree = fifaTree.predict(x_testset)
## score/accuracy
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
'''
##EVALUATION untuk decision tree
error=[]
for i in range(1,int(untukx.shape[1])+1):
    drugTree=tree.DecisionTreeClassifier(criterion='entropy',max_depth=i)
    drugTree.fit(x_trainset,y_trainset)
    predTree = drugTree.predict(x_testset)
    err=metrics.accuracy_score(y_testset, predTree)
    error.append(err)
from sklearn import metrics
# import matplotlib.pyplot as plt
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
print(error)
'''
###====================================================================
##LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
modellr=LogisticRegression(solver='lbfgs')
modellr.fit(x_trainset,y_trainset)
predlr=modellr.predict(x_testset)
## score/accuracy
from sklearn import metrics
print("Logistic's Accuracy: ", metrics.accuracy_score(y_testset, predlr))

###====================================================================
##KNN
from sklearn.neighbors import KNeighborsClassifier
k = 11 #paling bagus di gambar pada step evaluation
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_trainset,y_trainset)
predknn = neigh.predict(x_testset)
##score/accuracy
from sklearn import metrics
print("KNN's Accuracy: ", metrics.accuracy_score(y_testset, predknn))
'''
## Evaluation buat KNN
Ks = round((len(x_testset)+len(x_trainset))**.5) ##akar dari jumlah data
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_trainset,y_trainset)
    yhat=neigh.predict(x_testset)
    mean_acc[n-1] = metrics.accuracy_score(y_testset, yhat)
    std_acc[n-1]=np.std(yhat==y_testset)/np.sqrt(yhat.shape[0])
##plot accuracy
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
'''

# ###====================================================================
# ###CROSSVALL
from sklearn.model_selection import cross_val_score
acctree = cross_val_score(fifaTree,x_trainset,y_trainset,cv=kf) #kf udah diassigned di atas
print('Cross_Val_Score untuk Decision Tree',acctree.mean())
accreg = cross_val_score(modellr,x_trainset,y_trainset,cv=kf) #kf udah diassigned di atas
print('Cross_Val_Score untuk Linear Regression',accreg.mean())
accknn = cross_val_score(neigh,x_trainset,y_trainset,cv=kf) #kf udah diassigned di atas
print('Cross_Val_Score untuk KNN',accknn.mean())

