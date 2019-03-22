import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv('data.csv')
print(list(df))
df_n=df[['Name','Age','Overall','Potential']] #buat df lebih mudah dilihat


xao=df_n['Age'][(df_n['Age']<=25)&(df_n['Overall']>=80)&(df_n['Potential']>=80)]
yao=df_n['Overall'][(df_n['Age']<=25)& (df_n['Overall']>=80)&(df_n['Potential']>=80)]
yap=df_n['Potential'][(df_n['Age']<=25)& (df_n['Potential']>=80)&(df_n['Overall']>=80)]
ind_xao=xao.index.tolist()
xao_bukan=df_n['Age'].loc[~df_n.index.isin(ind_xao)] #bukan target
yao_bukan=df_n['Overall'].loc[~df_n.index.isin(ind_xao)] #bukan target
yap_bukan=df_n['Potential'].loc[~df_n.index.isin(ind_xao)] #bukan target

## Plot Age vs Overall dan Age vs Potential
plt.figure(figsize=(13,8))
ax=plt.subplot(121)
plt.scatter(xao,yao,color='green',label='Target')
plt.scatter(xao_bukan,yao_bukan,color='red',label='Non-Target')
plt.ylabel('Overall')
plt.legend()
plt.xlabel('Age')
plt.grid(True)
ax.set_title("Age vs Overall")
bx=plt.subplot(122)
plt.scatter(xao,yap,color='green',label='Target')
plt.scatter(xao_bukan,yap_bukan,color='red',label='Non-Target')
plt.ylabel('Overall')
plt.xlabel('Age')
plt.legend()
bx.set_title("Age vs Potential")
plt.grid(True)
plt.show()