#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("Chennai houseing sale.csv")


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[8]:


df["QS_OVERALL"]=df["QS_OVERALL"].fillna(df["QS_OVERALL"].mean())


# In[9]:


df.isnull().sum()


# In[10]:


chn=df


# In[11]:


chn.head()


# In[12]:


chn.dropna(axis=0,inplace =True)


# In[13]:


chn.isnull().sum()


# In[14]:


import seaborn as sns


# In[15]:


sns.pairplot(chn,hue="SALES_PRICE")


# In[16]:


chn.shape


# In[17]:


cor=chn.corr()


# In[18]:


sns.heatmap(cor)


# In[19]:


chn


# In[20]:


chn["PARK_FACIL"]=np.where(chn["PARK_FACIL"]=="Yes",1,0)


# In[21]:


chn


# In[22]:


chn["SALE_COND"].unique()


# In[23]:


chennai={k:i for i,k in enumerate(chn["UTILITY_AVAIL"].unique(),0)}
chn["UTILITY_AVAIL"]=chn["UTILITY_AVAIL"].map(chennai)


# In[24]:



chennai={k:i for i,k in enumerate(chn["SALE_COND"].unique(),0)}
chn["SALE_COND"]=chn["SALE_COND"].map(chennai)


# In[25]:


chn


# In[26]:


chennai={k:i for i,k in enumerate(chn["UTILITY_AVAIL"].unique(),0)}
chn["UTILITY_AVAIL"]=chn["UTILITY_AVAIL"].map(chennai)


# In[27]:


chennai={k:i for i ,k in enumerate(chn["MZZONE"].unique(),0)}
chn["MZZONE"]=chn["MZZONE"].map(chennai)


# In[28]:


chn


# In[29]:


street={k:i for i ,k in enumerate(chn["STREET"].unique(),0)}
chn["STREET"]=chn["STREET"].map(street)


# In[30]:


street


# In[31]:


chn.drop("DATE_SALE",axis=1)


# In[32]:


area={k:i for i ,k in enumerate(chn["AREA"].unique(),0)}
chn["AREA"]=chn["AREA"].map(area)


# In[33]:


chn


# In[34]:


news=chn.drop(columns=['PRT_ID', 'DATE_SALE','DATE_BUILD'],axis=1)


# In[35]:



build={k:i for i ,k in enumerate(chn["BUILDTYPE"].unique(),0)}
news["BUILDTYPE"]=news["BUILDTYPE"].map(build)


# In[36]:


news.head()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plot


# In[38]:


sns.barplot(x="COMMIS",y="SALES_PRICE",data=news)


# In[39]:


sns.countplot(data=news,x="SALES_PRICE")


# In[40]:


news


# In[41]:


sns.histplot(data=news,x="REG_FEE",kde=True)


# In[42]:


sns.histplot(data=news,x="COMMIS",kde=True)


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[45]:


scaling=StandardScaler()


# In[48]:


news[["REG_FEE","COMMIS"]]=scaling.fit_transform(news[["REG_FEE","COMMIS"]])


# In[49]:


news


# In[77]:


news.describe()


# In[50]:


x =news.drop("SALES_PRICE",axis=1)
y=news["SALES_PRICE"]


# In[51]:


x.shape,y.shape


# In[83]:


news.head()


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[86]:


from sklearn.feature_selection import f_regression,SelectKBest


# In[87]:


fs=SelectKBest(score_func=f_regression,k='all')
fit=fs.fit(x,y)


# In[88]:


print(np.round(fit.pvalues_,4))
print(np.round(fit.pvalues_,3))


# In[89]:


# feature selection using anova

features_score=pd.DataFrame(fit.scores_)
features_pvalue=pd.DataFrame(np.round(fit.pvalues_,4))
features=pd.DataFrame(x.columns)
feature_score=pd.concat([features,features_score,features_pvalue],axis=1)
feature_score.columns=["input_features","f_score","p_values"]
print(feature_score.nlargest(15,columns="f_score"))


# In[90]:


features_score=pd.DataFrame(fit.scores_)


# In[91]:


features_pvalue=pd.DataFrame(np.round(fit.pvalues_,4))


# In[92]:


features=pd.DataFrame(x.columns)


# In[93]:


feature_score=pd.concat([features,features_score,features_pvalue],axis=1)


# In[94]:


feature_score.columns=["input_features","f_score","p_values"]


# In[95]:


print(feature_score.nlargest(15,columns="f_score"))


# In[ ]:





# In[ ]:





# # GCV DTR

# In[137]:


from sklearn.tree import DecisionTreeRegressor


# In[138]:


depth  =list(range(3,30))
param_grid =dict(max_depth =depth)
tree =GridSearchCV(DecisionTreeRegressor(),param_grid,cv =10)
tree.fit(x_train,y_train)


# In[139]:


y_train_pred =tree.predict(x_train) 
y_test_pred =tree.predict(x_test)


# In[140]:


r2_score(y_train.values, y_train_pred)


# In[141]:


r2_score(y_test, y_test_pred)


# # GCV KNNR

# In[133]:


k_range = list(range(1, 30))
params = dict(n_neighbors = k_range)
knn_regressor = GridSearchCV(KNeighborsRegressor(), params, cv =10, scoring = 'neg_mean_squared_error')
knn_regressor.fit(x_train, y_train)


# In[134]:


y_train_pred =knn_regressor.predict(x_train)
y_test_pred =knn_regressor.predict(x_test)


# In[135]:


r2_score(y_train.values, y_train_pred)


# In[136]:


r2_score(y_test, y_test_pred)


# # RCV RFR

# In[164]:


tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 20, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)
random_regressor.fit(x_train, y_train)


# In[170]:


from sklearn.metrics import r2_score


# In[171]:


y_train_pred = random_regressor.predict(x_train)
y_test_pred = random_regressor.predict(x_test)


# In[180]:


r2_score(y_train, y_train_pred)


# In[183]:


r2_score(y_test, y_test_pred)


# In[184]:


print('MAE:', metrics.mean_absolute_error(y_test,y_test_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# # normal methods

# In[96]:



from sklearn.ensemble import RandomForestRegressor
rfg=RandomForestRegressor()


# In[97]:


rfg.fit(x_train,y_train)


# In[98]:


y_pred=rfg.predict(x_test)


# In[99]:


y_pred


# In[104]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# In[105]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[106]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from math import sqrt
import matplotlib.pyplot as plt

    


# In[190]:


rmse_val = [] 
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = r2_score(y_test,pred) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[74]:


curve = pd.DataFrame(rmse_val)
curve.plot()


# In[75]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_


# In[76]:


from sklearn.model_selection import RandomizedSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11]}

knn = neighbors.KNeighborsRegressor()

model = RandomizedSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_


# In[143]:


from sklearn.linear_model import LinearRegression


# In[144]:


LR=LinearRegression()


# In[155]:


LR.fit(x_train,y_train)


# In[159]:


LR.score(x_train,y_train)


# In[158]:


LR.score(x_test,y_test)


# In[124]:


from sklearn.metrics import accuracy_score


# In[149]:


y_pred=LR.predict(x_test)


# In[187]:


mean_squared_error(y_test,y_pred)


# In[118]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[153]:


sns.distplot(y_pred)


# In[154]:


plt.scatter(y_test,y_pred)


# In[161]:


from sklearn import metrics


# In[162]:


print('MAE:', metrics.mean_absolute_error(y_test,y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[189]:


linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
linear_regression.predict(x_test)

from sklearn.model_selection import cross_val_score
print(cross_val_score(linear_regression, x, y, cv=10, scoring="r2").mean())


# In[ ]:




