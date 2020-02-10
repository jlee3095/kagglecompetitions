# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact
from sklearn.preprocessing import OrdinalEncoder


train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

train.head()


xtest=test[test.columns[1:]]

train.info()

# +
ytrain=train['SalePrice']
train.shape, test.shape

#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()
# -

f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(),vmin=.2,square=True,mask=mask);


# +
#top correlated variables

s=train.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
corr=pd.DataFrame(s)
corr=corr.rename(columns={0: 'abs(r)'})
corr[corr['abs(r)']>0.5]
# -

#top correlated variables with respect to salesprice
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

#About the highest R value OverallQual
# this is an ordinal categorical variable represented by integers
np.sort(train.OverallQual.unique())

#Boxplots to show trends of Saleprice vs Overall Qual
bplot_data=train[['SalePrice', 'OverallQual']]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=bplot_data)
plt.xlabel('OverallQual',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
plt.title('Boxplots of Overall Qual')

#Continuous variable
train.GrLivArea

#There are two outliers GrLvArea>4000, but generally there is a linear relationship
GrLiv_data=train[['SalePrice', 'GrLivArea']]
GrLiv_data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000));
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

#Possible candidates for outliers
train[(GrLiv_data['GrLivArea']>4000) & (GrLiv_data['SalePrice']<200000) ][['GrLivArea','OverallQual','SalePrice']]

# ## What to do with missing values
# (A) Impute missing values with the total data set (test and train)
#
# (B) Impute missing values with train set and use train set values on train/test set
#
# Since each data set is drawn from the same population (assumption of learning)
# The statistics describing each sample data should be similiar
# and so to prevent "poisoning" the analysis with test data I choose (B)
#
# NOTE: there are some missing values in test not found in train

# +
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
# -

print(misval.shape[0]," total missing values" )
print((misval['misval_train']!=0).sum(), "missing values in training set")
print((misval['misval_test']!=0).sum(), "missing values in test set")
misval

# ## Missing Values Part 1: Filling in NA that are not actually missing values but a description of data
#
# NA can be a descriptor of the data and not a missing value. For example NA for PoolQC describes No Pool
#
# There are 14 categories that match this description. 5 of which are basement qualities and 4 of which are garage qualities. I will handle all the non basement and garage qualities first.

# !grep -B8 NA data_description.txt
#14 with NA in description 5 basement qual 4 garage qual


# +
# impute NA to category NA
cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

train['MiscFeature'].unique(),xtest['MiscFeature'].unique()
# -

# ### Additional comments: 
# NA's for any of these categories could also still mean missing data. If there is additional data(in this case there is for Fireplace and Pool), validate the assumption that NA= a column descriptor and not missing data. 

#NA and 0's match there are no NAs that are missing values!
print(xtest[(xtest['Fireplaces']==0) & (xtest['FireplaceQu']!='NA')].shape[0],
      train[(train['Fireplaces']==0) & (train['FireplaceQu']!='NA')].shape[0])

# +
#Three missing values for Poolarea
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

xtest[(xtest['PoolQC']== 'NA') & (xtest['PoolArea']>0)][['OverallQual', 'PoolQC', 'PoolArea']]

# +
fig = sns.catplot(y='OverallQual', x="PoolQC",order=["NA", "Fa","TA", "Gd","Ex"], data=train)
fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)


# -

#slight increase in quality vs pool qc-- give FA for all missing data
xtest['PoolQC']=xtest['PoolQC'].replace({'NA':'Fa'})
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

# ## Missing Values Part 2: Garage qualities

# +
## Training set: 81 entries with no garage 

cols=['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
print(train.filter(regex='Garage').isna().sum())
print('total number of entries with misvals is',
      pd.isnull(train[cols].any(axis=1)).sum())
# -

#Double check to make sure that all entries with no garage have zero values for 
#garagecars and garage area
train[pd.isnull(train['GarageType'])][['GarageCars','GarageArea']].sum()

# +
print(xtest.filter(regex='Garage').isna().sum())

#look at all the entries with missing garagetype, check if the other entries are zero or NA)

print(xtest[pd.isnull(xtest['GarageType'])][cols].notna().sum().sum(), 
xtest[pd.isnull(xtest['GarageType'])][['GarageCars','GarageArea']].sum().sum())

#76 entries confirmed with no garage

#This confirms that there may be 2 entries with missval
print(xtest[pd.isnull(xtest['GarageYrBlt'])][cols].notna().sum())
print('The two potential misvals occur at',xtest[pd.isnull(xtest['GarageYrBlt'])]['GarageType'].notna().nonzero()[0])


xtest[pd.isnull(xtest['GarageYrBlt'])].filter(regex='Garage').iloc[[33, 55]]
#xtest.iloc[[666,1116]].fillna('NANA')
# -

cols= ['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea']
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].mode()[0]} 
    xtest[cols[i]].loc[[666,1116]]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])


#Cannot use 0 for area! use 440 as the next most common value.
print(train['GarageArea'].value_counts())
xtest.loc[1116,'GarageArea']=440
xtest.loc[[666,1116]].filter(regex='Garage')

# +
#Need to remove all NA's before running imputation

cols= ['GarageType','GarageFinish','GarageQual','GarageCond','GarageYrBlt']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())
# -

# ## Missing Values Part 3: Basement Qualities

cols=train.filter(regex='Bsmt').loc[:, train.filter(regex='Bsmt').isnull().any()].columns


# +
print(train.filter(regex='Bsmt').isna().sum())
print('total number of entries in training set with misvals is',
      pd.isnull(train.filter(regex='Bsmt')).any(axis=1).sum())

#Total of 39 entries, there is 37 common entries that have missing values across
#and there are two unique entries that have misvals in exposure and type2

# -

train[pd.isnull(train[cols]).any(axis=1)].filter(regex='Bsmt').loc[[332,948]]

# +
#Impute 332 to Rec, it is not unfinished because all 3 bsmtfinSF >0
print(train['BsmtFinType2'].value_counts())
#Impute 948 to No
print(train['BsmtExposure'].value_counts())

train.loc[332,'BsmtFinType2']='Rec'
train.loc[948,'BsmtExposure']='No'
train.loc[[332,948]].filter(regex='Bsmt')

# +
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
#temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]
# -

#And then impute the Missing values for the 7 other values
temp=xtest[pd.isnull(xtest[cols]).any(axis=1)].filter(regex='Bsmt')
cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
temp[~temp[cols].isnull().all(axis=1)]

# +
#Impute missing values for the 7 entries
cols= ['BsmtQual','BsmtCond','BsmtExposure']
ind=temp[~temp[cols].isnull().all(axis=1)].index
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].value_counts().keys()[0]} 
    xtest[cols[i]].loc[ind]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

#Check if it worked    
xtest[cols].loc[ind]

# +
#Fill in the NA values
cols=xtest.filter(regex='Bsmt').columns.tolist()

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')
#Check to see that there are no NA values
(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())
# -

# ## Missing Values Part 4: Filling in rest of missing vals
#
#

# +
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# +
#Will run simple imputation of mode

train['LotFrontage'].hist()
plt.title('LotFrontage');
plt.ylabel('Counts');
plt.xlabel('LotFrontage')
LF_mode=train['LotFrontage'].mode()[0]

train['LotFrontage']=train['LotFrontage'].fillna(LF_mode)
xtest['LotFrontage']=xtest['LotFrontage'].fillna(LF_mode)
# A second Pass could entail looking at lot variables, street variables, to learn regression for lot frontage
#plt.scatter(x='LotArea', y='LotFrontage', data=train);
#axes = plt.axes()
#axes.set_xlim([0, 100000])

# +
cols=['MasVnrArea','MasVnrType']

train[(train['MasVnrArea']<20) & (train['MasVnrArea']>0)][cols]
xtest[(xtest['MasVnrArea']<20) & (xtest['MasVnrArea']>0)][cols]


#Lets assume that the MasVnrArea of 1 is actually meant to be 0

# +
#Lets assume MassVnrtype is None for the two cases where MasVnrArea=0
train[(train['MasVnrArea']==0) &(train['MasVnrType']!='None')][cols]
xtest[(xtest['MasVnrArea']==0) &(xtest['MasVnrType']!='None')][cols]
train.loc[[688,1241],cols[1]]= 'None'
xtest.loc[[859],cols[1]]= 'None'

#Lets also assume that all NA's are 'None and zero'
train[train[cols].isnull().all(axis=1)][cols]
xtest[xtest[cols].isnull().all(axis=1)][cols]
train[cols[0]].value_counts()

train[cols[0]]=train[cols[0]].fillna(0)
xtest[cols[0]]=xtest[cols[0]].fillna(0)
train[cols[1]]=train[cols[1]].fillna('None')
xtest[cols[1]]=xtest[cols[1]].fillna('None')


# +
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval
# -

cols=misval.index.tolist()
for i in range(len(cols)):
    #print(xtest[cols[i]].fillna(xtest[cols[i]].value_counts()[0]))
    xtest[cols[i]]=xtest[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    train[cols[i]]=train[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    

# +
## NO missing Values!

misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval
# -

# ## Encoding of Variables
#
# Will have to change categorical values to numerical values. 
#
# After numerical conversion, they will be split into ordinal or one hot encoding.
#
# *One hot encoding will be conducted for linear regression. However, for random forest it reduces accuracy so will leave as numerical values

import re
file=open('./categorical.txt','r')
stringlist=file.read().splitlines()

dictlist=[]
for i in range(len(stringlist)):
    value=[]
    #if there is no spaces in the line entry, add as key
    if re.match('^\S', stringlist[i]):
        key=stringlist[i]
        n=i
        #add all entries that have a space as a value 
        for j in range(len(stringlist[n+1:])):
            if re.match('^\s',stringlist[n+1:][j]):
                value.append(stringlist[n+1:][j].split(maxsplit=1)[0])  
        #break until the next key is observed
            elif re.match('^\S',stringlist[n+1:][j]):
                break       
        dictlist.append({key: value})


dictlist[13]

#dictlist.pop(0,14,13) remove all the numerical values
len(dictlist)

dictlist

for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value)}
    print(key)
    print(scale_mapper)
    train[key]=train[key].replace(scale_mapper)
    xtest[key]=xtest[key].replace(scale_mapper)
#train[key]

error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[9]
train[x].unique()

# +

#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

#remap the misspelled variable names
scale_mapper= {'Brk Cmn': 2}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()
# -

#Test to make sure that that the encoding worked
temp=pd.read_csv("./train.csv")
x='Functional'
temp[x].value_counts(),train[x].value_counts()

xtest.info()

xtest['Condition2'].unique()

fig = sns.boxplot(y='SalePrice', x="MSSubClass", data=train)
#MSsubclass is not ordinal

# +
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain.head()

# +
#Make sure that we are correctly binning everytime this cell is run
x='YearBuilt'

r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# +
x='MoSold'

r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)


GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# +
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3]).astype('int64')

x='MoSold'
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)


# +
#Replace yr sold with num values 2006: 0,2007:1,2008:2,2009:3, 2010:4
# will need to one hot encode for regression
x='YrSold'


r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

scale_mapper= {2006: 0,2007:1,2008:2,2009:3, 2010:4}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# +
x='YearRemodAdd'




GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

train[x].unique()


# +
#Bin Yearemodadd by [1949,1960,1970,1980,1990,2000,2010]


arf=pd.DataFrame()
bins=[1949,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5]).astype('int64')




# -

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

temp['GarageYrBlt'].unique()

# +
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1900,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
# -

bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
xtest.loc[xtest[x] == -9223372036854775808, x]=6

xtest[x]=xtest[x].astype('int64')
xtest[x].unique()

# +
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].astype('int64')
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# +
#Change rest of the objects into int64, Now everything is numerically encoded

x=['Utilities','ExterQual','KitchenQual','Functional','Condition2','Electrical','Fence','PoolQC']

train[x]=train[x].astype('int64')
xtest[x]=xtest[x].astype('int64')
# -

train.info()
xtest.info()

# +

hotcols=
numcols=
intcols=

# -



# +
#Try random forest and see results. One hot encode other values for regression
ytrain=train['SalePrice']
xtrain=train[train.columns[1:80]]

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=20, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_test), (y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_train), (y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))


#Model is overfit, hyper parameter tuning leads to the following at best
#remove features instead
# -

feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

ytrain

# +
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(ytrain)
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    
"""
    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr)) 
          """
# -

cvscores

plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# +
plt.scatter((y_test),(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)
# -

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_traind


train.to_csv('train_mod.csv',index=False)
xtest.to_csv('test_mod.csv',index=False)

# +
#Linear Regression---> Onehot non nominal variables, 
# -

xtrain=pd.read_csv


