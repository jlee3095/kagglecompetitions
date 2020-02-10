# @@ Cell 1
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact
from sklearn.preprocessing import OrdinalEncoder

# @@ Cell 3
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 4
train.head()

# @@ Cell 5
train.shape, test.shape

# @@ Cell 6
xtest=test[test.columns[1:]]

# @@ Cell 7
train.info()

# @@ Cell 8
xtest

# @@ Cell 9
#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()

# @@ Cell 10
#Salesprice is skewed to the right
train.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
train.describe()

# @@ Cell 11
train.head()
ytrain=train['SalePrice']
train.shape, test.shape

# @@ Cell 12
ytrain=train['SalePrice']
train.shape, test.shape

# @@ Cell 13
train.head()

# @@ Cell 14
#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()

# @@ Cell 15
f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(),vmin=.2,square=True,mask=mask);

# @@ Cell 16
s=train.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
corr=pd.DataFrame(s)
corr=corr.rename(columns={0: 'abs(r)'})
corr[corr['abs(r)']>0.5]

# @@ Cell 17
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 18
#top correlated variables with respect to salesprice
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 19
#About the highest R value OverallQual
# this is an ordinal categorical variable represented by integers
np.sort(train.OverallQual.unique())

# @@ Cell 20
#Boxplots to show trends of Saleprice vs Overall Qual
bplot_data=train[['SalePrice', 'OverallQual']]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=bplot_data)
plt.xlabel('OverallQual',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
plt.title('Boxplots of Overall Qual')

# @@ Cell 21
#Continuous variable
train.GrLivArea

# @@ Cell 22
#There are two outliers GrLvArea>4000, but generally there is a linear relationship
GrLiv_data=train[['SalePrice', 'GrLivArea']]
GrLiv_data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000));
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 23
#Possible candidates for outliers
train[(GrLiv_data['GrLivArea']>4000) & (GrLiv_data['SalePrice']<200000) ][['GrLivArea','OverallQual','SalePrice']]

# @@ Cell 24
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)

# @@ Cell 25
print(misval.shape[0]," total missing values" )
print((misval['misval_train']!=0).sum(), "missing values in training set")
print((misval['misval_test']!=0).sum(), "missing values in test set")
misval

# @@ Cell 26
get_ipython().system('grep -B8 NA data_description.txt')
#14 with NA in description 5 basement qual 4 garage qual

# @@ Cell 27
# impute NA to category NA
cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

train['MiscFeature'].unique(),xtest['MiscFeature'].unique()

# @@ Cell 28
#NA and 0's match there are no NAs that are missing values!
print(xtest[(xtest['Fireplaces']==0) & (xtest['FireplaceQu']!='NA')].shape[0],
      train[(train['Fireplaces']==0) & (train['FireplaceQu']!='NA')].shape[0])

# @@ Cell 29
#Three missing values for Poolarea
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

xtest[(xtest['PoolQC']== 'NA') & (xtest['PoolArea']>0)][['OverallQual', 'PoolQC', 'PoolArea']]

# @@ Cell 30
fig = sns.catplot(y='OverallQual', x="PoolQC",order=["NA", "Fa","TA", "Gd","Ex"], data=train)
fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 31
#slight increase in quality vs pool qc-- give FA for all missing data
xtest['PoolQC']=xtest['PoolQC'].replace({'NA':'Fa'})
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

# @@ Cell 32
## Training set: 81 entries with no garage 

cols=['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
print(train.filter(regex='Garage').isna().sum())
print('total number of entries with misvals is',
      pd.isnull(train[cols].any(axis=1)).sum())

# @@ Cell 33
#Double check to make sure that all entries with no garage have zero values for 
#garagecars and garage area
train[pd.isnull(train['GarageType'])][['GarageCars','GarageArea']].sum()

# @@ Cell 34
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

# @@ Cell 35
cols= ['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea']
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].mode()[0]} 
    xtest[cols[i]].loc[[666,1116]]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

# @@ Cell 36
#Cannot use 0 for area! use 440 as the next most common value.
print(train['GarageArea'].value_counts())
xtest.loc[1116,'GarageArea']=440
xtest.loc[[666,1116]].filter(regex='Garage')

# @@ Cell 37
#Need to remove all NA's before running imputation

cols= ['GarageType','GarageFinish','GarageQual','GarageCond','GarageYrBlt']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 38
cols=train.filter(regex='Bsmt').loc[:, train.filter(regex='Bsmt').isnull().any()].columns

# @@ Cell 39
print(train.filter(regex='Bsmt').isna().sum())
print('total number of entries in training set with misvals is',
      pd.isnull(train.filter(regex='Bsmt')).any(axis=1).sum())

#Total of 39 entries, there is 37 common entries that have missing values across
#and there are two unique entries that have misvals in exposure and type2

# @@ Cell 40
train[pd.isnull(train[cols]).any(axis=1)].filter(regex='Bsmt').loc[[332,948]]

# @@ Cell 41
#Impute 332 to Rec, it is not unfinished because all 3 bsmtfinSF >0
print(train['BsmtFinType2'].value_counts())
#Impute 948 to No
print(train['BsmtExposure'].value_counts())

train.loc[332,'BsmtFinType2']='Rec'
train.loc[948,'BsmtExposure']='No'
train.loc[[332,948]].filter(regex='Bsmt')

# @@ Cell 42
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]

# @@ Cell 43
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
#temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]

# @@ Cell 44
#And then impute the Missing values for the 7 other values
temp=xtest[pd.isnull(xtest[cols]).any(axis=1)].filter(regex='Bsmt')
cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
temp[~temp[cols].isnull().all(axis=1)]

# @@ Cell 45
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]

# @@ Cell 46
#Impute missing values for the 7 entries
cols= ['BsmtQual','BsmtCond','BsmtExposure']
ind=temp[~temp[cols].isnull().all(axis=1)].index
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].value_counts().keys()[0]} 
    xtest[cols[i]].loc[ind]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

#Check if it worked    
xtest[cols].loc[ind]

# @@ Cell 47
#Fill in the NA values
cols=xtest.filter(regex='Bsmt').columns.tolist()

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')
#Check to see that there are no NA values
(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 48
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 49
#Will run simple imputation of mode

train['LotFrontage'].hist()
plt.title('LotFrontage');
plt.ylabel('Counts');
plt.xlabel('LotFrontage')
LF_mode=xtrain['LotFrontage'].mode()[0]

train['LotFrontage']=train['LotFrontage'].fillna(LF_mode)
xtest['LotFrontage']=xtest['LotFrontage'].fillna(LF_mode)
# A second Pass could entail looking at lot variables, street variables, to learn regression for lot frontage
#plt.scatter(x='LotArea', y='LotFrontage', data=train);
#axes = plt.axes()
#axes.set_xlim([0, 100000])

# @@ Cell 50
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

# @@ Cell 51
cols=['MasVnrArea','MasVnrType']

train[(train['MasVnrArea']<20) & (train['MasVnrArea']>0)][cols]
xtest[(xtest['MasVnrArea']<20) & (xtest['MasVnrArea']>0)][cols]


#Lets assume that the MasVnrArea of 1 is actually meant to be 0

# @@ Cell 52
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

# @@ Cell 53
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 54
cols=misval.index.tolist()
for i in range(len(cols)):
    #print(xtest[cols[i]].fillna(xtest[cols[i]].value_counts()[0]))
    xtest[cols[i]]=xtest[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    train[cols[i]]=train[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    

# @@ Cell 55
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

# @@ Cell 56
cols

# @@ Cell 57
import re
file=open('./categorical.txt','r')
stringlist=file.read().splitlines()

# @@ Cell 58
dictlist=[]
for i in range(len(stringlist)):
    value=[]
    if re.match('^\S', stringlist[i]):
        key=stringlist[i]
        n=i
        for j in range(len(stringlist[n+1:])):
            if re.match('^\s',stringlist[n+1:][j]):
                value.append(stringlist[n+1:][j].split(maxsplit=1)[0])                
            elif re.match('^\S',stringlist[n+1:][j]):
                break       
        dictlist.append({key: value})

# @@ Cell 59
dictlist

# @@ Cell 60
dictlist[0]

# @@ Cell 61
# dictlist.pop(0), dictlist.pop(13) remove all the numerical values
len(dictlist)

# @@ Cell 62
stringlist

# @@ Cell 63
stringlist[0]

# @@ Cell 64
stringlist[1]

# @@ Cell 65
dictlist

# @@ Cell 66
dictlist={}
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

# @@ Cell 67
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

# @@ Cell 68
dictlist

# @@ Cell 69
dictlist[0].key

# @@ Cell 70
dictlist[0].keys

# @@ Cell 71
dictlist[0].keys[0]

# @@ Cell 72
dictlist[0].keys

# @@ Cell 73
dictlist[0].keys(0)

# @@ Cell 74
dictlist[0].keys()

# @@ Cell 75
dictlist[0].values()

# @@ Cell 76
dictlist

# @@ Cell 77
dictlist(13)

# @@ Cell 78
dictlist[13]

# @@ Cell 79
dictlist[12]

# @@ Cell 80
dictlist[14]

# @@ Cell 81
dictlist.pop(0)#, dictlist.pop(13) remove all the numerical values
len(dictlist)

# @@ Cell 82
dictlist[14]

# @@ Cell 83
dictlist.pop(14)#, dictlist.pop(13) remove all the numerical values
len(dictlist)

# @@ Cell 84
dictlist[14]

# @@ Cell 85
dictlist[13]

# @@ Cell 86
dictlist.pop(13)#, dictlist.pop(13) remove all the numerical values
len(dictlist)

# @@ Cell 87
dictlist[13]

# @@ Cell 88
dictlist[]

# @@ Cell 89
dictlist

# @@ Cell 90
for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value)}
    print(key)
    print(scale_mapper)
    train[key]=train[key].replace(scale_mapper)
    xtest[key]=xtest[key].replace(scale_mapper)
#train[key]

# @@ Cell 91
# = pd.Series(['A', 'B', 'Aaba', 'Baca', np.nan, 'CABA', 'cat'])
train.str.contains('\S', regex=True)

# @@ Cell 92
fig = sns.boxplot(y='SalePrice', x="MSSubClass", data=train)
#MSsubclass is not ordinal

# @@ Cell 93
train.info()

# @@ Cell 94
train.info()==object

# @@ Cell 95
select_dtypes(exclude=['int'])

# @@ Cell 96
train.select_dtypes(exclude=['int'])

# @@ Cell 97
train.select_dtypes(exclude=['int','float'])

# @@ Cell 98
train.select_dtypes(exclude=['int','float']).info()

# @@ Cell 99
train.select_dtypes(exclude=['int64','float']).info()

# @@ Cell 100
train.select_dtypes(exclude=['int64','float']).info()[0]

# @@ Cell 101
train.select_dtypes(exclude=['int64','float']).info()

# @@ Cell 102
train.select_dtypes(exclude=['int64','float']).columns

# @@ Cell 103
train.select_dtypes(exclude=['int64','float']).columns[0]

# @@ Cell 104
train.select_dtypes(exclude=['int64','float']).columns

# @@ Cell 105
train.select_dtypes(exclude=['int64','float']).columns.aslist()

# @@ Cell 106
train.select_dtypes(exclude=['int64','float']).columns.tolist()

# @@ Cell 107
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols]

# @@ Cell 108
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols].describe()

# @@ Cell 109
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols].unique()

# @@ Cell 110
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols].unique

# @@ Cell 111
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[0]]

# @@ Cell 112
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[0]].unique

# @@ Cell 113
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[0]].unique()

# @@ Cell 114
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[1]].unique()

# @@ Cell 115
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[2]].unique()

# @@ Cell 116
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[3]].unique()

# @@ Cell 117
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[4]].unique()

# @@ Cell 118
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[5]].unique()

# @@ Cell 119
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[6]].unique()

# @@ Cell 120
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[7]].unique()

# @@ Cell 121
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[8]].unique()

# @@ Cell 122
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[9]].unique()

# @@ Cell 123
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[10]].unique()

# @@ Cell 124
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[0]].unique()

# @@ Cell 125
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[9]].unique()

# @@ Cell 126
error_cols

# @@ Cell 127
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
train[error_cols[8]].unique()

# @@ Cell 128
#Test to make sure that that the encoding worked
temp=pd.read_csv("./train.csv")
x='Functional'
temp[x].value_counts(),train[x].value_counts()

# @@ Cell 129
train[error_cols[8]].describe()

# @@ Cell 130
train[error_cols[8]].hist()

# @@ Cell 131
train[error_cols[8]].describe()

# @@ Cell 132
train[error_cols[8]].unique()

# @@ Cell 133
train[error_cols[8]].describe()

# @@ Cell 134
train[error_cols[8]].unique()

# @@ Cell 135
train[error_cols[7]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#

# @@ Cell 136
train[error_cols[6]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#

# @@ Cell 137
train[error_cols[5]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#

# @@ Cell 138
train[error_cols[4]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#

# @@ Cell 139
train[error_cols[3]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances

# @@ Cell 140
train[error_cols[2]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#

# @@ Cell 141
train[error_cols[1]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood 'NAmes'

# @@ Cell 142
train[error_cols[0]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood 'NAmes'
#KitchenQual has in values but shows dtype of object for some reaso

# @@ Cell 143
train[error_cols[0]].unique()[3]
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 144
train[error_cols[0]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 145
train[error_cols[0]].unique()[1]
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 146
train[error_cols[0]].unique()[2]
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 147
train[error_cols[1]]
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 148
train[error_cols[1]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 149
train[error_cols[2]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 150
train[error_cols[2]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 151
train[error_cols[3]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 152
train[error_cols[4]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 153
train[error_cols[0]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 154
train[error_cols[1]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 155
train[error_cols[1]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 156
train[error_cols[2]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 157
train[error_cols[3]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 158
train[error_cols[3]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 159
train[error_cols[4]].value_counts()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 160
train[error_cols[4]].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'

# @@ Cell 161
x=error_cols[4]
train[x].unique()
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Wd Sdng': 15}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 162
x=error_cols[0]
train[x].unique()

# @@ Cell 163
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'C (all)': 1}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 164
x=error_cols[1]
train[x].unique()

# @@ Cell 165
x=error_cols[2]
train[x].unique()

# @@ Cell 166
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'NAmes': 12}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 167
x=error_cols[3]
train[x].unique()

# @@ Cell 168
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'2fmCon': 1}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 169
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Duplex': 2}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 170
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Twnhs': 4}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 171
x=error_cols[4]
train[x].unique()

# @@ Cell 172
x=error_cols[5]
train[x].unique()

# @@ Cell 173
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'CmentBd': 5}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 174
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Wd Sdng': 15}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 175
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Wd Shng': 16}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 176
#GarageYrBlt has instances of NA---FIX
#Functional has int values but shows dytpe of object for some reason
#KitchenQual has in values but shows dtype of object for some reason
#ExterQual ""
#Exter2nd(5) has uncoverted instances
#Exter1st has uncoverted instances
#'Bldgtype' ""
#Neighborhood "" 'NAmes'
#MSZoning ""  'C (all)'


scale_mapper= {'Brk Cmn': 2}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 177
x=error_cols[6]
train[x].unique()

# @@ Cell 178
x=error_cols[7]
train[x].unique()

# @@ Cell 179
x=error_cols[8]
train[x].unique()

# @@ Cell 180
x=error_cols[9]
train[x].unique()

# @@ Cell 181
#Test to make sure that that the encoding worked
temp=pd.read_csv("./train.csv")
x='Functional'
temp[x].value_counts(),train[x].value_counts()

# @@ Cell 182
x

# @@ Cell 183
x

# @@ Cell 184
train['MSZoning'].value_counts()

# @@ Cell 185
train.describe()

# @@ Cell 186
train.info()

# @@ Cell 187
x

# @@ Cell 188
train.select_dtypes(exclude=['int64','float']).columns.tolist()

# @@ Cell 189
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()

# @@ Cell 190
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=err_cols[0]
train[x]

# @@ Cell 191
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[x]

# @@ Cell 192
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[x].unique

# @@ Cell 193
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[x].describe()

# @@ Cell 194
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[1]
train[x].describe()

# @@ Cell 195
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[2]
train[x].describe()

# @@ Cell 196
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[2]
train[x].unique()

# @@ Cell 197
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[1]
train[x].unique()

# @@ Cell 198
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[x].unique()

# @@ Cell 199
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[1]
train[x].unique()

# @@ Cell 200
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[2]
train[x].unique()

# @@ Cell 201
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[3]
train[x].unique()

# @@ Cell 202
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[4]
train[x].unique()

# @@ Cell 203
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[5]
train[x].unique()

# @@ Cell 204
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[4]
train[x].unique()

# @@ Cell 205
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[3]
train[x].unique()

# @@ Cell 206
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[2]
train[x].unique()

# @@ Cell 207
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[1]
train[x].unique()

# @@ Cell 208
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[2]
train[x].unique()

# @@ Cell 209
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[3]
train[x].unique()

# @@ Cell 210
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[x].unique()

# @@ Cell 211
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[0]].unique()

# @@ Cell 212
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[1]].unique()

# @@ Cell 213
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[2]].unique()

# @@ Cell 214
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[3]].unique()

# @@ Cell 215
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[4]].unique()

# @@ Cell 216
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[5]].unique()

# @@ Cell 217
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[6]].unique()

# @@ Cell 218
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[7]].unique()

# @@ Cell 219
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[0]
train[train.columns[8]].unique()

# @@ Cell 220
xtest.info()

# @@ Cell 221
xtest['Electrical']

# @@ Cell 222
xtest['Electrical'].unique()

# @@ Cell 223
xtest['Electrical'].unique()[0]

# @@ Cell 224
xtest['Electrical'].unique()[1]

# @@ Cell 225
type(xtest['Electrical'].unique()[1])

# @@ Cell 226
type(xtest['Electrical'].unique()[0])

# @@ Cell 227
type(xtest['Electrical'].unique()[1])

# @@ Cell 228
type(xtest['Electrical'].unique()[2])

# @@ Cell 229
type(xtest['Electrical'].unique()[3])

# @@ Cell 230
type(xtest['Electrical'].unique()[4])

# @@ Cell 231
type(xtest['Electrical'].unique()[3])

# @@ Cell 232
xtest['Electrical'].unique()

# @@ Cell 233
xtest.info()

# @@ Cell 234
xtest['ExterQual'].unique()

# @@ Cell 235
xtest['PoolQc'].unique()

# @@ Cell 236
xtest['PoolQC'].unique()

# @@ Cell 237
xtest['Functional'].unique()

# @@ Cell 238
xtest['Condition2'].unique()

# @@ Cell 239
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain[xtrain['MSZoning']=='C (all)']['MSZoning'][30] #.value_counts()

# @@ Cell 240
#remove ID number

xtrain=train[train.columns[1:80]]

# @@ Cell 241
#remove ID number

xtrain=train[train.columns[1:80]]
xtrain

# @@ Cell 242
xtrain['YearBuilt'].hist()

# @@ Cell 243
xtrain['RemodAdd'].hist()

# @@ Cell 244
xtrain['YearRemodAdd'].hist()

# @@ Cell 245
xtrain['YearRemodAdd'].value_counts()

# @@ Cell 246
xtrain['YearRemodAdd'].unique()

# @@ Cell 247
GrLiv_data=train[['SalePrice', 'YearRemodAdd']]
GrLiv_data.plot.scatter(x='YearRemodAdd', y='SalePrice',ylim=(0,800000));
plt.xlabel('YearRemodAdd',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 248
GrLiv_data=train[['SalePrice', 'YearRemodAdd']]
GrLiv_data.plot.scatter(x='YearRemodAdd', y='SalePrice',ylim=(0,800000));
plt.xlabel('YearBuilt',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 249
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 250
x='GarageYrBlt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 251
train[x].unique()

# @@ Cell 252
train[x].describe()

# @@ Cell 253
x='YrSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 254
train[x].describe()

# @@ Cell 255
train[x].unique()

# @@ Cell 256
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 257
x='YrSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 258
x='GarageYrBlt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 259
train[x].unique()

# @@ Cell 260
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 261
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)

# @@ Cell 262
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbin=True)

# @@ Cell 263
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)

# @@ Cell 264
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0]

# @@ Cell 265
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[1]

# @@ Cell 266
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0]

# @@ Cell 267
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][0]

# @@ Cell 268
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][1]

# @@ Cell 269
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][2]

# @@ Cell 270
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][3]

# @@ Cell 271
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][4]

# @@ Cell 272
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][5]

# @@ Cell 273
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][6]

# @@ Cell 274
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][6]

# @@ Cell 275
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0][5]

# @@ Cell 276
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0].tolist()

# @@ Cell 277
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True)[0]

# @@ Cell 278
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True,[1,3,5]

# @@ Cell 279
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True,[1,3,5])

# @@ Cell 280
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,retbins=True,bins=[1,3,5])

# @@ Cell 281
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,bins=[1,3,5])

# @@ Cell 282
pd.cut(np.array([1, 7, 5, 4, 6, 3]),bins=[1,3,5])

# @@ Cell 283
pd.cut(np.array([1, 7, 5, 4, 6, 3]),bins=[1,2,3,5])

# @@ Cell 284
pd.cut(train[x], 3)

# @@ Cell 285
pd.cut(train[x], 5)

# @@ Cell 286
pd.cut(train[x], 5,retbins=True)

# @@ Cell 287
x='YrSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 288
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 289
pd.cut(train[x], 7,retbins=True)

# @@ Cell 290
pd.cut(train[x], 8,retbins=True)

# @@ Cell 291
pd.cut(train[x], 10,retbins=True)

# @@ Cell 292
pd.cut(train[x], 12,retbins=True)

# @@ Cell 293
pd.cut(train[x], 13,retbins=True)

# @@ Cell 294
pd.cut(train[x], 14,retbins=True)

# @@ Cell 295
bins=[1870,1920,1960,1970,1980,1990,2010]

# @@ Cell 296
pd.cut(train[x])

# @@ Cell 297
pd.cut(train[x],bins)

# @@ Cell 298
pd.cut(train[x],bins).hist()

# @@ Cell 299
bins=[1870,1920,1960,1970,1980,1990,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins).hist()

# @@ Cell 300
bins=[1870,1920,1960,1970,1980,1990,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)

# @@ Cell 301
bins=[1870,1920,1960,1970,1980,1990,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf

# @@ Cell 302
bins=[1870,1920,1960,1970,1980,1990,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

# @@ Cell 303
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

# @@ Cell 304
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

scale_mapper= {(1920, 1960]  : 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 305
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

scale_mapper= {'(1920, 1960]'  : 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 306
x

# @@ Cell 307
x.columns()

# @@ Cell 308
x.columns

# @@ Cell 309
x

# @@ Cell 310
train[x]

# @@ Cell 311
arf

# @@ Cell 312
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

scale_mapper= {'(1920, 1960]'  : 0}
    
arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 313
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

scale_mapper= {(1920, 1960]  : 0}
    
arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 314
arf[0]

# @@ Cell 315
arf.value_counts()

# @@ Cell 316
arf.value_counts()[9]

# @@ Cell 317
arf.value_counts()

# @@ Cell 318
arf.value_counts()[0]

# @@ Cell 319
arf.value_counts()[1]

# @@ Cell 320
arf.value_counts()

# @@ Cell 321
arf.value_counts().columns

# @@ Cell 322
arf.unique()

# @@ Cell 323
arf.unique()[0]

# @@ Cell 324
arf.unique()[9]

# @@ Cell 325
arf.unique()[8]

# @@ Cell 326
arf.unique()[7]

# @@ Cell 327
arf.unique()[6]

# @@ Cell 328
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins)
arf.value_counts()

scale_mapper= {(arf.unique()[6]  : 0}
    
arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 329
len(bins)

# @@ Cell 330
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=len(bins)-1)
arf.value_counts()

scale_mapper= {(arf.unique()[6]  : 0}
    
arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 331
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=len(bins)-1)
arf.value_counts()

#scale_mapper= {(arf.unique()[6]  : 0}
    
arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 332
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=len(bins)-1)
arf.value_counts()

#scale_mapper= {(arf.unique()[6]  : 0}
    
#arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 333
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=7)
arf.value_counts()

#scale_mapper= {(arf.unique()[6]  : 0}
    
#arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 334
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6,7])
arf.value_counts()

#scale_mapper= {(arf.unique()[6]  : 0}
    
#arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 335
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6])
arf.value_counts()

#scale_mapper= {(arf.unique()[6]  : 0}
    
#arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 336
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6])
arf.hist()

#scale_mapper= {(arf.unique()[6]  : 0}
    
#arf[x]=arf[x].replace(scale_mapper)
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 337
bins=[1870,1920,1960,1970,1980,1990,2000,2010]

arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6])
arf.hist()

train[x]=arf
#xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 338
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 339
train[x]

# @@ Cell 340
train[x].astype(int64)

# @@ Cell 341
train[x].astype(int)

# @@ Cell 342
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 343
train[x]=train[x].astype(int)

# @@ Cell 344
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 345
train[x]=train[x].astype('int64')

# @@ Cell 346
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 347
x='Mosold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 348
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 349
#Bin Months

arf=pd.DataFrame()
arf=pd.cut(train[x],4,labels=[0,1,2,3])
arf
#train[x]=arf

# @@ Cell 350
#Bin Months

arf=pd.DataFrame()
arf=pd.cut(train[x],4)
arf
#train[x]=arf

# @@ Cell 351
#Bin Months

arf=pd.DataFrame()
bins=[1,2,3,4]
arf=pd.cut(train[x],bins,labels=bins)
arf
#train[x]=arf

# @@ Cell 352
#Bin Months

arf=pd.DataFrame()
bins=[0,1,2,3,4]
arf=pd.cut(train[x],bins,labels=bins)
arf
#train[x]=arf

# @@ Cell 353
#Bin Months

arf=pd.DataFrame()
bins=[0,1,2,3,4]
arf=pd.cut(train[x],bins)
arf
#train[x]=arf

# @@ Cell 354
train[x]

# @@ Cell 355
#Bin Months

arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins)
arf
#train[x]=arf

# @@ Cell 356
#Bin Months

arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
arf
#train[x]=arf

# @@ Cell 357
#Bin Months

arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
train[x]=arf

# @@ Cell 358
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 359
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
train[x]=arf.astype('int64')

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 360
train[x]

# @@ Cell 361
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
#train[x]=arf.astype('int64')

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 362
temp=pd.read_csv("./train.csv")
temp['MoSold']

# @@ Cell 363
temp=pd.read_csv("./train.csv")
temp['MoSold']=train[x]

# @@ Cell 364
train[x]

# @@ Cell 365
tempsold

# @@ Cell 366
train[x]

# @@ Cell 367
temp['MoSold']

# @@ Cell 368
temp['MoSold']

# @@ Cell 369
temp

# @@ Cell 370
temp['MoSold']

# @@ Cell 371
r['MoSold']

# @@ Cell 372
r=pd.read_csv("./train.csv")
#train[x]=temp['MoSold']

# @@ Cell 373
r['MoSold']

# @@ Cell 374
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']

# @@ Cell 375
train[x]

# @@ Cell 376
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 377
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3])
#train[x]=arf.astype('int64')

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 378
arf

# @@ Cell 379
arf.astype(int)

# @@ Cell 380
arf.astype('int64')

# @@ Cell 381
arf

# @@ Cell 382
arf.astype('int64')

# @@ Cell 383
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
#train[x]=arf.astype('int64')

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 384
arf

# @@ Cell 385
#Bin Months
x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 386
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']

# @@ Cell 387
train[x]

# @@ Cell 388
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6])
train[x]=arf

# @@ Cell 389
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 390
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf

# @@ Cell 391
x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 392
r=pd.read_csv("./train.csv")
train[x]=r['YearBuilt']

# @@ Cell 393
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 394
train[x]

# @@ Cell 395
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']

# @@ Cell 396
train[x]

# @@ Cell 397
train[x]==12

# @@ Cell 398
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']

# @@ Cell 399
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 400
train[x]==12
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x]

# @@ Cell 401
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 402
#Bin Months

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 403
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 404
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 405
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 406
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 407
train[x]==12
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 408
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 409
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 410
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 411
train[x]==12
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 412
r=pd.read_csv("./test.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 413
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 414
train[x]==12
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train

# @@ Cell 415
train[x]==12
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x]

# @@ Cell 416
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 417
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[0,3,6,9,12]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
#train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 418
arf

# @@ Cell 419
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
#train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 420
arf

# @@ Cell 421
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 422
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 423
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 424
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 425
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 426
r=pd.read_csv("./train.csv")
train[x]=r['MoSold']
train[x]

# @@ Cell 427
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 428
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
xtrain

# @@ Cell 429
scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x]

# @@ Cell 430
x='MoSold'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 431
#Bin Months(seasons) 12,1,2  3,4,5 6,7,8 9,10,11

x='MoSold'
arf=pd.DataFrame()
bins=[-1,2,6,8,11]
arf=pd.cut(train[x],bins,labels=[0,1,2,3]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3]).astype('int64')

# @@ Cell 432
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 433
data=test[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 434
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 435
xtest[x]

# @@ Cell 436
r=pd.read_csv("./train.csv")
train[x]=r['YearBuilt']
train[x]

# @@ Cell 437
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtext[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 438
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 439
r=pd.read_csv("./train.csv")
train[x]=r['YearBuilt']
train[x]

# @@ Cell 440
r=pd.read_csv("./train.csv")
train[x]=r['YearBuilt']
train[x]

# @@ Cell 441
#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 442
xtest[x].describe()

# @@ Cell 443
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
train[x]

# @@ Cell 444
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

# @@ Cell 445
#Make sure that we are correctly binning everytime this cell is run
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

#Bin Years built

bins=[1870,1920,1960,1970,1980,1990,2000,2010]
x='YearBuilt'
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')



x='YearBuilt'
GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 446
xtest[x].describe()

# @@ Cell 447
xtest[x].unique()

# @@ Cell 448
arf

# @@ Cell 449
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 450
x='MoSold'
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 451
x

# @@ Cell 452
x

# @@ Cell 453
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

# @@ Cell 454
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

# @@ Cell 455
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

# @@ Cell 456
x

# @@ Cell 457
#remove ID number

xtrain=train[train.columns[1:80]]
xtrain

# @@ Cell 458
#remove ID number

xtrain=train[train.columns[1:80]]
train['YearBuilt']

# @@ Cell 459
#remove ID number

xtrain=train[train.columns[1:80]]

# @@ Cell 460
#remove ID number

xtrain=train[train.columns[1:80]]

train

# @@ Cell 461
#remove ID number

xtrain=train[train.columns[1:80]]

train.head()

# @@ Cell 462
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain.head()

# @@ Cell 463
x='YrSold'

r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]


GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 464
x.unique()

# @@ Cell 465
train[x].unique()

# @@ Cell 466
x='YrSold'

r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
xtest[x]

scale_mapper= {2006: 0,2007:1,2008:2,2009:3 2010:4}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 467
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

# @@ Cell 468
x='YearBuilt'
train(x)

# @@ Cell 469
x='YearBuilt'
train[x]

# @@ Cell 470
x='YearBuilt'
train[x],xtest[x]

# @@ Cell 471
x='YearBuilt'
train[x].unique()

# @@ Cell 472
x='YearBuilt'
xtest[x].unique()

# @@ Cell 473
x='YearRemodAdd'
xtest[x].unique()

# @@ Cell 474
x='YearRemodAdd'
train[x].unique()

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 475
x='YearRemodAdd'

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

train[x].unique()

# @@ Cell 476
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5,6]
arf=pd.cut(train[x],bins).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins).astype('int64')

# @@ Cell 477
train.unique()

# @@ Cell 478
train(x).unique()

# @@ Cell 479
train[x].unique()

# @@ Cell 480
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5,6]
arf=pd.cut(train[x],bins,labels).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 481
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5,6]
arf=pd.cut(train[x],bins,labels) #.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 482
arf

# @@ Cell 483
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5]
arf=pd.cut(train[x],bins,labels) #.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 484
arf

# @@ Cell 485
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]) #.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 486
arf

# @@ Cell 487
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]) #.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 488
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5]
arf=pd.cut(train[x],bins,labels).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 489
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5]
arf=pd.cut(train[x],bins,labels)#.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 490
arf

# @@ Cell 491
arf

# @@ Cell 492
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]
labels=[0,1,2,3,4,5]
arf=pd.cut(train[x],bins,labels)#.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 493
arf

# @@ Cell 494
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5])#.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 495
arf

# @@ Cell 496
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]).astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels).astype('int64')

# @@ Cell 497
arf

# @@ Cell 498
arf=pd.DataFrame()
bins=[1950,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5]).astype('int64')

# @@ Cell 499
train[x]

# @@ Cell 500
xtest[x]

# @@ Cell 501
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 502
xtest[x]

# @@ Cell 503
train[x]

# @@ Cell 504
train[x].unique()

# @@ Cell 505
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
r.unique()

# @@ Cell 506
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
r[x].unique()

# @@ Cell 507
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
train[x]=r[x]
xtest[x]=ra[x]
#xtest[x]
r[x].unique()

# @@ Cell 508
#Bin Yearemodadd by [1949,1960,1970,1980,1990,2000,2010]


arf=pd.DataFrame()
bins=[1949,1960,1970,1980,1990,2000,2010]

arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5]).astype('int64')
train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5]).astype('int64')

# @@ Cell 509
train[x].unique()

# @@ Cell 510
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 511
x="GarageYrBlt"
train[x].unique()

# @@ Cell 512
x="GarageYrBlt"
train[x].unique()

data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 513
fig = sns.catplot(y='SalePrice', x, data=train)
#fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 514
fig = sns.catplot(y='SalePrice', x='GarageYrBuilt', data=train)
#fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 515
fig = sns.catplot(y='SalePrice', x='GarageYrBlt', data=train)
#fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 516
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp=train[x].replace(scale_mapper)

data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 517
temp

# @@ Cell 518
temp.unique()

# @@ Cell 519
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 520
temp

# @@ Cell 521
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 522
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(1750,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 523
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(1750,80000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 524
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',xlim=(1750,80000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 525
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1750,80000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 526
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1750,8000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 527
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2010));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 528
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2012));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 529
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2013));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 530
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2014));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 531
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2015));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 532
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2016));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 533
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2017));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 534
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1780,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 535
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

# @@ Cell 536
temp.unique()

# @@ Cell 537
train.head()

# @@ Cell 538
temp

# @@ Cell 539
temp['GarageYrBlt'].unique()

# @@ Cell 540
temp['GarageYrBlt'].value_counts()

# @@ Cell 541
temp['GarageYrBlt'].unique()

# @@ Cell 542
x="GarageYrBlt"
train[x].unique()

scale_mapper= {'NA': 0}
temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1850,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 543
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

# @@ Cell 544
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
arf

# @@ Cell 545
train[x]

# @@ Cell 546
train[x].unique()

# @@ Cell 547
train[x].replace(scale_mapper)

# @@ Cell 548
train[x].replace(scale_mapper).unique()

# @@ Cell 549
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
arf

# @@ Cell 550
train[x]=train[x].replace(scale_mapper)

# @@ Cell 551
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
arf

# @@ Cell 552
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 553
#train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

# @@ Cell 554
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
#arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
#train[x]=arf
xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')

# @@ Cell 555
xtest[x]

# @@ Cell 556
xtest[x].unique()

# @@ Cell 557
train[x].unique()

# @@ Cell 558
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')

# @@ Cell 559
arf

# @@ Cell 560
arf.hist()

# @@ Cell 561
arf[0]

# @@ Cell 562
xtest[x]

# @@ Cell 563
xtest[x].unique()

# @@ Cell 564
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
ra[x].unique()

# @@ Cell 565
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
ra[x].hist()

# @@ Cell 566
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
ra[x].unique()

# @@ Cell 567
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]) #.astype('int64')
#train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')

# @@ Cell 568
arf

# @@ Cell 569
arf[0]

# @@ Cell 570
arf

# @@ Cell 571
arf[0]

# @@ Cell 572
type(arf[0])

# @@ Cell 573
arf[0]

# @@ Cell 574
arf.unique()

# @@ Cell 575
arf.value_counts()

# @@ Cell 576
train[x].value_counts()

# @@ Cell 577
arf.bins()

# @@ Cell 578
arf

# @@ Cell 579
train[x]

# @@ Cell 580
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
ra[x]==2207

# @@ Cell 581
ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]
ra[ra[x]==2207][x]

# @@ Cell 582
xtest[xtest[x]==2207][x]

# @@ Cell 583
xtest[xtest[x]].loc(1132)

# @@ Cell 584
xtest[xtest[x]].loc[1132]

# @@ Cell 585
xtest[xtest[x]].iloc[1132]

# @@ Cell 586
xtest[xtest[x]].loc[1132]

# @@ Cell 587
xtest[xtest[x]].iloc[1132]

# @@ Cell 588
xtest.iloc[1132]

# @@ Cell 589
xtest.iloc[1132][x]

# @@ Cell 590
xtest.iloc[1132][x]=6

# @@ Cell 591
xtest.iloc[1132][x]

# @@ Cell 592
xtest.iloc[1132][x]==6

# @@ Cell 593
xtest.iloc[1132][x]=6

# @@ Cell 594
xtest.iloc[1132][x]

# @@ Cell 595
xtest.loc[train[x] == -9223372036854775808, x]

# @@ Cell 596
xtest.loc[xtest[x] == -9223372036854775808, x]

# @@ Cell 597
xtest.loc[xtest[x] == -9223372036854775808, x]=6

# @@ Cell 598
xtest.iloc[1132][x]

# @@ Cell 599
bins=[-1,1899,1920,1940,1960,1980,2000,2010]
arf=pd.DataFrame()
arf=pd.cut(train[x],bins,labels=[0,1,2,3,4,5,6]) #.astype('int64')
train[x]=arf
#xtest[x]=pd.cut(xtest[x],bins,labels=[0,1,2,3,4,5,6]).astype('int64')

# @@ Cell 600
train[x]

# @@ Cell 601
xtest[x]

# @@ Cell 602
xtest[x].value_counts()

# @@ Cell 603
xtest.info()

# @@ Cell 604
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].replace(scale_mapper)
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1900,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 605
train[x].unique()

# @@ Cell 606
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x]
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1900,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 607
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].astype('int64')
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(1900,2019));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 608
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].astype('int64')
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(0,8));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 609
train[x]=train[x].astype('int64')
train[x].unique

# @@ Cell 610
train[x]=train[x].astype('int64')
train[x].unique()

# @@ Cell 611
xtest[x]=xtest[x].astype('int64')
xtest[x].unique()

# @@ Cell 612
x="GarageYrBlt"
train[x].unique()


temp=pd.DataFrame()    
temp['GarageYrBlt']=train[x].astype('int64')
temp['SalePrice']=train['SalePrice']
data=temp[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 613
x="Utilities"
train[x].unique()


temp=pd.DataFrame()    
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 614
train[x].value_counts()

# @@ Cell 615
train[x].dtype

# @@ Cell 616
train[x].dtypes

# @@ Cell 617
train[x].dtypes

# @@ Cell 618
train.dtypes

# @@ Cell 619
train['Id'.dtypes

# @@ Cell 620
train['Id'].dtypes

# @@ Cell 621
train.astype('int64')

# @@ Cell 622
x="Utilities"
train[x].astype('int64')


temp=pd.DataFrame()    
data=train[['SalePrice', x]]
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 623
train.type()

# @@ Cell 624
train.types()

# @@ Cell 625
train.info()

# @@ Cell 626
x="Utilities"
train[x].astype('int64')


temp=pd.DataFrame()    
data=train[['SalePrice', x]].astype('int64')
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 627
x=['Utilities','ExterQual','KitchenQual','Functional']

train[x]=train[x].astype('int')

# @@ Cell 628
train.info()

# @@ Cell 629
x=['Utilities','ExterQual','KitchenQual','Functional']

train[x]=train[x].astype('int64')

# @@ Cell 630
train.info()

# @@ Cell 631
xtest.info()

# @@ Cell 632
x=['Utilities','ExterQual','KitchenQual','Functional']

xtest[x]=xtest[x].astype('int64')

# @@ Cell 633
xtest.info()

# @@ Cell 634
x=['Condition2','Electrical','Fence']

xtest[x]=xtest[x].astype('int64')


xtest.info()

# @@ Cell 635
x=['Condition2','Electrical','Fence','PoolQC']

xtest[x]=xtest[x].astype('int64')


xtest.info()

# @@ Cell 636
train.info()
xtest.info()

# @@ Cell 637
x="Utilities"
train[x].astype('int64')


temp=pd.DataFrame()    
data=train[['SalePrice', x]].astype('int64')
data.plot.scatter(x, y='SalePrice',ylim=(0,800000),xlim=(-1,7));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 638
#Try random forest and see results. One hot encode other values for regression


from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

# @@ Cell 639
X

# @@ Cell 640
y

# @@ Cell 641
X.shape

# @@ Cell 642
#Try random forest and see results. One hot encode other values for regression


from sklearn.datasets import make_regression
X= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

# @@ Cell 643
X.shape

# @@ Cell 644
X

# @@ Cell 645
#Try random forest and see results. One hot encode other values for regression


from sklearn.datasets import make_regression
X,= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

# @@ Cell 646
#Try random forest and see results. One hot encode other values for regression


from sklearn.datasets import make_regression
X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

# @@ Cell 647
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

# @@ Cell 648
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)

# @@ Cell 649
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 650
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 651
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=50, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 652
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 653
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, estimator=10)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 654
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimator=10)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 655
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=10)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# @@ Cell 656
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=10)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 657
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=100
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 658
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 659
y.hist()

# @@ Cell 660
pd.DataFrame(y).hist()

# @@ Cell 661
y

# @@ Cell 662
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=1000
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 663
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=40, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 664
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 665
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 666
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 667
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=100, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 668
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 669
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 670
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 671
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 672
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 673
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 674
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=5, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 675
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=10, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 676
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=1, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 677
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 678
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 679
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=5, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 680
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=8, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 681
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 682
y_test

# @@ Cell 683
y_test, y_pred

# @@ Cell 684
plt.plot(y_test, y_pred)

# @@ Cell 685
plt.scatterplot(y_test, y_pred)

# @@ Cell 686
plt.scatter(y_test, y_pred)

# @@ Cell 687
plt.scatter(y_test, y_pred)
(y_test-y_pred)

# @@ Cell 688
plt.scatter(y_test, y_pred)
(y_test-y_pred)^2

# @@ Cell 689
plt.scatter(y_test, y_pred)
(y_test-y_pred)*(y_test-y_pred)

# @@ Cell 690
import numpy as np

# @@ Cell 691
error=np.sqrt(mean_squared_error(y_test, y_pred))

# @@ Cell 692
np

# @@ Cell 693
error

# @@ Cell 694
y_test-y_pred

# @@ Cell 695
plt.hist(y_test-y_pred)

# @@ Cell 696
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=5, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 697
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=4, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 698
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 699
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 700
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=3, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 701
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=5, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 702
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 703
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 704
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=150, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 705
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 706
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 707
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 708
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 709
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features=None,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 710
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features=sqrt,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 711
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='sqrt',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 712
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='sqrt',min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 713
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='sqrt',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 714
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='sqrt',min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 715
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='sqrt',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 716
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='lg2',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 717
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features='log2',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 718
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 719
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=10000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 720
error=np.sqrt(mean_squared_error(y_test, y_pred))

# @@ Cell 721
error=np.sqrt(mean_squared_error(y_test, y_pred))

# @@ Cell 722
error

# @@ Cell 723
plt.hist(y_test-y_pred)

# @@ Cell 724
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=10000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 725
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=10000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 726
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X,y= make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=10000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 727
train

# @@ Cell 728
train[1:80]

# @@ Cell 729
train[2:80]

# @@ Cell 730
train[:,2:80]

# @@ Cell 731
train[2:80]

# @@ Cell 732
train[2:80,:]

# @@ Cell 733
train[1]

# @@ Cell 734
train[0]

# @@ Cell 735
train[1]

# @@ Cell 736
train[1:]

# @@ Cell 737
ytrain=train['SalePrice']
train.columns[1:80]

# @@ Cell 738
ytrain=train['SalePrice']
train.columns[1:]

# @@ Cell 739
ytrain=train['SalePrice']
train[train.columns[1:]]

# @@ Cell 740
ytrain=train['SalePrice']
train[train.columns[1:80]]

# @@ Cell 741
ytrain=train['SalePrice']
xtrain=train[train.columns[1:80]]

# @@ Cell 742
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 743
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 744
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=10000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 745
y_test,y_pred

# @@ Cell 746
y_test-y_pred

# @@ Cell 747
y_test.hist()

# @@ Cell 748
plt.plot(y_test,y_pred)

# @@ Cell 749
plt.scatter(y_test,y_pred)

# @@ Cell 750
np.log(ytrain)

# @@ Cell 751
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 752
plt.scatter(y_test,y_pred)

# @@ Cell 753
np.log(5)

# @@ Cell 754
2^1.6

# @@ Cell 755
2**1.6

# @@ Cell 756
10**1.6

# @@ Cell 757
e**1.6

# @@ Cell 758
exp(1.6)

# @@ Cell 759
np.exp(1.6)

# @@ Cell 760
np.exp(1.609)

# @@ Cell 761
plt.scatter(np.log(y_test),np.log(y_pred))

# @@ Cell 762
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 763
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred))))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 764
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 765
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 766
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_test), (y_pred))))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 767
plt.scatter((y_test),(y_pred))

# @@ Cell 768
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_test), (y_pred))))

rint('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_train), (y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 769
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_test), (y_pred))))

print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_train), (y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# @@ Cell 770
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 771
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 20,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 772
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 20,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 773
plt.scatter((y_train),(y_predtr))

# @@ Cell 774
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 775
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= auto,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 776
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 777
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 20,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 778
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 779
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=10, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 780
#Try random forest and see results. One hot encode other values for regression

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


#Model is overfit, must reduce noise, 

# @@ Cell 781
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 782
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=30, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 783
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=10, max_features= 10,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 784
#Try random forest and see results. One hot encode other values for regression

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


#Model is overfit, must reduce noise, 

# @@ Cell 785
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=20, max_features= 10,min_samples_split=3, min_samples_leaf=1, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 786
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=20, max_features= 10,min_samples_split=3, min_samples_leaf=3, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 787
#Try random forest and see results. One hot encode other values for regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X,y= make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)


X_train, X_test, y_train, y_test = train_test_split(
     xtrain, ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=20, max_features= 10,min_samples_split=1, min_samples_leaf=2, random_state=0, n_estimators=1000)
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


#Model is overfit, must reduce noise, 

# @@ Cell 788
#Try random forest and see results. One hot encode other values for regression

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


#Model is overfit, must reduce noise, 

# @@ Cell 789
regr.feature_importances_

# @@ Cell 790
regr.feature_importances_.hist()

# @@ Cell 791
regr.feature_importances_

# @@ Cell 792
train.feature_names

# @@ Cell 793
feat_importances = pd.Series(regr.feature_importances_, index=train.columns)
feat_importances.nlargest(4).plot(kind='barh')

# @@ Cell 794
train.columns

# @@ Cell 795
feat_importances = pd.Series(regr.feature_importances_, index=train.columns)
#feat_importances.nlargest(4).plot(kind='barh')

# @@ Cell 796
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(4).plot(kind='barh')

# @@ Cell 797
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(4).plot(kind='barh')

# @@ Cell 798
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(20).plot(kind='barh')

# @@ Cell 799
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(20).plot(kind='bar')

# @@ Cell 800
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(50).plot(kind='bar')

# @@ Cell 801
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(40).plot(kind='bar')

# @@ Cell 802
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(30).plot(kind='bar')

# @@ Cell 803
feat_importances.nlargest(30)

# @@ Cell 804
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 805
feat_importances.nlargest()

# @@ Cell 806
feat_importances.nlargest(50)

# @@ Cell 807
xtrain.columns.shape[0]

# @@ Cell 808
feat_importances.nlargest(30)

# @@ Cell 809
feat_importances.nlargest(50)

# @@ Cell 810
feat_importances.nlargest(50)[0]

# @@ Cell 811
feat_importances.nlargest(50)

# @@ Cell 812
feat_importances.nlargest(50).aslist()

# @@ Cell 813
feat_importances.nlargest(50).as_list()

# @@ Cell 814
feat_importances.nlargest(50)

# @@ Cell 815
feat_importances.nlargest(50).index

# @@ Cell 816
feat_importances.nlargest(50).index.aslist()

# @@ Cell 817
feat_importances.nlargest(50).index.as_list

# @@ Cell 818
feat_importances.nlargest(50).index.aslist

# @@ Cell 819
feat_importances.nlargest(50).index

# @@ Cell 820
feat_importances.nlargest(50).index()

# @@ Cell 821
feat_importances.nlargest(50).index

# @@ Cell 822
xtrain[feat_importances.nlargest(50).index]

# @@ Cell 823
topcols=feat_importances.nlargest(50).index

# @@ Cell 824
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain[topcols], test_size=0.33, random_state=42)

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

# @@ Cell 825
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

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

# @@ Cell 826
topcols=feat_importances.nlargest(30).index

# @@ Cell 827
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

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

# @@ Cell 828
topcols=feat_importances.nlargest(15).index

# @@ Cell 829
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

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

# @@ Cell 830
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 831
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 832
topcols=feat_importances.nlargest(10).index

# @@ Cell 833
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 834
topcols=feat_importances.nlargest(15).index

# @@ Cell 835
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 'auto',min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 836
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 3,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 837
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 838
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
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

# @@ Cell 839
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=100)
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

# @@ Cell 840
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
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

# @@ Cell 841
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(15).index

# @@ Cell 842
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(15).index

# @@ Cell 843
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')

# @@ Cell 844
regr.feature_importances_

# @@ Cell 845
#Try random forest and see results. One hot encode other values for regression

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

# @@ Cell 846
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(15).index

# @@ Cell 847
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
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

# @@ Cell 848
plt.scatter(y_train,y_pred)

# @@ Cell 849
plt.scatter(y_test,y_pred)

# @@ Cell 850
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], ytrain, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.log(y_test), (y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.log(y_train), (y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))

# @@ Cell 851
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
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

# @@ Cell 852
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 853
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(1+ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
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

# @@ Cell 854
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 855
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 856
np.exp(0.15)

# @@ Cell 857
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
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

# @@ Cell 858
np.exp(0.15)

# @@ Cell 859
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error((y_train), (y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))

# @@ Cell 860
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))

# @@ Cell 861
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 862
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 863
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

# @@ Cell 864
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

# @@ Cell 865
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

# @@ Cell 866
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=200)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))

# @@ Cell 867
X_train, X_test, y_train, y_test = train_test_split(
     xtrain[topcols], np.log(ytrain), test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
y_predtr=regr.predict(X_train)

print('For cross validation set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('For Training set')
print('Root mean squared error: %.2f'
      % np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_predtr))))
print('Coefficient of determination: %.2f'
      % r2_score(y_train, y_predtr))

# @@ Cell 868
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# @@ Cell 869
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# @@ Cell 870
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# @@ Cell 871
kf.get_n_splits(X)

# @@ Cell 872
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain[train_index], xtest[test_index]
    y_train, y_test = ytrain[train_index], ytest[test_index]

# @@ Cell 873
xtest

# @@ Cell 874
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

# @@ Cell 875
xtrain[train_index]

# @@ Cell 876
xtrain.iloc[train_index]

# @@ Cell 877
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]

# @@ Cell 878
kf = KFold(n_splits=5)
cvscores=[]
trscores[]
for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    

    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr))

# @@ Cell 879
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    

    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr))

# @@ Cell 880
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    

    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr)) 

# @@ Cell 881
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    

    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr)) 

# @@ Cell 882
xtrain

# @@ Cell 883
ytrain

# @@ Cell 884
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    

    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr)) 

# @@ Cell 885
cvscores

# @@ Cell 886
plt.scatter(y_test,y_pred)

# @@ Cell 887
y_pred

# @@ Cell 888
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')

# @@ Cell 889
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 890
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 891
x=[0,1,2,3,5,70000]
y=[0,1,2,3,5,70000]

# @@ Cell 892
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 893
x=[0,1,2,3,5,700000]
y=[0,1,2,3,5,700000]

# @@ Cell 894
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 895
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 896
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

# @@ Cell 897
kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    
"""
    print('For cross validation set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    print('For Training set')
    print('Root mean squared error: %.2f'
          % np.sqrt(mean_squared_error((y_train), (y_predtr))))
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_predtr)) 
          """

# @@ Cell 898
cvscores

# @@ Cell 899
kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
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

# @@ Cell 900
cvscores

# @@ Cell 901
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
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

# @@ Cell 902
cvscores

# @@ Cell 903
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 904
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 905
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 906
xtrain

# @@ Cell 907
log(cvscores)

# @@ Cell 908
np.log(cvscores)

# @@ Cell 909
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=np.log(ytrain)
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

# @@ Cell 910
cvscores

# @@ Cell 911
ytrain.hist()

# @@ Cell 912
np.log(ytrain).hist()

# @@ Cell 913
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 914
plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 915
plt.scatter(np.exp(y_train),np.exp(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 916
#x=[0,1,2,3,5,700000]
#y=[0,1,2,3,5,700000]

plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 917
#x=[0,1,2,3,5,700000]
#y=[0,1,2,3,5,700000]

plt.scatter(y_train,y_predtr)
plt.xlabel('y_test')
plt.xlabel('y_pred')
#plt.plot(x,y)

# @@ Cell 918
plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.xlabel('y_pred')

#plt.plot(x,y)

# @@ Cell 919
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

#plt.plot(x,y)

# @@ Cell 920
regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_train)

# @@ Cell 921
regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_train)

y_pred

# @@ Cell 922
regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_traind
Y_pred.to_csv

# @@ Cell 923
y_pred.to_csv

# @@ Cell 924
savetxt('data.csv', y_pred, delimiter=',')

# @@ Cell 925
y_pred

# @@ Cell 926
from numpy import savetxt
savetxt('data.csv', y_pred, delimiter=',')

# @@ Cell 927
po=pd.read_csv('./data.csv')

# @@ Cell 928
po

# @@ Cell 929
po[:1]

# @@ Cell 930
po.columns

# @@ Cell 931
po.columns[0]

# @@ Cell 932
po.columns[0:2]

# @@ Cell 933
po[po.columns[0:2]]

# @@ Cell 934
po[po.columns[0:2]].to_csv('data.csv')

# @@ Cell 935
po[po.columns[0:2]].to_csv('data.csv',index=False)

# @@ Cell 936
po[po.columns[0:2]].to_csv('data.csv',index=False)

# @@ Cell 937
train.to_csv('train_mod.csv',index=False)
xtest.to_csv('test_mod.csv',index=False)

# @@ Cell 938
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,70000]
x=y
#plt.plot(x,y)

# @@ Cell 939
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=np.array([0,70000])
x=y
#plt.plot(x,y)

# @@ Cell 940
y

# @@ Cell 941
x

# @@ Cell 942
y

# @@ Cell 943
x=np.array([0,70000])

# @@ Cell 944
x=np.array([0,70000])
x

# @@ Cell 945
x=np.array([0,70000])
x
y=x

# @@ Cell 946
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=np.array([0,70000])
y=x
#plt.plot(x,y)

# @@ Cell 947
x=np.array([0,70000])
x
y=x

# @@ Cell 948
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 949
plt.scatter(np.exp(y_train),np.exp(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 950
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=np.log(ytrain)
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

# @@ Cell 951
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 952
x=np.array([0,700000])
x
y=x

# @@ Cell 953
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 954
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

# @@ Cell 955
cvscores

# @@ Cell 956
plt.scatter(np.exp(y_train),np.exp(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 957
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')

# @@ Cell 958
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 959
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 960
plt.scatter((y_test),(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 961
regr = RandomForestRegressor(max_depth=None, max_features= 5,min_samples_split=2, min_samples_leaf=1, random_state=0, n_estimators=1000)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_traind

# @@ Cell 962
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# @@ Cell 963
grid_search

# @@ Cell 964
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 965
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 966
grid_search

# @@ Cell 967
grid_search.fit(X,y)

# @@ Cell 968
xtrain

# @@ Cell 969
yltrain

# @@ Cell 970
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 971
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2
                           
#grid_search.fit(xtrain, yltrain)

# @@ Cell 972
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 973
grid_search.best_scores_

# @@ Cell 974
grid_search.best_score_

# @@ Cell 975
grid_search.best_parameters_

# @@ Cell 976
grid_search.best_parameter_

# @@ Cell 977
grid_search.best_params_

# @@ Cell 978
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, scoring=‘neg_mean_squared_error’,param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 979
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, scoring='neg_mean_squared_error',param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 980
grid_search.best_params_
grid_search.best_score_

# @@ Cell 981
grid_search.best_params_
#grid_search.best_score_

# @@ Cell 982
grid_search.best_params_
grid_search.best_score_

# @@ Cell 983
grid_search.best_params_
#er_search.best_score_

# @@ Cell 984
grid_search.best_params_
grid_search.best_score_

# @@ Cell 985
grid_search.best_params_
#earch.best_score_

# @@ Cell 986
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=8, random_state=0, n_estimators=1000)
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

# @@ Cell 987
cv_scores

# @@ Cell 988
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=8, random_state=0, n_estimators=1000)
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

# @@ Cell 989
cv_scores

# @@ Cell 990
trscores

# @@ Cell 991
cvscores

# @@ Cell 992
cvscores.mean()

# @@ Cell 993
np.mean(cvscores)

# @@ Cell 994
r2_score(y_test, y_pred))

# @@ Cell 995
r2_score(y_test, y_pred)

# @@ Cell 996
r2_score(y_train, y_predtr)

# @@ Cell 997
xtrain

# @@ Cell 998
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 999
x=np.array([0,1])
x
y=x

# @@ Cell 1000
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1001
x=np.array([10,15])
x
y=x

# @@ Cell 1002
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1003
plt.scatter((y_test),(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 1004
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [1,3,8],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 1005
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 1006
grid_search.best_params_
#earch.best_score_

# @@ Cell 1007
grid_search.best_params_
grid_search.best_score_

# @@ Cell 1008
grid_search.best_params_
#grid_search.best_score_

# @@ Cell 1009
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
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

# @@ Cell 1010
r2_score(y_train, y_predtr)

# @@ Cell 1011
cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))

# @@ Cell 1012
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
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

# @@ Cell 1013
cvscores

# @@ Cell 1014
np.mean(cvscores)

# @@ Cell 1015
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1016
x=[10,14]
x=y
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1017
x=[10,14]
x=y
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1018
x=[10,12]
x=y
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1019
x=[10,12]
x=y
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
#plt.plot(x,y)

# @@ Cell 1020
x=[10,12]
x=y
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1021
x=[10,12]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1022
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1023
plt.scatter((y_test),(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 1024
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')

plt.plot(x,y)

# @@ Cell 1025
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
x=y
plt.plot(x,y)

# @@ Cell 1026
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1027
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index
xtrain=xtrain[topcols]
ytrain=ytrain[topcols]
xtest=xtest[topcols]

# @@ Cell 1028
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 1029
ytrain

# @@ Cell 1030
ytrain=train['SalePrice']

# @@ Cell 1031
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 1032
grid_search.best_params_
#grid_search.best_score_

# @@ Cell 1033
grid_search.best_params_
grid_search.best_score_

# @@ Cell 1034
grid_search.best_params_
#grid_search.best_score_

# @@ Cell 1035
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain, yltrain)

# @@ Cell 1036
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1037
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=300)
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

# @@ Cell 1038
np.mean(cvscores)

# @@ Cell 1039
np.mean(cvscores)
cvscores

# @@ Cell 1040
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1041
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1042
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=300)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_traind

# @@ Cell 1043
y_pred

# @@ Cell 1044
xtest

# @@ Cell 1045
xtest.index

# @@ Cell 1046
xtest

# @@ Cell 1047
test

# @@ Cell 1048
ytrain

# @@ Cell 1049
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1050
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1051
ans

# @@ Cell 1052
ans.to_csv('data.csv',index=False)

# @@ Cell 1053
np.mean(cvscores)
#cvscores

# @@ Cell 1054
np.mean(cvscores)
cvscores

# @@ Cell 1055
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=10000)
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

# @@ Cell 1056
np.mean(cvscores)
cvscores

# @@ Cell 1057
np.mean(cvscores)
#cvscores

# @@ Cell 1058
np.mean(cvscores)
cvscores

# @@ Cell 1059
xtrain

# @@ Cell 1060
x=['YearRemodAdd','YearBuilt']
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")
temptrain=xtrain
temptest=xtest
xtrain[x]=r[x]
xtest[x]=ra[x]

# @@ Cell 1061
xtrain

# @@ Cell 1062
xtrain[x]

# @@ Cell 1063
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=10000)
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

# @@ Cell 1064
np.mean(cvscores)
cvscores

# @@ Cell 1065
np.mean(cvscores)
#cvscores

# @@ Cell 1066
np.mean(cvscores)
cvscores

# @@ Cell 1067
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index
#xtrain=xtrain[topcols]
#xtest=xtest[topcols]

# @@ Cell 1068
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index
#xtrain=xtrain[topcols]
#xtest=xtest[topcols]
xtrain.shape

# @@ Cell 1069
train.shape

# @@ Cell 1070
xtrain=train[1:81]

# @@ Cell 1071
xtrain=train[1:81]
xtrain

# @@ Cell 1072
xtrain=train[1:81]
#xtrain

# @@ Cell 1073
xtrain=train[1:81]
#xtrain
xtrain

# @@ Cell 1074
xtrain=train[:,1:81]
#xtrain
xtrain

# @@ Cell 1075
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain.head()

# @@ Cell 1076
#xtrain
xtrain

# @@ Cell 1077
#xtrain
xtest

# @@ Cell 1078
temptrain.shape[]

# @@ Cell 1079
temptrain.shape

# @@ Cell 1080
xtrain.shape

# @@ Cell 1081
xtrain.describe()

# @@ Cell 1082
r[x]

# @@ Cell 1083
xtrain

# @@ Cell 1084
xtrain

# @@ Cell 1085
xtest

# @@ Cell 1086
xtrain

# @@ Cell 1087
ra=pd.read_csv("./testmod.csv")

# @@ Cell 1088
ra=pd.read_csv("./test_mod.csv")

# @@ Cell 1089
ra.shape[]

# @@ Cell 1090
ra.shape

# @@ Cell 1091
xtest=pd.read_csv("./test_mod.csv")

# @@ Cell 1092
xtest.shape

# @@ Cell 1093
x=['YearRemodAdd','YearBuilt']
r=pd.read_csv("./train.csv")
ra=pd.read_csv("./test.csv")

xtrain[x]=r[x]
xtest[x]=ra[x]

# @@ Cell 1094
xtrain[x]

# @@ Cell 1095
xtest.shape

# @@ Cell 1096
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=10000)
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

# @@ Cell 1097
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1098
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1099
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=100)
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

# @@ Cell 1100
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain[topcols]):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=100)
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

# @@ Cell 1101
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1102
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1103
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1104
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=100)
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

# @@ Cell 1105
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1106
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1107
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain[topcols], yltrain)

# @@ Cell 1108
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1109
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=200)
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

# @@ Cell 1110
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1111
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1112
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1113
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1114
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1115
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=300)
regr.fit(xtrain, ytrain)
y_pred=regr.predict(xtest)
#y_predtr=regr.predict(X_traind

# @@ Cell 1116
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])
#y_predtr=regr.predict(X_traind

# @@ Cell 1117
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1118
ans.to_csv('data.csv',index=False)

# @@ Cell 1119
train.info()

# @@ Cell 1120
cat=pd.read_csv("./categorical.csv")

# @@ Cell 1121
cat

# @@ Cell 1122
cat['encode']==hot

# @@ Cell 1123
cat['encode']=='hot'

# @@ Cell 1124
cat[cat['encode']=='hot']

# @@ Cell 1125
cat[cat['encode']=='hot']['categories']

# @@ Cell 1126
cat[cat['encode']=='hot']['categories'][0]

# @@ Cell 1127
cat[cat['encode']=='hot']['categories']

# @@ Cell 1128
cat[cat['encode']=='hot']['categories'].tolist()

# @@ Cell 1129
hotcols=cat[cat['encode']=='hot']['categories'].tolist()

# @@ Cell 1130
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
hotcols

# @@ Cell 1131
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
xtrain[hotcols]

# @@ Cell 1132
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
xtrain[hotcols].info

# @@ Cell 1133
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
xtrain[hotcols].info)

# @@ Cell 1134
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
xtrain[hotcols].info()

# @@ Cell 1135
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()

# @@ Cell 1136
nomcols

# @@ Cell 1137
dictlist
#for i in range(len(dictlist)):
   # [[key, value]] = dictlist[i].items()
  #  scale_mapper={k: v for v, k in enumerate(value)}
   # print(key)
  #  print(scale_mapper)

# @@ Cell 1138
dictlist[0]
#for i in range(len(dictlist)):
   # [[key, value]] = dictlist[i].items()
  #  scale_mapper={k: v for v, k in enumerate(value)}
   # print(key)
  #  print(scale_mapper)

# @@ Cell 1139
[[key, value]] = dictlist[0].items()
value
#scale_mapper={k: v for v, k in enumerate(value)}

# @@ Cell 1140
[[key, value]] = dictlist[0].items()
enumerate(value)
#scale_mapper={k: v for v, k in enumerate(value)}

# @@ Cell 1141
[[key, value]] = dictlist[0].items()
k in enumerate(value)
#scale_mapper={k: v for v, k in enumerate(value)}

# @@ Cell 1142
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value)}
#scale_mapper={k: v for v, k in enumerate(value)}

# @@ Cell 1143
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value)}
#scale_mapper={k: v for v, k in enumerate(value)}
scale

# @@ Cell 1144
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value)}
#scale_mapper={k: v for v, k in enumerate(value)}
value

# @@ Cell 1145
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value)}
#scale_mapper={k: v for v, k in enumerate(value)}
scale

# @@ Cell 1146
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value)}
#scale_mapper={k: v for v, k in enumerate(value)}
v

# @@ Cell 1147
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value[::-1])}
#scale_mapper={k: v for v, k in enumerate(value)}
enumerate(range(5)[::-1])

# @@ Cell 1148
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value[::-1])}
#scale_mapper={k: v for v, k in enumerate(value)}
enumerate(range(5)[::-1])
scale

# @@ Cell 1149
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value[::-1])}
#scale_mapper={k: v for v, k in enumerate(value)}
enumerate(range(5)[::-1])
scale,key

# @@ Cell 1150
[[key, value]] = dictlist[0].items()
scale={k:v for v, k in enumerate(value[::-1])}
#scale_mapper={k: v for v, k in enumerate(value)}
enumerate(range(5)[::-1])
scale,value

# @@ Cell 1151
train['MSZoning']

# @@ Cell 1152
scalemapper

# @@ Cell 1153
dictlist[0]
len(dictlist)
for i in range(1):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)

# @@ Cell 1154
dictlist[0]
len(dictlist)

for i in range(1):
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)
    #for i in xtrain[nomcols[i]]
        #replace with reverse order

# @@ Cell 1155
dictlist[0]
len(dictlist)

for i in range(1):
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)
    #for i in xtrain[nomcols[i]]
        #replace with reverse order
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)

# @@ Cell 1156
dictlist[0]
len(dictlist)

for i in range(1):
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    print(key)
    print(scale_mapper)
    #for i in xtrain[nomcols[i]]
        #replace with reverse order
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)

# @@ Cell 1157
train

# @@ Cell 1158
terp=train
terst=xtest

# @@ Cell 1159
dictlist[0]
len(dictlist)

for i in range(1):
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    print(key)
    print(scale_mapper)
    #revert back
    
    
    #for i in xtrain[nomcols[i]]
    #replace with reverse order
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)
    #train[key]
    
    #scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)

# @@ Cell 1160
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    print(key)
    print(scale_mapper)
    #revert back
    
    
    #for i in xtrain[nomcols[i]]
    #replace with reverse order
    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)
    #train[key]
    
    #scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)

# @@ Cell 1161
terp

# @@ Cell 1162
terp['GarageYrBuilt']

# @@ Cell 1163
terp['GarageYrBlt']

# @@ Cell 1164
terp['GarageYrBlt'].unique()

# @@ Cell 1165
terp['GarageYrBlt'].value_counts()

# @@ Cell 1166
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1167
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    #terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1168
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1169
terp[key]

# @@ Cell 1170
terp[key].unique()

# @@ Cell 1171
terp[key].value_counts()

# @@ Cell 1172
terp[key-1].value_counts()

# @@ Cell 1173
dictlist[0]
len(dictlist)

for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1174
terp[key].value_counts()
test[key].value_counts()

# @@ Cell 1175
terp[key].value_counts()
xtest[key].value_counts()

# @@ Cell 1176
terp[key].value_counts()
train[key].value_counts()

# @@ Cell 1177
terp[key].value_counts()
#train[key].value_counts()

# @@ Cell 1178
terp[key].value_counts()
train[key].value_counts()

# @@ Cell 1179
terp[key].value_counts()
#train[key].value_counts()

# @@ Cell 1180
terp[key].value_counts(),train[key].value_counts()

# @@ Cell 1181
dictlist[0]
len(dictlist)

for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1182
dictlist[0]
len(dictlist)

for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1183
terp[key].value_counts(),train[key].value_counts()

# @@ Cell 1184
dictlist[0]
len(dictlist)

for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1185
dictlist[0]
len(dictlist)

for i in range(2):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1186
terp[key].value_counts(),train[key].value_counts()

# @@ Cell 1187
terp[key].value_counts(),train[key].value_counts()
scale_mapper

# @@ Cell 1188
terp[key].value_counts(),train[key].value_counts()
scale_mapper
terp[key]

# @@ Cell 1189
terp[key].value_counts(),train[key].value_counts()
scale_mapper
terp[key].value_counts()

# @@ Cell 1190
terp[key].value_counts(),train[key].value_counts()
scale_mapper
terp[key].value_counts()

# @@ Cell 1191
terp[key].value_counts(),train[key].value_counts()

terp[key].value_counts()
scale_mapper

# @@ Cell 1192
dictlist[0]
len(dictlist)

for i in range(2):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1193
terp[key].value_counts(),train[key].value_counts()

terp[key].value_counts()
#scale_mapper

# @@ Cell 1194
terp[key].value_counts(),train[key].value_counts()


#scale_mapper

# @@ Cell 1195
terp[key].value_counts(),

train[key].value_counts()


#scale_mapper

# @@ Cell 1196
terp[key].value_counts(),

xtrain[key].value_counts()


#scale_mapper

# @@ Cell 1197
terp[key].value_counts(),

train[key].value_counts()


#scale_mapper

# @@ Cell 1198
terp[key].value_counts(),

xtrain[key].value_counts()


#scale_mapper

# @@ Cell 1199
terp=xtrain
terst=xtest

# @@ Cell 1200
dictlist[0]
len(dictlist)

for i in range(len):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1201
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1202
terp

# @@ Cell 1203
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terp[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1204
scalemapper

# @@ Cell 1205
scale_mapper

# @@ Cell 1206
scale_mapper

terst[key]

# @@ Cell 1207
scale_mapper

terst[key],scale_mapper

# @@ Cell 1208
scale_mapper

terst[key],scale_mapper,terp[key]

# @@ Cell 1209
scale_mapper

terp[key]

# @@ Cell 1210
scale_mapper

terst[key]

# @@ Cell 1211
scale_mapper

terst[key].replace(scale_mapper)

# @@ Cell 1212
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1213
terp=xtrain
terst=xtest

# @@ Cell 1214
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1215
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1216
terst[key]

# @@ Cell 1217
terst[key].value_counts xtest[key].value_counts

# @@ Cell 1218
terst[key].value_counts,xtest[key].value_counts

# @@ Cell 1219
terst[key].value_counts(),xtest[key].value_counts()

# @@ Cell 1220
terp=xtrain
terst=xtest

# @@ Cell 1221
terp=xtrain
terst=xtest
terst[key]

# @@ Cell 1222
terp=xtrain
terst=xtest
terst[key].value_counts()

# @@ Cell 1223
xtest[key].value_counts()

# @@ Cell 1224
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={v: k for v, k in enumerate(value)}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1225
#terp=xtrain
#terst=xtest
terst[key].value_counts()

# @@ Cell 1226
xtest[key].value_counts()

# @@ Cell 1227
xtest

# @@ Cell 1228
xtrain

# @@ Cell 1229
train

# @@ Cell 1230
xtrain

# @@ Cell 1231
terp['GarageYrBlt'].value_counts()

# @@ Cell 1232
#terp=xtrain
#terst=xtest
terst[key].value_counts()

# @@ Cell 1233
#terp=xtrain
#terst=xtest
terst[key].value_counts()
xtest

# @@ Cell 1234
#terp=xtrain
#terst=xtest
terst[key].value_counts()
xtest[key].value_counts()

# @@ Cell 1235
#terp=xtrain
#terst=xtest
terst[key].value_counts(),xtest[key].value_counts()

# @@ Cell 1236
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value::-1)}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1237
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate[value::-1]}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1238
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate([value::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1239
dictlist[0]
len(dictlist)

for i in range(len(dictlist)):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    terp[key]=terp[key].replace(scale_mapper)
    terst[key]=terst[key].replace(scale_mapper)

    print(key)
    print(scale_mapper)

# @@ Cell 1240
#terp=xtrain
#terst=xtest
terst[key].value_counts(),xtest[key].value_counts()

# @@ Cell 1241
#terp=xtrain
#terst=xtest
terst[key].value_counts(),xtest[key].value_counts()
train

# @@ Cell 1242
#terp=xtrain
#terst=xtest
terst[key].value_counts(),xtest[key].value_counts()
xtrain[key]

# @@ Cell 1243
xtrain

# @@ Cell 1244
for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for k, v in enumerate(value)}
    terp[key]=terp[key].replace(scale_mapper)

# @@ Cell 1245
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()

# @@ Cell 1246
terp==train

# @@ Cell 1247
xtrain

# @@ Cell 1248
terp

# @@ Cell 1249
todoo=terp

# @@ Cell 1250
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()

# @@ Cell 1251
for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for k, v in enumerate(value[::-1])}
    todoo[key]=todoo[key].replace(scale_mapper)

# @@ Cell 1252
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()

# @@ Cell 1253
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()
toodoo[key].value_counts()

# @@ Cell 1254
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()
todoo[key].value_counts()

# @@ Cell 1255
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts(),todoo[key].value_counts()

# @@ Cell 1256
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()
todoo[key].value_counts()

# @@ Cell 1257
todoo[key].value_counts()

# @@ Cell 1258
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()

# @@ Cell 1259
for i in range(1):

    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value)}
    todoo[key]=todoo[key].replace(scale_mapper)

# @@ Cell 1260
#terp=xtrain
#terst=xtest
terp[key].value_counts(),xtrain[key].value_counts()

# @@ Cell 1261
todoo[key].value_counts()

# @@ Cell 1262
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 1263
train.head()

# @@ Cell 1264
ytrain=train['SalePrice']
train.shape, test.shape

#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()

# @@ Cell 1265
f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(),vmin=.2,square=True,mask=mask);

# @@ Cell 1266
#top correlated variables

s=train.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
corr=pd.DataFrame(s)
corr=corr.rename(columns={0: 'abs(r)'})
corr[corr['abs(r)']>0.5]

# @@ Cell 1267
#top correlated variables with respect to salesprice
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 1268
#About the highest R value OverallQual
# this is an ordinal categorical variable represented by integers
np.sort(train.OverallQual.unique())

# @@ Cell 1269
#Boxplots to show trends of Saleprice vs Overall Qual
bplot_data=train[['SalePrice', 'OverallQual']]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=bplot_data)
plt.xlabel('OverallQual',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
plt.title('Boxplots of Overall Qual')

# @@ Cell 1270
#Continuous variable
train.GrLivArea

# @@ Cell 1271
#There are two outliers GrLvArea>4000, but generally there is a linear relationship
GrLiv_data=train[['SalePrice', 'GrLivArea']]
GrLiv_data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000));
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1272
#Possible candidates for outliers
train[(GrLiv_data['GrLivArea']>4000) & (GrLiv_data['SalePrice']<200000) ][['GrLivArea','OverallQual','SalePrice']]

# @@ Cell 1273
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)

# @@ Cell 1274
print(misval.shape[0]," total missing values" )
print((misval['misval_train']!=0).sum(), "missing values in training set")
print((misval['misval_test']!=0).sum(), "missing values in test set")
misval

# @@ Cell 1275
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 1276
train.head()

# @@ Cell 1277
xtest=test[test.columns[1:]]

# @@ Cell 1278
train.info()

# @@ Cell 1279
train.info()

# @@ Cell 1280
ytrain=train['SalePrice']
train.shape, test.shape

#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()

# @@ Cell 1281
f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(),vmin=.2,square=True,mask=mask);

# @@ Cell 1282
#top correlated variables

s=train.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
corr=pd.DataFrame(s)
corr=corr.rename(columns={0: 'abs(r)'})
corr[corr['abs(r)']>0.5]

# @@ Cell 1283
#top correlated variables with respect to salesprice
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 1284
#About the highest R value OverallQual
# this is an ordinal categorical variable represented by integers
np.sort(train.OverallQual.unique())

# @@ Cell 1285
#Boxplots to show trends of Saleprice vs Overall Qual
bplot_data=train[['SalePrice', 'OverallQual']]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=bplot_data)
plt.xlabel('OverallQual',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
plt.title('Boxplots of Overall Qual')

# @@ Cell 1286
#Continuous variable
train.GrLivArea

# @@ Cell 1287
#There are two outliers GrLvArea>4000, but generally there is a linear relationship
GrLiv_data=train[['SalePrice', 'GrLivArea']]
GrLiv_data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000));
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1288
#Possible candidates for outliers
train[(GrLiv_data['GrLivArea']>4000) & (GrLiv_data['SalePrice']<200000) ][['GrLivArea','OverallQual','SalePrice']]

# @@ Cell 1289
xtest=test[test.columns[1:]]
xtest

# @@ Cell 1290
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)

# @@ Cell 1291
print(misval.shape[0]," total missing values" )
print((misval['misval_train']!=0).sum(), "missing values in training set")
print((misval['misval_test']!=0).sum(), "missing values in test set")
misval

# @@ Cell 1292
get_ipython().system('grep -B8 NA data_description.txt')
#14 with NA in description 5 basement qual 4 garage qual

# @@ Cell 1293
# impute NA to category NA
cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

train['MiscFeature'].unique(),xtest['MiscFeature'].unique()

# @@ Cell 1294
#NA and 0's match there are no NAs that are missing values!
print(xtest[(xtest['Fireplaces']==0) & (xtest['FireplaceQu']!='NA')].shape[0],
      train[(train['Fireplaces']==0) & (train['FireplaceQu']!='NA')].shape[0])

# @@ Cell 1295
#Three missing values for Poolarea
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

xtest[(xtest['PoolQC']== 'NA') & (xtest['PoolArea']>0)][['OverallQual', 'PoolQC', 'PoolArea']]

# @@ Cell 1296
fig = sns.catplot(y='OverallQual', x="PoolQC",order=["NA", "Fa","TA", "Gd","Ex"], data=train)
fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 1297
#slight increase in quality vs pool qc-- give FA for all missing data
xtest['PoolQC']=xtest['PoolQC'].replace({'NA':'Fa'})
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

# @@ Cell 1298
## Training set: 81 entries with no garage 

cols=['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
print(train.filter(regex='Garage').isna().sum())
print('total number of entries with misvals is',
      pd.isnull(train[cols].any(axis=1)).sum())

# @@ Cell 1299
#Double check to make sure that all entries with no garage have zero values for 
#garagecars and garage area
train[pd.isnull(train['GarageType'])][['GarageCars','GarageArea']].sum()

# @@ Cell 1300
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

# @@ Cell 1301
cols= ['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea']
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].mode()[0]} 
    xtest[cols[i]].loc[[666,1116]]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

# @@ Cell 1302
#Cannot use 0 for area! use 440 as the next most common value.
print(train['GarageArea'].value_counts())
xtest.loc[1116,'GarageArea']=440
xtest.loc[[666,1116]].filter(regex='Garage')

# @@ Cell 1303
#Need to remove all NA's before running imputation

cols= ['GarageType','GarageFinish','GarageQual','GarageCond','GarageYrBlt']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 1304
cols=train.filter(regex='Bsmt').loc[:, train.filter(regex='Bsmt').isnull().any()].columns

# @@ Cell 1305
print(train.filter(regex='Bsmt').isna().sum())
print('total number of entries in training set with misvals is',
      pd.isnull(train.filter(regex='Bsmt')).any(axis=1).sum())

#Total of 39 entries, there is 37 common entries that have missing values across
#and there are two unique entries that have misvals in exposure and type2

# @@ Cell 1306
train[pd.isnull(train[cols]).any(axis=1)].filter(regex='Bsmt').loc[[332,948]]

# @@ Cell 1307
#Impute 332 to Rec, it is not unfinished because all 3 bsmtfinSF >0
print(train['BsmtFinType2'].value_counts())
#Impute 948 to No
print(train['BsmtExposure'].value_counts())

train.loc[332,'BsmtFinType2']='Rec'
train.loc[948,'BsmtExposure']='No'
train.loc[[332,948]].filter(regex='Bsmt')

# @@ Cell 1308
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
#temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]

# @@ Cell 1309
#And then impute the Missing values for the 7 other values
temp=xtest[pd.isnull(xtest[cols]).any(axis=1)].filter(regex='Bsmt')
cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
temp[~temp[cols].isnull().all(axis=1)]

# @@ Cell 1310
#Impute missing values for the 7 entries
cols= ['BsmtQual','BsmtCond','BsmtExposure']
ind=temp[~temp[cols].isnull().all(axis=1)].index
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].value_counts().keys()[0]} 
    xtest[cols[i]].loc[ind]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

#Check if it worked    
xtest[cols].loc[ind]

# @@ Cell 1311
#Fill in the NA values
cols=xtest.filter(regex='Bsmt').columns.tolist()

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')
#Check to see that there are no NA values
(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 1312
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 1313
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

# @@ Cell 1314
cols=['MasVnrArea','MasVnrType']

train[(train['MasVnrArea']<20) & (train['MasVnrArea']>0)][cols]
xtest[(xtest['MasVnrArea']<20) & (xtest['MasVnrArea']>0)][cols]


#Lets assume that the MasVnrArea of 1 is actually meant to be 0

# @@ Cell 1315
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

# @@ Cell 1316
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 1317
cols=misval.index.tolist()
for i in range(len(cols)):
    #print(xtest[cols[i]].fillna(xtest[cols[i]].value_counts()[0]))
    xtest[cols[i]]=xtest[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    train[cols[i]]=train[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    

# @@ Cell 1318
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

# @@ Cell 1319
import re
file=open('./categorical.txt','r')
stringlist=file.read().splitlines()

# @@ Cell 1320
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

# @@ Cell 1321
dictlist[13]

# @@ Cell 1322
dictlist[0]

# @@ Cell 1323
#dictlist.pop(0,14,13) remove all the numerical values
dictlist.pop(0)
len(dictlist)

# @@ Cell 1324
dictlist[14]

# @@ Cell 1325
#dictlist.pop(0,14,13) remove all the numerical values
dictlist.pop(14)
len(dictlist)

# @@ Cell 1326
dictlist[13]

# @@ Cell 1327
#dictlist.pop(0,14,13) remove all the numerical values
dictlist.pop(13)
len(dictlist)

# @@ Cell 1328
dictlist

# @@ Cell 1329
for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value::-1)}
    print(key)
    print(scale_mapper)
    train[key]=train[key].replace(scale_mapper)
    xtest[key]=xtest[key].replace(scale_mapper)
#train[key]

# @@ Cell 1330
for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)
    train[key]=train[key].replace(scale_mapper)
    xtest[key]=xtest[key].replace(scale_mapper)
#train[key]

# @@ Cell 1331
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
x=error_cols[9]
train[x].unique()

# @@ Cell 1332
error_cols

# @@ Cell 1333
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
error_cols
#train[x].unique()

# @@ Cell 1334
train[error_cols[0]]

# @@ Cell 1335
train[error_cols[0]].unique()

# @@ Cell 1336
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
scale_mapper= {'C (all)': 6}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1337
train[error_cols[0]].unique()

# @@ Cell 1338
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
scale_mapper= {'C (all)': 6}
x=error_cols[0]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1339
train[error_cols[1]].unique()

# @@ Cell 1340
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
scale_mapper= {'NAmes': 8}
x=error_cols[1]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1341
train[error_cols[2]].unique()

# @@ Cell 1342
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
scale_mapper= {'Twnhs': 0,'Duplex':2,'2fmCon':3}
x=error_cols[1]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1343
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
scale_mapper= {'Twnhs': 0,'Duplex':2,'2fmCon':3}
x=error_cols[2]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1344
train[error_cols[2]].unique()

# @@ Cell 1345
train[error_cols[1]].unique()

# @@ Cell 1346
train[error_cols[3]].unique()

# @@ Cell 1347
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
scale_mapper= {'Twnhs': 0,'Duplex':2,'2fmCon':3}
x=error_cols[3]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1348
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
scale_mapper= {'Wd Sdng':1}
x=error_cols[3]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1349
train[error_cols[4]].unique()

# @@ Cell 1350
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
scale_mapper= {'Wd Sdng':1, 'Wd Shng':0, 'CmentBd':11, 'Brk Cmn':14}
x=error_cols[3]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1351
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
scale_mapper= {'Wd Sdng':1, 'Wd Shng':0, 'CmentBd':11, 'Brk Cmn':14}
x=error_cols[4]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1352
train[error_cols[4]].unique()

# @@ Cell 1353
train[error_cols[5]].unique()

# @@ Cell 1354
train[error_cols[6]].unique()

# @@ Cell 1355
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
scale_mapper= {'NA':0}
x=error_cols[6]    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1356
train[error_cols[6]].unique()

# @@ Cell 1357
train[error_cols[7]].unique()

# @@ Cell 1358
train.info()

# @@ Cell 1359
error_cols=train.select_dtypes(exclude=['int64','float']).columns.tolist()
error_cols
#train[x].unique()

# @@ Cell 1360
train[error_cols[0]]

# @@ Cell 1361
train[error_cols[0]].value_counts()

# @@ Cell 1362
train[error_cols[1]].value_counts()

# @@ Cell 1363
train[error_cols[0]].value_counts()

# @@ Cell 1364
#Test to make sure that that the encoding worked
temp=pd.read_csv("./train.csv")
x='Functional'
temp[x].value_counts(),train[x].value_counts()

# @@ Cell 1365
xtest.info()

# @@ Cell 1366
error_cols=xtest.select_dtypes(exclude=['int64','float']).columns.tolist()
error_cols
#train[x].unique()

# @@ Cell 1367
xtest[error_cols[0]].value_counts()

# @@ Cell 1368
xtest[error_cols[1]].value_counts()

# @@ Cell 1369
xtest[error_cols[2]].value_counts()

# @@ Cell 1370
xtest[error_cols[3]].value_counts()

# @@ Cell 1371
xtest[error_cols[4]].value_counts()

# @@ Cell 1372
xtest['Condition2'].unique()

# @@ Cell 1373
fig = sns.boxplot(y='SalePrice', x="MSSubClass", data=train)
#MSsubclass is not ordinal

# @@ Cell 1374
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain.head()

# @@ Cell 1375
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

# @@ Cell 1376
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

# @@ Cell 1377
#Replace yr sold with num values 2006: 0,2007:1,2008:2,2009:3, 2010:4
# will need to one hot encode for regression
x='YrSold'


#r=pd.read_csv("./train.csv")
#ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]

scale_mapper= {2006: 0,2007:1,2008:2,2009:3, 2010:4}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1378
temp['GarageYrBlt'].unique()

# @@ Cell 1379
test['GarageYrBlt'].unique()

# @@ Cell 1380
train['GarageYrBlt'].unique()

# @@ Cell 1381
train.info()
xtest.info()

# @@ Cell 1382
#Change rest of the objects into int64, Now everything is numerically encoded

x=['Heating','BsmtCond','GarageQual','MiscFeature','RoofMatl']

train[x]=train[x].astype('int64')
xtest[x]=xtest[x].astype('int64')

# @@ Cell 1383
train.info()
xtest.info()

# @@ Cell 1384
xtrain=train[cols[1:81]]

# @@ Cell 1385
xtrain

# @@ Cell 1386
train

# @@ Cell 1387
xtrain=train[cols[1:]]

# @@ Cell 1388
xtrain

# @@ Cell 1389
xtrain=train[cols[[1:]]]

# @@ Cell 1390
xtrain=train[cols[[1:]]

# @@ Cell 1391
xtrain=train[cols[[1:]]]

# @@ Cell 1392
xtrain=train[columns[1:]]

# @@ Cell 1393
xtrain=train[traincolumns[1:]]

# @@ Cell 1394
xtrain=train[train.columns[1:]]

# @@ Cell 1395
xtrain

# @@ Cell 1396
xtrain=train[train.columns[1:81]]

# @@ Cell 1397
xtrain

# @@ Cell 1398
xtrain=train[train.columns[1:80]]

# @@ Cell 1399
xtrain

# @@ Cell 1400
train

# @@ Cell 1401
train
xtrain

# @@ Cell 1402
xtrain=train[train.columns[1:80]]
xtrain

# @@ Cell 1403
#train_mod and xtest_mod 
train.to_csv('train_v1.csv',index=False)
xtest.to_csv('test_v2.csv',index=False)

# @@ Cell 1404
xtrain=train[train.columns[1:80]]
xtrain

# @@ Cell 1405
train

# @@ Cell 1406
train['GarageYrBlt']

# @@ Cell 1407
train['GarageYrBlt'].unique()

# @@ Cell 1408
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=200)
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

# @@ Cell 1409
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1410
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
#cvscores

# @@ Cell 1411
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1412
train.info()

# @@ Cell 1413
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()

# @@ Cell 1414
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols]

# @@ Cell 1415
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
nomcols

# @@ Cell 1416
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
nomcols[0]

# @@ Cell 1417
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols[0]]

# @@ Cell 1418
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols[1]]

# @@ Cell 1419
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols[2]]

# @@ Cell 1420
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols

# @@ Cell 1421
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols]

# @@ Cell 1422
train.columns

# @@ Cell 1423
train.columns
nomcols

# @@ Cell 1424
train.columns,nomcols

# @@ Cell 1425
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[hotcols]

# @@ Cell 1426
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols]

# @@ Cell 1427
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
train[nomcols]

# @@ Cell 1428
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
cols = [col for col in df.columns if col not in hotcols]

# @@ Cell 1429
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
cols = [col for col in train.columns if col not in hotcols]

# @@ Cell 1430
cat=pd.read_csv("./categorical.csv")
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
cols = [col for col in train.columns if col not in hotcols]
cols

# @@ Cell 1431
cat

# @@ Cell 1432
cat['categories'].unique().tolist()

# @@ Cell 1433
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in train.columns if col not in catcols]
intcols

# @@ Cell 1434
train[intcols]

# @@ Cell 1435
xtrain[intcols]

# @@ Cell 1436
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]

# @@ Cell 1437
xtrain[intcols]

# @@ Cell 1438
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)

# @@ Cell 1439
data

# @@ Cell 1440
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)

# @@ Cell 1441
data

# @@ Cell 1442
pd.DataFrame(data)

# @@ Cell 1443
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols[0]]
scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)

# @@ Cell 1444
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)

# @@ Cell 1445
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)

# @@ Cell 1446
data

# @@ Cell 1447
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)

# @@ Cell 1448
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
data[[intcols]]=scaler.transform(data)

# @@ Cell 1449
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
scaler.fit(data)
data[[intcols]]=scaler.fit_transform(data.to_numpy())

# @@ Cell 1450
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()

data[[intcols]]=scaler.fit_transform(data.to_numpy())

# @@ Cell 1451
data

# @@ Cell 1452
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()

data=scaler.fit_transform(data.to_numpy())

# @@ Cell 1453
data

# @@ Cell 1454
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()

data=scaler.fit_transform(data)

# @@ Cell 1455
data

# @@ Cell 1456
pd.DataFrame(data, columns=intcols)

# @@ Cell 1457
pd.DataFrame(data, columns=intcols)['LotFrontage'].hist()

# @@ Cell 1458
xtest['LotFrontage'].hist()

# @@ Cell 1459
pd.DataFrame(data, columns=intcols)

# @@ Cell 1460
xtrain[intcols]=pd.DataFrame(data, columns=intcols)

# @@ Cell 1461
xtrain

# @@ Cell 1462
xtrain[intcols]

# @@ Cell 1463
xtrain[nomcols]

# @@ Cell 1464
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]

# @@ Cell 1465
from sklearn.preprocessing import StandardScaler
data=xtrain[intcols]
scaler = StandardScaler()
data=scaler.fit_transform(data)

# @@ Cell 1466
xtrain[intcols]=pd.DataFrame(data, columns=intcols)

# @@ Cell 1467
xtrain[nomcols]

# @@ Cell 1468
xtrain[intcols]

# @@ Cell 1469
xtrain[hot]

# @@ Cell 1470
xtrain['hot']

# @@ Cell 1471
xtrain[hotcols]

# @@ Cell 1472
temptest=scaler.transform(temptest)

# @@ Cell 1473
temptest=xtest[intcols]
temptest=scaler.transform(temptest)

# @@ Cell 1474
xtest[intcols]=pd.DataFrame(temptest, columns=intcols)

# @@ Cell 1475
xtrain[intcols]

# @@ Cell 1476
xtrain.to_csv('xtrain_sscale.csv',index=False)

# @@ Cell 1477
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtrain_sscale.csv',index=False)

# @@ Cell 1478
xtrain[intcols]

# @@ Cell 1479
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtrain_sscale.csv',index=False)
xtrain[intcols]

# @@ Cell 1480
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=2, random_state=0, n_estimators=200)
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

# @@ Cell 1481
#feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
#feat_importances.nlargest(15).plot(kind='bar')
#topcols=feat_importances.nlargest(30).index

np.mean(cvscores)
cvscores

# @@ Cell 1482
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(15).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

#np.mean(cvscores)
#cvscores

# @@ Cell 1483
xtrain['KitchenQual']

# @@ Cell 1484
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(30).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

#np.mean(cvscores)
#cvscores

# @@ Cell 1485
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(30).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

#np.mean(cvscores)
#cvscores

# @@ Cell 1486
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain[topcols], yltrain)

# @@ Cell 1487
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1488
grid_search.best_estimator_

# @@ Cell 1489
grid_search.best_params_,grid_search.best_score_
y_pred=grid_search.predict(xtest[topcols])

# @@ Cell 1490
grid_search.best_params_,grid_search.best_score_
#y_pred=grid_search.predict(xtest[topcols])
#y_pred=grid_search.predict(xtest[topcols])

# @@ Cell 1491
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1492
cvscores

# @@ Cell 1493
np.mean(cvscores)

# @@ Cell 1494
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1495
np.mean(cvscores)

# @@ Cell 1496
np.mean(cvscores)
cvscores

# @@ Cell 1497
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1498
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1499
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])
#y_predtr=regr.predict(X_traind

# @@ Cell 1500
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1501
ans.to_csv('data.csv',index=False)

# @@ Cell 1502
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]

# @@ Cell 1503
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]

# @@ Cell 1504
#Linear Regression---> Onehot non nominal variables, 
xtrain[nomcols]
#standardscale int values

# @@ Cell 1505
#Linear Regression---> Onehot non nominal variables, 
xtrain[hotcols]

# @@ Cell 1506
#Linear Regression---> Onehot non nominal variables, 
xtrain[hotcols]
pd.get_dummies(xtrain[hotcols])

# @@ Cell 1507
#Linear Regression---> Onehot non nominal variables, 
#xtrain[hotcols]
pd.get_dummies(xtrain[hotcols])

# @@ Cell 1508
#Linear Regression---> Onehot non nominal variables, 
#xtrain[hotcols]
df=pd.get_dummies(xtrain[hotcols])

# @@ Cell 1509
df

# @@ Cell 1510
df
xtrain[hotcols]

# @@ Cell 1511
df
#xtrain[hotcols]

# @@ Cell 1512
df
xtrain[hotcols]

# @@ Cell 1513
#Linear Regression---> Onehot non nominal variables, 
#xtrain[hotcols]
df=pd.get_dummies(xtrain[hotcols],drop_first=True)

# @@ Cell 1514
df
xtrain[hotcols]

# @@ Cell 1515
df
#xtrain[hotcols]

# @@ Cell 1516
hotcols

# @@ Cell 1517
hotcols.shape[]

# @@ Cell 1518
hotcols.shape

# @@ Cell 1519
len(hotcols)

# @@ Cell 1520
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
              'C': [1, 2, 3]})

# @@ Cell 1521
df

# @@ Cell 1522
pd.get_dummies(df,drop_first=True)

# @@ Cell 1523
df

# @@ Cell 1524
pd.get_dummies(df)

# @@ Cell 1525
df=xtrain[hotcols].astype(str)

# @@ Cell 1526
df.info()

# @@ Cell 1527
df=xtrain[hotcols].astype(str)
pd.get_dummies(xtrain[hotcols])

# @@ Cell 1528
pd.get_dummies(df[0])

# @@ Cell 1529
pd.get_dummies(df.columns[0])

# @@ Cell 1530
df=xtrain[hotcols].astype(str)
pd.get_dummies(df)

# @@ Cell 1531
df=xtrain[hotcols].astype(str)
dt=xtest[hotcols].astype(str)
hottrain=pd.get_dummies(df)
hottest=pd.get_dummies(dt)

# @@ Cell 1532
hottest

# @@ Cell 1533
df

# @@ Cell 1534
temptest

# @@ Cell 1535
temptest=xtest

# @@ Cell 1536
temptest

# @@ Cell 1537
temptest.drop([hotcols], axis=1, inplace=True)

# @@ Cell 1538
hotcols

# @@ Cell 1539
temptest.drop(hotcols, axis=1, inplace=True)

# @@ Cell 1540
temptest

# @@ Cell 1541
pd.concat([temptest, hottest])

# @@ Cell 1542
temptest

# @@ Cell 1543
temptest=xtest

# @@ Cell 1544
temptest

# @@ Cell 1545
temptest.drop(hotcols, axis=1, inplace=True)
#pd.concat([temptest, hottest])

# @@ Cell 1546
temptest=xtest

# @@ Cell 1547
temptest.drop(hotcols, axis=1, inplace=True)
#pd.concat([temptest, hottest])

# @@ Cell 1548
temptest

# @@ Cell 1549
xtest

# @@ Cell 1550
#temptest.drop(hotcols, axis=1, inplace=True)
temptest=pd.concat([temptest, hottest])

# @@ Cell 1551
temptest

# @@ Cell 1552
temptest=xtest

# @@ Cell 1553
temptest=xtest
xtest

# @@ Cell 1554
#temptest.drop(hotcols, axis=1, inplace=True)
temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1555
temptest

# @@ Cell 1556
temptest[hotcol]

# @@ Cell 1557
temptest[hotcols]

# @@ Cell 1558
hottrain

# @@ Cell 1559
hottest

# @@ Cell 1560
hottrain

# @@ Cell 1561
df

# @@ Cell 1562
dt

# @@ Cell 1563
df

# @@ Cell 1564
hottrain=pd.get_dummies(df)

# @@ Cell 1565
hottrain=pd.get_dummies(df)
hottrain

# @@ Cell 1566
hottrain=pd.get_dummies(df)
hottest

# @@ Cell 1567
hottrain=pd.get_dummies(dt)
hottest

# @@ Cell 1568
xtest

# @@ Cell 1569
xtrain

# @@ Cell 1570
#temptest.drop(hotcols, axis=1, inplace=True)
xtest

# @@ Cell 1571
get_ipython().system('ls')

# @@ Cell 1572
pd.read_csv('xtest_scale.csv')

# @@ Cell 1573
pd.read_csv('xtest_sscale.csv')

# @@ Cell 1574
xtest=pd.read_csv('xtest_sscale.csv')

# @@ Cell 1575
#temptest.drop(hotcols, axis=1, inplace=True)
xtest

# @@ Cell 1576
#df=xtrain[hotcols].astype(str)

#hottrain=pd.get_dummies(df)

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=xtest
temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1577
xtest

# @@ Cell 1578
temptest

# @@ Cell 1579
temptest
xtest

# @@ Cell 1580
temptest

# @@ Cell 1581
temptest
hotcols

# @@ Cell 1582
temptest
nomcols

# @@ Cell 1583
temptest
hotcols

# @@ Cell 1584
temptest

# @@ Cell 1585
#df=xtrain[hotcols].astype(str)

#hottrain=pd.get_dummies(df)

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=xtest
temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1586
temptest

# @@ Cell 1587
#df=xtrain[hotcols].astype(str)

#hottrain=pd.get_dummies(df)

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=xtest
temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1588
temptest

# @@ Cell 1589
xtrain

# @@ Cell 1590
temptest.drop(hotcols, axis=1, inplace=True)

# @@ Cell 1591
temptest.drop(hotcols, axis=1, inplace=True)

# @@ Cell 1592
xtest

# @@ Cell 1593
temptest

# @@ Cell 1594
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
temptrain=xtrain
temptrain=pd.concat([xtrain])

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=xtest
temptest=pd.DataFrame()
#temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1595
temptest

# @@ Cell 1596
temptest

# @@ Cell 1597
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
temptrain=xtrain
temptrain=pd.concat([xtrain])

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=pd.DataFrame()
temptest=xtest

#temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1598
temptest

# @@ Cell 1599
temptest.drop(hotcols, axis=1, inplace=True)
temptest

# @@ Cell 1600
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
temptrain=xtrain
temptrain=pd.concat([xtrain])

hottest=pd.get_dummies(xtest[hotcols].astype(str))
temptest=pd.DataFrame()
temptest=xtest

#temptest=pd.concat([temptest, hottest],axis=1)

# @@ Cell 1601
xtest

# @@ Cell 1602
temptest.drop(hotcols, axis=1, inplace=True)
temptest

# @@ Cell 1603
temptest

# @@ Cell 1604
temptest
xtrain

# @@ Cell 1605
temptest

# @@ Cell 1606
hottest

# @@ Cell 1607
newtest=pd.DataFrame()
newtest=pd.concat([xtest, hottest],axis=1)

# @@ Cell 1608
newtest

# @@ Cell 1609
newtest([hotcols])

# @@ Cell 1610
newtest[hotcols]

# @@ Cell 1611
newtest

# @@ Cell 1612
xtest=newtest

# @@ Cell 1613
#
xtest

# @@ Cell 1614
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain])

# @@ Cell 1615
xtrain

# @@ Cell 1616
#hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain,hottest],axis=1)

# @@ Cell 1617
xtrain

# @@ Cell 1618
hottrain

# @@ Cell 1619
#hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain,hottrain],axis=1)

# @@ Cell 1620
xtrain

# @@ Cell 1621
#
xtest

# @@ Cell 1622
get_ipython().system('ls')

# @@ Cell 1623
get_ipython().system('ls')

pd.read_csv('xtest_sscale')

# @@ Cell 1624
get_ipython().system('ls')

pd.read_csv('xtest_sscale.csv')

# @@ Cell 1625
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)

# @@ Cell 1626
xtrain

# @@ Cell 1627
xtrain=pd.read_csv('xtest_sscale.csv')

# @@ Cell 1628
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)

# @@ Cell 1629
hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)

# @@ Cell 1630
#hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain,hottrain],axis=1)

# @@ Cell 1631
xtrain

# @@ Cell 1632
xtrain

# @@ Cell 1633
xtrain
xtest

# @@ Cell 1634
xtrain
xtest.info()

# @@ Cell 1635
#xtrain
xtest.info()

# @@ Cell 1636
#xtrain
xtest

# @@ Cell 1637
#xtrain
xtest.info()

# @@ Cell 1638
#xtrain
xtrain.info()

# @@ Cell 1639
xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1640
xtrain

# @@ Cell 1641
get_ipython().system('ls')

# @@ Cell 1642
pd.read_csv('xtrain_sscale.csv')

# @@ Cell 1643
ar=pd.read_csv('xtrain_sscale.csv')
ar.columns

# @@ Cell 1644
ar=pd.read_csv('xtrain_sscale.csv')
ar['YrSold']

# @@ Cell 1645
ar=pd.read_csv('xtrain_sscale.csv')
ar['MoSold']

# @@ Cell 1646
xtrain['MoSold']

# @@ Cell 1647
xtrain

# @@ Cell 1648
ar=pd.read_csv('xtrain_sscale.csv')
ar['YrBuilt']

# @@ Cell 1649
ar=pd.read_csv('xtrain_sscale.csv')
ar['YearBuilt']

# @@ Cell 1650
ar=pd.read_csv('xtrain_sscale.csv')
ar['YearRemodAdd']

# @@ Cell 1651
xtest

# @@ Cell 1652
xtrain=pd.read_csv('xtrain_sscale.csv')
xtrain['YearRemodAdd']

# @@ Cell 1653
#One hot encoded AND num coded train and test sets

#hottest=pd.get_dummies(xtest[hotcols].astype(str))
#xtest.drop(hotcols, axis=1, inplace=True)
#xtest=pd.concat([xtest, hottest],axis=1)

hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1654
hottrain

# @@ Cell 1655
#One hot encoded AND num coded train and test sets

#hottest=pd.get_dummies(xtest[hotcols].astype(str))
#xtest.drop(hotcols, axis=1, inplace=True)
#xtest=pd.concat([xtest, hottest],axis=1)

hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1656
xtrain

# @@ Cell 1657
xtrain.columns()

# @@ Cell 1658
xtrain.columns

# @@ Cell 1659
xtrain

# @@ Cell 1660
xtrain=pd.read_csv('xtest_sscale.csv')
xtrain['YearRemodAdd']

# @@ Cell 1661
xtest=pd.read_csv('xtest_sscale.csv')
xtrain['YearRemodAdd']

# @@ Cell 1662
xtest=pd.read_csv('xtest_sscale.csv')
xtest['YearRemodAdd']

# @@ Cell 1663
xtest=pd.read_csv('xtest_sscale.csv')
#xtest['YearRemodAdd']

# @@ Cell 1664
xtest

# @@ Cell 1665
xtest['YearRemodAdd']

# @@ Cell 1666
xtest

# @@ Cell 1667
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]
xtest[intcols]

# @@ Cell 1668
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]
#xtest[intcols]

# @@ Cell 1669
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]
xtest[intcols]

# @@ Cell 1670
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]
#xtest[intcols]

# @@ Cell 1671
xtrain

# @@ Cell 1672
xtrain=pd.read_csv('xtrain_sscale.csv')
#xtest['YearRemodAdd']

# @@ Cell 1673
xtrain

# @@ Cell 1674
xtrain[intcols]

# @@ Cell 1675
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]
#xtest[intcols]

# @@ Cell 1676
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#temptrain=scaler.fit_transform(xtrain[intcols])
temptest=scaler.transform(xtest[intcols])
xtest[intcols]=pd.DataFrame(temptest, columns=intcols)
#xtrain[intcols]=pd.DataFrame(temptrain, columns=intcols)

# @@ Cell 1677
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest[intcols]

# @@ Cell 1678
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#temptrain=scaler.fit_transform(xtrain[intcols])
temptest=scaler.transform(xtest[intcols])
xtest[intcols]=pd.DataFrame(temptest, columns=intcols)
#xtrain[intcols]=pd.DataFrame(temptrain, columns=intcols)

# @@ Cell 1679
#xtrain.to_csv('xtrain_sscale.csv',index=False)
#xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest[intcols]

# @@ Cell 1680
get_ipython().system('ls')

# @@ Cell 1681
get_ipython().system('ls')
pd.read_csv('test_v2.csv')

# @@ Cell 1682
get_ipython().system('ls')
xtest=pd.read_csv('test_v2.csv')

# @@ Cell 1683
get_ipython().system('ls')
xtest=pd.read_csv('test_v2.csv')
xtrain=pd.read_csv('train_v1.csv')

# @@ Cell 1684
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
temptrain=scaler.fit(xtrain[intcols])
#temptest=scaler.transform(xtest[intcols])
#xtest[intcols]=pd.DataFrame(temptest, columns=intcols)
#xtrain[intcols]=pd.DataFrame(temptrain, columns=intcols)

# @@ Cell 1685
temptrain

# @@ Cell 1686
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
scl=scaler.fit(xtrain[intcols])
ttrain=scaler.transform(xtrain[intcols])
#temptest=scaler.transform(xtest[intcols])
#xtest[intcols]=pd.DataFrame(temptest, columns=intcols)
#xtrain[intcols]=pd.DataFrame(temptrain, columns=intcols)

# @@ Cell 1687
ttrain

# @@ Cell 1688
pd.DataFrame(ttrain,columns=intcols)

# @@ Cell 1689
pd.DataFrame(ttrain,columns=intcols)
xtrain

# @@ Cell 1690
pd.DataFrame(ttrain,columns=intcols)

# @@ Cell 1691
pd.DataFrame(ttrain,columns=intcols)
xtrain

# @@ Cell 1692
pd.DataFrame(ttrain,columns=intcols)
ttrain

# @@ Cell 1693
pd.DataFrame(ttrain,columns=intcols)

# @@ Cell 1694
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
scl=scaler.fit(xtrain[intcols])
ttrain=scaler.transform(xtrain[intcols])
ttest=scaler.transform(xtest[intcols])
#xtest[intcols]=pd.DataFrame(temptest, columns=intcols)
#xtrain[intcols]=pd.DataFrame(temptrain, columns=intcols)

# @@ Cell 1695
pd.DataFrame(ttest,columns=intcols)

# @@ Cell 1696
xtrain

# @@ Cell 1697
xtrain[xtrain.columns[1:80]]

# @@ Cell 1698
xtrain=xtrain[xtrain.columns[1:80]]

# @@ Cell 1699
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
scl=scaler.fit(xtrain[intcols])
ttrain=scaler.transform(xtrain[intcols])
ttest=scaler.transform(xtest[intcols])
xtest[intcols]=pd.DataFrame(ttest, columns=intcols)
xtrain[intcols]=pd.DataFrame(ttrain, columns=intcols)

# @@ Cell 1700
xtest

# @@ Cell 1701
xtest[intcols]

# @@ Cell 1702
xtrain

# @@ Cell 1703
xtest

# @@ Cell 1704
#xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest[intcols]

# @@ Cell 1705
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1706
np.mean(cvscores)

# @@ Cell 1707
np.mean(cvscores)
cvscores

# @@ Cell 1708
xtest

# @@ Cell 1709
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1710
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1711
ans.to_csv('data.csv',index=False)

# @@ Cell 1712
xtest

# @@ Cell 1713
xtrain

# @@ Cell 1714
ytrain

# @@ Cell 1715
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1716
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1717
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1718
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1719
ans.to_csv('data.csv',index=False)

# @@ Cell 1720
xtrain

# @@ Cell 1721
xtrain[topcols]

# @@ Cell 1722
xtest[topcols]

# @@ Cell 1723
xtrain[topcols]

# @@ Cell 1724
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1725
xtrain

# @@ Cell 1726
pd.read_csv('xtrain_v1')

# @@ Cell 1727
pd.read_csv('xtrain_v1.csv')

# @@ Cell 1728
pd.read_csv('train_v1.csv')

# @@ Cell 1729
xtrain=pd.read_csv('train_v1.csv')
xtrain=xtrain[xtrain.columns[1:80]]

# @@ Cell 1730
xtrain

# @@ Cell 1731
xtest=pd.read_csv('test_v2.csv')

# @@ Cell 1732
xtest

# @@ Cell 1733
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
scl=scaler.fit(xtrain[intcols])
ttrain=scaler.transform(xtrain[intcols])
ttest=scaler.transform(xtest[intcols])
xtest[intcols]=pd.DataFrame(ttest, columns=intcols)
xtrain[intcols]=pd.DataFrame(ttrain, columns=intcols)

# @@ Cell 1734
xtest

# @@ Cell 1735
xtest[intcols]

# @@ Cell 1736
xtrain[intcols]

# @@ Cell 1737
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest[intcols]

# @@ Cell 1738
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1739
np.mean(cvscores)
cvscores

# @@ Cell 1740
np.mean(cvscores)
#cvscores

# @@ Cell 1741
np.mean(cvscores)
cvscores

# @@ Cell 1742
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(30).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

#np.mean(cvscores)
#cvscores

# @@ Cell 1743
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1744
np.mean(cvscores)
cvscores

# @@ Cell 1745
feat_importances = pd.Series(regr.feature_importances_, index=xtrain.columns)
feat_importances.nlargest(30).plot(kind='bar')
topcols=feat_importances.nlargest(30).index

#np.mean(cvscores)
#cvscores

# @@ Cell 1746
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1747
np.mean(cvscores)
cvscores

# @@ Cell 1748
np.mean(cvscores),cvscores

# @@ Cell 1749
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain[topcols], yltrain)

# @@ Cell 1750
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1751
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=60, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
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

# @@ Cell 1752
np.mean(cvscores),cvscores

# @@ Cell 1753
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1754
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1755
regr = RandomForestRegressor(max_depth=60, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1756
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1757
ans.to_csv('data.csv',index=False)

# @@ Cell 1758
regr = RandomForestRegressor(max_depth=60, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1759
xtest

# @@ Cell 1760
xtrain

# @@ Cell 1761
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1762
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1763
ans.to_csv('data.csv',index=False)

# @@ Cell 1764
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1765
xtrain

# @@ Cell 1766
xtest

# @@ Cell 1767
xtest
test

# @@ Cell 1768
xtest
#test

# @@ Cell 1769
xtest
test

# @@ Cell 1770
xtest
#test

# @@ Cell 1771
xtest[topcols]
#test

# @@ Cell 1772
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1773
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1774
ans.to_csv('data.csv',index=False)

# @@ Cell 1775
xtrain[intcols]

# @@ Cell 1776
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest[intcols]

# @@ Cell 1777
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
xtest

# @@ Cell 1778
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]
test

# @@ Cell 1779
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 1780
train.head()

# @@ Cell 1781
xtest=test[test.columns[1:]]

# @@ Cell 1782
ytrain=train['SalePrice']
train.shape, test.shape

#Salesprice is skewed to the right
ytrain.hist(bins=50)
plt.xlabel('SalePrice',fontsize=12)
plt.ylabel('Counts',fontsize=12)
plt.title('SalePrice Histogram')
ytrain.describe()

# @@ Cell 1783
f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(),vmin=.2,square=True,mask=mask);

# @@ Cell 1784
#top correlated variables

s=train.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
corr=pd.DataFrame(s)
corr=corr.rename(columns={0: 'abs(r)'})
corr[corr['abs(r)']>0.5]

# @@ Cell 1785
#top correlated variables with respect to salesprice
corrdf=train.corr()
topscorr=pd.DataFrame(corrdf[corrdf['SalePrice']>0.50]['SalePrice'].abs().sort_values(ascending=False))
mask = np.zeros_like(train[topscorr.index].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[topscorr.index].corr(),square=True,annot=True, mask=mask, fmt='.2f',annot_kws={'size': 10},vmin=.2)
topscorr

# @@ Cell 1786
#About the highest R value OverallQual
# this is an ordinal categorical variable represented by integers
np.sort(train.OverallQual.unique())

# @@ Cell 1787
#Boxplots to show trends of Saleprice vs Overall Qual
bplot_data=train[['SalePrice', 'OverallQual']]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=bplot_data)
plt.xlabel('OverallQual',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)
plt.title('Boxplots of Overall Qual')

# @@ Cell 1788
#Continuous variable
train.GrLivArea

# @@ Cell 1789
#There are two outliers GrLvArea>4000, but generally there is a linear relationship
GrLiv_data=train[['SalePrice', 'GrLivArea']]
GrLiv_data.plot.scatter(x='GrLivArea', y='SalePrice',ylim=(0,800000));
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1790
#Possible candidates for outliers
train[(GrLiv_data['GrLivArea']>4000) & (GrLiv_data['SalePrice']<200000) ][['GrLivArea','OverallQual','SalePrice']]

# @@ Cell 1791
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)

# @@ Cell 1792
print(misval.shape[0]," total missing values" )
print((misval['misval_train']!=0).sum(), "missing values in training set")
print((misval['misval_test']!=0).sum(), "missing values in test set")
misval

# @@ Cell 1793
# impute NA to category NA
cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

train['MiscFeature'].unique(),xtest['MiscFeature'].unique()

# @@ Cell 1794
#NA and 0's match there are no NAs that are missing values!
print(xtest[(xtest['Fireplaces']==0) & (xtest['FireplaceQu']!='NA')].shape[0],
      train[(train['Fireplaces']==0) & (train['FireplaceQu']!='NA')].shape[0])

# @@ Cell 1795
#Three missing values for Poolarea
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

xtest[(xtest['PoolQC']== 'NA') & (xtest['PoolArea']>0)][['OverallQual', 'PoolQC', 'PoolArea']]

# @@ Cell 1796
fig = sns.catplot(y='OverallQual', x="PoolQC",order=["NA", "Fa","TA", "Gd","Ex"], data=train)
fig1 = sns.catplot(x='PoolQC', y="PoolArea", order=["NA", "Fa","TA", "Gd","Ex"], data=train)

# @@ Cell 1797
#slight increase in quality vs pool qc-- give FA for all missing data
xtest['PoolQC']=xtest['PoolQC'].replace({'NA':'Fa'})
print(xtest[(xtest['PoolQC']=='NA') & (xtest['PoolArea']>0)].shape[0],
train[(train['PoolQC']=='NA') & (train['PoolArea']>0)].shape[0])

# @@ Cell 1798
## Training set: 81 entries with no garage 

cols=['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
print(train.filter(regex='Garage').isna().sum())
print('total number of entries with misvals is',
      pd.isnull(train[cols].any(axis=1)).sum())

# @@ Cell 1799
#Double check to make sure that all entries with no garage have zero values for 
#garagecars and garage area
train[pd.isnull(train['GarageType'])][['GarageCars','GarageArea']].sum()

# @@ Cell 1800
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

# @@ Cell 1801
cols= ['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea']
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].mode()[0]} 
    xtest[cols[i]].loc[[666,1116]]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

# @@ Cell 1802
#Cannot use 0 for area! use 440 as the next most common value.
print(train['GarageArea'].value_counts())
xtest.loc[1116,'GarageArea']=440
xtest.loc[[666,1116]].filter(regex='Garage')

# @@ Cell 1803
#Need to remove all NA's before running imputation

cols= ['GarageType','GarageFinish','GarageQual','GarageCond','GarageYrBlt']

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')

(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 1804
cols=train.filter(regex='Bsmt').loc[:, train.filter(regex='Bsmt').isnull().any()].columns

# @@ Cell 1805
print(train.filter(regex='Bsmt').isna().sum())
print('total number of entries in training set with misvals is',
      pd.isnull(train.filter(regex='Bsmt')).any(axis=1).sum())

#Total of 39 entries, there is 37 common entries that have missing values across
#and there are two unique entries that have misvals in exposure and type2

# @@ Cell 1806
train[pd.isnull(train[cols]).any(axis=1)].filter(regex='Bsmt').loc[[332,948]]

# @@ Cell 1807
#Impute 332 to Rec, it is not unfinished because all 3 bsmtfinSF >0
print(train['BsmtFinType2'].value_counts())
#Impute 948 to No
print(train['BsmtExposure'].value_counts())

train.loc[332,'BsmtFinType2']='Rec'
train.loc[948,'BsmtExposure']='No'
train.loc[[332,948]].filter(regex='Bsmt')

# @@ Cell 1808
print(xtest.filter(regex='Bsmt').isna().sum())
print('total number of entries in test set with misvals is',
      pd.isnull(xtest.filter(regex='Bsmt')).any(axis=1).sum())



#First Fix the the NA values that are clearly meant to be 0
cols1=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
#temp[temp[cols].isnull().all(axis=1)].loc[[660,728]]
xtest.loc[[660,728],cols1]=0
xtest.loc[[660,728],cols1]

# @@ Cell 1809
#And then impute the Missing values for the 7 other values
temp=xtest[pd.isnull(xtest[cols]).any(axis=1)].filter(regex='Bsmt')
cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
temp[~temp[cols].isnull().all(axis=1)]

# @@ Cell 1810
#Impute missing values for the 7 entries
cols= ['BsmtQual','BsmtCond','BsmtExposure']
ind=temp[~temp[cols].isnull().all(axis=1)].index
for i in range(len(cols)):
    scale_mapper = {np.nan: train[cols[i]].value_counts().keys()[0]} 
    xtest[cols[i]].loc[ind]=xtest[cols[i]].replace(scale_mapper)
    print(train[cols[i]].mode()[0])

#Check if it worked    
xtest[cols].loc[ind]

# @@ Cell 1811
#Fill in the NA values
cols=xtest.filter(regex='Bsmt').columns.tolist()

train[cols]=train[cols].fillna('NA')
xtest[cols]=xtest[cols].fillna('NA')
#Check to see that there are no NA values
(train[cols].isna().sum().sum(),xtest[cols].isna().sum().sum())

# @@ Cell 1812
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 1813
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

# @@ Cell 1814
cols=['MasVnrArea','MasVnrType']

train[(train['MasVnrArea']<20) & (train['MasVnrArea']>0)][cols]
xtest[(xtest['MasVnrArea']<20) & (xtest['MasVnrArea']>0)][cols]


#Lets assume that the MasVnrArea of 1 is actually meant to be 0

# @@ Cell 1815
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

# @@ Cell 1816
misval=pd.DataFrame()
misval['misval_test']=xtest.isnull().sum()
misval['misval_test_%']=round(100*pd.DataFrame(xtest.isnull().sum())/(1459),1)

misval['misval_train']=pd.DataFrame(train.isnull().sum())
misval['misval_train_%']=round(100*pd.DataFrame(train.isnull().sum())/(1460),1)

misval['total_misval']=misval['misval_test']+misval['misval_train']
misval['total_misval_%']=round((100*misval['total_misval'])/(1460+1459),1)
misval=misval[misval['total_misval']>0].sort_values(by='total_misval_%',ascending=False)
misval

# @@ Cell 1817
cols=misval.index.tolist()
for i in range(len(cols)):
    #print(xtest[cols[i]].fillna(xtest[cols[i]].value_counts()[0]))
    xtest[cols[i]]=xtest[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    train[cols[i]]=train[cols[i]].fillna(train[cols[i]].value_counts().keys()[0])
    

# @@ Cell 1818
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

# @@ Cell 1819
import re
file=open('./categorical.txt','r')
stringlist=file.read().splitlines()

# @@ Cell 1820
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

# @@ Cell 1821
#dictlist.pop(0,14,13) remove all the numerical values
len(dictlist)

# @@ Cell 1822
#dictlist.pop(0,14,13) remove all the numerical values
len(dictlist)
dictlist.pop(0) 

# @@ Cell 1823
#dictlist.pop(0,14,13) remove all the numerical values
len(dictlist)
dictlist.pop(14) 

# @@ Cell 1824
#dictlist.pop(0,14,13) remove all the numerical values
len(dictlist)
dictlist.pop(13) 

# @@ Cell 1825
dictlist

# @@ Cell 1826
for i in range(len(dictlist)):
    [[key, value]] = dictlist[i].items()
    scale_mapper={k: v for v, k in enumerate(value[::-1])}
    print(key)
    print(scale_mapper)
    train[key]=train[key].replace(scale_mapper)
    xtest[key]=xtest[key].replace(scale_mapper)
#train[key]

# @@ Cell 1827
error_cols=xtest.select_dtypes(exclude=['int64','float']).columns.tolist()
error_cols
#train[x].unique()

# @@ Cell 1828
xtest[error_cols[4]].value_counts()

# @@ Cell 1829
xtest[error_cols[0]].value_counts()

# @@ Cell 1830
x=error_cols[0]
xtest[x].value_counts()

# @@ Cell 1831
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
scale_mapper= {'C (all)':6}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1832
x=error_cols[1]
xtest[x].value_counts()

# @@ Cell 1833
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
scale_mapper= {'NAmes':12}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1834
x=error_cols[2]
xtest[x].value_counts()

# @@ Cell 1835
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
scale_mapper= {'Duplex':2,'Twnhs':0,'2fmCon':3}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1836
x=error_cols[3]
xtest[x].value_counts()

# @@ Cell 1837
x=error_cols[4]
xtest[x].value_counts()

# @@ Cell 1838
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
scale_mapper= {'Wd_Sdng':1}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1839
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
scale_mapper= {'Wd Sdng':1}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1840
x=error_cols[5]
xtest[x].value_counts()

# @@ Cell 1841
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
scale_mapper= {'Wd Sdng':1,'CmentBd':11, 'Wd Shng':0,'Brk Cmn':14}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1842
x=error_cols[6]
xtest[x].value_counts()

# @@ Cell 1843
x=error_cols[7]
xtest[x].value_counts()

# @@ Cell 1844
x=error_cols[8]
xtest[x].value_counts()

# @@ Cell 1845
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
scale_mapper= {'NA':0}
 
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)
train[x].unique()

# @@ Cell 1846
x=error_cols[9]
xtest[x].value_counts()

# @@ Cell 1847
x=error_cols[10]
xtest[x].value_counts()

# @@ Cell 1848
#Test to make sure that that the encoding worked
temp=pd.read_csv("./train.csv")
x='Functional'
temp[x].value_counts(),train[x].value_counts()

# @@ Cell 1849
xtest.info()

# @@ Cell 1850
xtest['Condition2'].unique()

# @@ Cell 1851
fig = sns.boxplot(y='SalePrice', x="MSSubClass", data=train)
#MSsubclass is not ordinal

# @@ Cell 1852
#remove ID number

xtrain=train[train.columns[1:80]]

xtrain.head()

# @@ Cell 1853
x='MoSold'

#r=pd.read_csv("./train.csv")
#ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]

scale_mapper= {12: 0}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)


GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1854
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

# @@ Cell 1855
#Replace yr sold with num values 2006: 0,2007:1,2008:2,2009:3, 2010:4
# will need to one hot encode for regression
x='YrSold'


#r=pd.read_csv("./train.csv")
#ra=pd.read_csv("./test.csv")
#train[x]=r[x]
#xtest[x]=ra[x]
#xtest[x]

scale_mapper= {2006: 0,2007:1,2008:2,2009:3, 2010:4}
    
train[x]=train[x].replace(scale_mapper)
xtest[x]=xtest[x].replace(scale_mapper)

GrLiv_data=train[['SalePrice', x]]
GrLiv_data.plot.scatter(x, y='SalePrice',ylim=(0,800000));
plt.xlabel(x,fontsize=12)
plt.ylabel('SalePrice',fontsize=12)

# @@ Cell 1856
train.info()
xtest.info()

# @@ Cell 1857
train.info()
#xtest.info()

# @@ Cell 1858
#Change rest of the objects into int64, Now everything is numerically encoded

x=['MiscFeature','GarageQual','Heating','BsmtCond','RoofMatl']

train[x]=train[x].astype('int64')
xtest[x]=xtest[x].astype('int64')

# @@ Cell 1859
train.info()
xtest.info()

# @@ Cell 1860
train.info()
#xtest.info()

# @@ Cell 1861
xtrain=train[train.columns[1:80]]
xtrain

# @@ Cell 1862
#train_mod and xtest_mod 
train.to_csv('train_v1.csv',index=False)
xtest.to_csv('test_v1.csv',index=False)

# @@ Cell 1863
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=60, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
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

# @@ Cell 1864
np.mean(cvscores),cvscores

# @@ Cell 1865
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1866
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1867
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1868
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1869
ans.to_csv('data.csv',index=False)

# @@ Cell 1870
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]

# @@ Cell 1871
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]

# @@ Cell 1872
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]
intcols

# @@ Cell 1873
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]
nomcols

# @@ Cell 1874
cat=pd.read_csv("./categorical.csv")
catcols=cat['categories'].unique().tolist()
hotcols=cat[cat['encode']=='hot']['categories'].tolist()
nomcols=cat[cat['encode']=='nom']['categories'].tolist()
intcols = [col for col in xtrain.columns if col not in catcols]
intcols

# @@ Cell 1875
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(xtrain[intcols])
ttrain=scaler.transform(xtrain[intcols])
ttest=scaler.transform(xtest[intcols])
xtest[intcols]=pd.DataFrame(ttest, columns=intcols)
xtrain[intcols]=pd.DataFrame(ttrain, columns=intcols)

# @@ Cell 1876
xtrain

# @@ Cell 1877
xtest

# @@ Cell 1878
xtest[intcols]

# @@ Cell 1879
xtrain[intcols]

# @@ Cell 1880
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=60, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=100)
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

# @@ Cell 1881
np.mean(cvscores),cvscores

# @@ Cell 1882
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1883
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1884
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], ytrain)
y_pred=regr.predict(xtest[topcols])

# @@ Cell 1885
xtest

# @@ Cell 1886
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=y_pred

# @@ Cell 1887
ans.to_csv('data.csv',index=False)

# @@ Cell 1888
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1889
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1890
from sklearn.model_selection import GridSearchCV
yltrain=(np.log(ytrain))
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 60],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2,3,8],
    'n_estimators': [100, 300]
}
# Create a based model
rf = RandomForestRegressor(criterion='mse')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
                           
grid_search.fit(xtrain[topcols], yltrain)

# @@ Cell 1891
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
#xtrain[intcols]

# @@ Cell 1892
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
xtrain[intcols]

# @@ Cell 1893
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
xtest[intcols]

# @@ Cell 1894
xtrain.to_csv('xtrain_sscale.csv',index=False)
xtest.to_csv('xtest_sscale.csv',index=False)
xtest

# @@ Cell 1895
grid_search.best_params_,grid_search.best_score_

# @@ Cell 1896
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1897
kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
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

# @@ Cell 1898
np.mean(cvscores),cvscores

# @@ Cell 1899
#Test parity
plt.scatter(np.exp(y_test),np.exp(y_pred))
plt.xlabel('y_test')
plt.xlabel('y_pred')
x=[0,700000]
y=x
plt.plot(x,y)

# @@ Cell 1900
#Train parity
x=[10.5,14]
y=x
plt.scatter((y_train),(y_predtr))
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.plot(x,y)

# @@ Cell 1901
regr = RandomForestRegressor(max_depth=20, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
ytest=regr.predict(xtest[topcols])

# @@ Cell 1902
regr = RandomForestRegressor(max_depth=30, max_features= 5,min_samples_split=2, min_samples_leaf=3, random_state=0, n_estimators=300)
regr.fit(xtrain[topcols], yltrain)
ytest=regr.predict(xtest[topcols])

# @@ Cell 1903
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=ytest
ans.to_csv('data.csv',index=False)

# @@ Cell 1904
#One hot encoded AND num coded train and test sets

#hottest=pd.get_dummies(xtest[hotcols].astype(str))
#xtest.drop(hotcols, axis=1, inplace=True)
#xtest=pd.concat([xtest, hottest],axis=1)

hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
xtrain.drop(hotcols, axis=1, inplace=True)
xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1905
xtrain

# @@ Cell 1906
#One hot encoded AND num coded train and test sets

hottest=pd.get_dummies(xtest[hotcols].astype(str))
xtest.drop(hotcols, axis=1, inplace=True)
xtest=pd.concat([xtest, hottest],axis=1)

#hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1907
xtest

# @@ Cell 1908
xtrain

# @@ Cell 1909
xtrain.columns

# @@ Cell 1910
xtest.columns

# @@ Cell 1911
get_ipython().system('ls')

# @@ Cell 1912
get_ipython().system('ls')

train=pd.read_csv('xtrain_sscale.csv')

# @@ Cell 1913
train

# @@ Cell 1914
get_ipython().system('ls')

train=pd.read_csv('xtrain_sscale.csv')
test=pd.read_csv('xtest_sscale.csv')

# @@ Cell 1915
test

# @@ Cell 1916
test[numcols]

# @@ Cell 1917
test[nomcols]

# @@ Cell 1918
train[nomcols]

# @@ Cell 1919
train[hotcols]

# @@ Cell 1920
test[hotcols]

# @@ Cell 1921
tar=test.drop(hotcols, inplace=True)

# @@ Cell 1922
test[hotcols]

# @@ Cell 1923
tar=test.drop(hotcols)

# @@ Cell 1924
tar=test.drop(hotcols,axis=1)

# @@ Cell 1925
tar=test.drop(hotcols,axis=1)
tar

# @@ Cell 1926
tar=test.drop(hotcols,axis=1)
tar
test

# @@ Cell 1927
tar=test.drop(hotcols,axis=1)
tar

# @@ Cell 1928
#One hot encoded AND num coded train and test sets

hottest=pd.get_dummies(test[hotcols].astype(str))
#temptest=test.drop(hotcols, axis=1)
#fintest=pd.concat([temptest, hottest],axis=1)

hottrain=pd.get_dummies(xtrain[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1929
#One hot encoded AND num coded train and test sets

hottest=pd.get_dummies(test[hotcols].astype(str))
#temptest=test.drop(hotcols, axis=1)
#fintest=pd.concat([temptest, hottest],axis=1)

hottrain=pd.get_dummies(train[hotcols].astype(str))
#xtrain.drop(hotcols, axis=1, inplace=True)
#xtrain=pd.concat([xtrain,hottrain],axis=1)


#xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
#xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 1930
hottest

# @@ Cell 1931
hottest.shape[0]
hottrain.shape[0]

# @@ Cell 1932
hottest.shape[],hottrain.shape[0]

# @@ Cell 1933
hottest.shape,hottrain.shape

# @@ Cell 1934
hottest.shape,hottrain.shape
test.shape,train.shape

# @@ Cell 1935
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1936
hottest.shape #,hottrain.shape
#test.shape,train.shape

# @@ Cell 1937
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1938
train[hotcols]

# @@ Cell 1939
train[hotcols].astype(str)

# @@ Cell 1940
hottest=train[hotcols].astype(str)

# @@ Cell 1941
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1942
hottest=train[hotcols].astype(str)
hottrain=(train[hotcols].astype(str))

# @@ Cell 1943
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1944
hottest=get_dummies(train[hotcols[0]].astype(str))
hottrain=(train[hotcols[0]].astype(str))

# @@ Cell 1945
hottest=pd.get_dummies(train[hotcols[0]].astype(str))
hottrain=pd.get_dummies(train[hotcols[0]].astype(str))

# @@ Cell 1946
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1947
hottest=pd.get_dummies(train[hotcols[1]].astype(str))
hottrain=pd.get_dummies(train[hotcols[1]].astype(str))

# @@ Cell 1948
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1949
hottest=pd.get_dummies(train[hotcols[2]].astype(str))
hottrain=pd.get_dummies(train[hotcols[2]].astype(str))

# @@ Cell 1950
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1951
hottest=pd.get_dummies(train[hotcols[3]].astype(str))
hottrain=pd.get_dummies(train[hotcols[3]].astype(str))

# @@ Cell 1952
hottest.shape,hottrain.shape
#test.shape,train.shape

# @@ Cell 1953
hottest.shape==hottrain.shape
#test.shape,train.shape

# @@ Cell 1954
hottest.shape!=hottrain.shape
#test.shape,train.shape

# @@ Cell 1955
hotcols.shape[0]

# @@ Cell 1956
len(hotcols)

# @@ Cell 1957
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    if hottest.shape!=hottrain.shape
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1958
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1959
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    elif hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1960
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    elif hottest.shape==hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1961
tot=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[i]+tot
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    elif hottest.shape==hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1962
tot=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(train[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[1]+tot
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    elif hottest.shape==hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1963
tot

# @@ Cell 1964
hottest=pd.get_dummies(train[hotcols].astype(str))
hottest.shape

# @@ Cell 1965
hottest=pd.get_dummies(test[hotcols].astype(str))
hottest.shape

# @@ Cell 1966
tot=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[1]+tot
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    elif hottest.shape==hottrain.shape:
        print(hottest.shape,hottrain.shape,i)

# @@ Cell 1967
tot=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[1]+tot
    if hottest.shape!=hottrain.shape:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1968
tot=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[1]+tot
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1969
tot=0
tottrain=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))

    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1970
tot

# @@ Cell 1971
tottrain

# @@ Cell 1972
pd.concat[test,train]

# @@ Cell 1973
pd.concat(test,train)

# @@ Cell 1974
pd.concat([test,train])

# @@ Cell 1975
test

# @@ Cell 1976
pd.concat([train,test])

# @@ Cell 1977
train

# @@ Cell 1978
train

# @@ Cell 1979
train.shape[]

# @@ Cell 1980
train.shape,test.shape

# @@ Cell 1981
tot=0
tottrain=0
totcomp=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train,test]).astype(str)
    hotcomp=pd.get_dummies(comp)
    
    totcomp=comp.shape[1]=totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1982
tot=0
tottrain=0
totcomp=0
for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train,test]).astype(str)
    hotcomp=pd.get_dummies(comp)
    
    totcomp=comp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1983
pd.concat([train,test])

# @@ Cell 1984
pd.concat([train[hotcols[i]],test[hotcols[i]]])

# @@ Cell 1985
tot=0
tottrain=0
totcomp=0

for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train[hotcols[i],test[hotcols[i]]]).astype(str)
    hotcomp=pd.get_dummies(comp)
    
    
    totcomp=comp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1986
tot=0
tottrain=0
totcomp=0

for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train[hotcols[i],test[hotcols[i]]]]).astype(str)
    hotcomp=pd.get_dummies(comp)
    
    
    totcomp=comp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1987
pd.concat([train[hotcols[0]],test[hotcols[0]]])

# @@ Cell 1988
pd.concat([train[hotcols[0]],test[hotcols[0]]]).astype('int')

# @@ Cell 1989
tot=0
tottrain=0
totcomp=0

for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train[hotcols[i]],test[hotcols[i]]]).astype('int')
    hotcomp=pd.get_dummies(comp)
    
    
    totcomp=comp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1990
comp.sahpe[1]

# @@ Cell 1991
comp.shape[1]

# @@ Cell 1992
comp.shape[0]

# @@ Cell 1993
hottrain

# @@ Cell 1994
hotcomp

# @@ Cell 1995
tot=0
tottrain=0
totcomp=0

for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train[hotcols[i]],test[hotcols[i]]]).astype('int')
    hotcomp=pd.get_dummies(comp)
    
    
    totcomp=hotcomp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 1996
hotcomp

# @@ Cell 1997
totcomp

# @@ Cell 1998
totcomp, tottrain

# @@ Cell 1999
totcomp, tottrain,tot

# @@ Cell 2000
comp=pd.concat([train[hotcols,test[hotcols]).astype('int')
hotcomp=pd.get_dummies(comp)

# @@ Cell 2001
comp=pd.concat([train[hotcols,test[hotcols]]).astype('int')
hotcomp=pd.get_dummies(comp)

# @@ Cell 2002
comp=pd.concat([train[hotcols,test[hotcols]]]).astype('int')
hotcomp=pd.get_dummies(comp)

# @@ Cell 2003
comp=pd.concat([train[hotcols],test[hotcols]]).astype('int')
hotcomp=pd.get_dummies(comp)

# @@ Cell 2004
comp=pd.concat([train[hotcols],test[hotcols]]).astype('int')
hotcomp=pd.get_dummies(comp)

hotcomp.shape

# @@ Cell 2005
comp=pd.concat([train[hotcols],test[hotcols]]).astype('int')
hotcomp=pd.get_dummies(comp)

hotcomp

# @@ Cell 2006
tot=0
tottrain=0
totcomp=0

for i in range(len(hotcols)):
    hottest=pd.get_dummies(test[hotcols[i]].astype(str))
    hottrain=pd.get_dummies(train[hotcols[i]].astype(str))
    comp=pd.concat([train[hotcols[i]],test[hotcols[i]]]).astype(str)
    hotcomp=pd.get_dummies(comp)
    
    
    totcomp=hotcomp.shape[1]+totcomp
    tot=hottest.shape[1]+tot
    tottrain=hottrain.shape[1]+tottrain
    if hottest.shape[1]!=hottrain.shape[1]:
        print(hottest.shape,hottrain.shape,i)
    #elif hottest.shape==hottrain.shape:
     #   print(hottest.shape,hottrain.shape,i)

# @@ Cell 2007
totcomp, tottrain,tot

# @@ Cell 2008
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

hotcomp

# @@ Cell 2009
xtrain.shape[0]

# @@ Cell 2010
hotcomp.loc[:1460]

# @@ Cell 2011
hotcomp.iloc[:1460]

# @@ Cell 2012
hotcomp.iloc[:1460].shape

# @@ Cell 2013
hotcomp.iloc[:1460]

# @@ Cell 2014
xtest.shape[0]

# @@ Cell 2015
hotcomp.iloc[1460:]

# @@ Cell 2016
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

# @@ Cell 2017
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

xtest_hot

# @@ Cell 2018
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

xtest_hot

temptrain=pd.concat([train,xtest_hot],axis=1)
temptest=pd.concat([test,xtest_hot],axis=1)

# @@ Cell 2019
temptrain.shape,temptest.shape

# @@ Cell 2020
temptrain=temptrain.drop(hotcols,axis=1)
temptrain=temptest.drop(hotcols,axis=1)

# @@ Cell 2021
temptrain.shape,temptest.shape

# @@ Cell 2022
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

xtest_hot

temptrain=pd.concat([train,xtest_hot],axis=1)
temptest=pd.concat([test,xtest_hot],axis=1)

# @@ Cell 2023
temptrain.shape,temptest.shape

# @@ Cell 2024
temptrain=temptrain.drop(hotcols,axis=1)
temptrain=temptest.drop(hotcols,axis=1)

# @@ Cell 2025
temptrain.shape,temptest.shape

# @@ Cell 2026
temptrain=temptrain.drop(hotcols,axis=1)
temptest=temptest.drop(hotcols,axis=1)

# @@ Cell 2027
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

xtest_hot

temptrain=pd.concat([train,xtest_hot],axis=1)
temptest=pd.concat([test,xtest_hot],axis=1)

# @@ Cell 2028
temptrain=temptrain.drop(hotcols,axis=1)
temptest=temptest.drop(hotcols,axis=1)

# @@ Cell 2029
temptrain.shape,temptest.shape

# @@ Cell 2030
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]

xtest_hot

temptrain=pd.concat([train,xtest_hot],axis=1)
temptest=pd.concat([test,xtest_hot],axis=1)

temptrain=temptrain.drop(hotcols,axis=1)
temptest=temptest.drop(hotcols,axis=1)

xtrain=temptrain
xtest=temptest

# @@ Cell 2031
xtrain

# @@ Cell 2032
xtest

# @@ Cell 2033
xtest[hotcols]

# @@ Cell 2034
xtest[numcols[]

# @@ Cell 2035
xtest[numcols]

# @@ Cell 2036
xtest[nomcols]

# @@ Cell 2037
xtest[intcols]

# @@ Cell 2038
xtrain[intcols]

# @@ Cell 2039
xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 2040
xtrain

# @@ Cell 2041
xtest

# @@ Cell 2042
ytrain

# @@ Cell 2043
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
clf.score(X, y)

# @@ Cell 2044
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)

# @@ Cell 2045
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)

# @@ Cell 2046
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,).fit(X, y)
clf.score(X, y)
alpha_

# @@ Cell 2047
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,).fit(X, y)
clf.score(X, y)
clf.alpha_

# @@ Cell 2048
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,).fit(X, y)
clf.score(X, y)
clf.cv_scores_

# @@ Cell 2049
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,).fit(X, y)
clf.score(X, y)
clf.cv_values_

# @@ Cell 2050
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,store_cv_values=True).fit(X, y)
clf.score(X, y)
clf.cv_values_

# @@ Cell 2051
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5,store_cv_values=False).fit(X, y)
clf.score(X, y)
clf.cv_values_

# @@ Cell 2052
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)
clf.cv_values_

# @@ Cell 2053
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)

clf.get_params

# @@ Cell 2054
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)

clf.alpha_

# @@ Cell 2055
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X, y)

#clf.alpha_

# @@ Cell 2056
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score

#clf.alpha_

# @@ Cell 2057
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X,y)

#clf.alpha_

# @@ Cell 2058
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X,y)

# @@ Cell 2059
X

# @@ Cell 2060
xtrain

# @@ Cell 2061
xtest

# @@ Cell 2062
xtrain

# @@ Cell 2063
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)

# @@ Cell 2064
comp

# @@ Cell 2065
comp.tail

# @@ Cell 2066
comp.tail()

# @@ Cell 2067
comp.head()

# @@ Cell 2068
comp

# @@ Cell 2069
xtrain_hot=hotcomp.iloc[:1460]

# @@ Cell 2070
xtrain_hot

# @@ Cell 2071
hotcomp.iloc[1460:1461]

# @@ Cell 2072
hotcomp.iloc[1460:1462]

# @@ Cell 2073
hotcomp.iloc[1460:1463]

# @@ Cell 2074
hotcomp.iloc[1460:1464]

# @@ Cell 2075
comp=pd.concat([train[hotcols],test[hotcols]]).astype(str)
hotcomp=pd.get_dummies(comp)

xtrain_hot=hotcomp.iloc[:1460]
xtest_hot=hotcomp.iloc[1460:]


temptrain=pd.concat([train,xtrain_hot],axis=1)
temptest=pd.concat([test,xtest_hot],axis=1)

temptrain=temptrain.drop(hotcols,axis=1)
temptest=temptest.drop(hotcols,axis=1)

xtrain=temptrain
xtest=temptest

xtrain.to_csv('xtrain_sscale_hot.csv',index=False)
xtest.to_csv('xtest_sscale_hot.csv',index=False)

# @@ Cell 2076
xtrain

# @@ Cell 2077
xtest

# @@ Cell 2078
xtrain

# @@ Cell 2079
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X,y)

# @@ Cell 2080
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2081
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2082
clf.predict(xtest)

# @@ Cell 2083
ytest=clf.predict(xtest)

# @@ Cell 2084
clf

# @@ Cell 2085
test

# @@ Cell 2086
train=pd.read_csv("./train.csv")
test=pd.read_csv('./test.csv')

# @@ Cell 2087
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=ytest
ans.to_csv('data_ridge.csv',index=False)

# @@ Cell 2088
ans

# @@ Cell 2089
yrg=pd.read_csv('data.csv')

# @@ Cell 2090
plt.scatter(yrg,ytest)

# @@ Cell 2091
yrg

# @@ Cell 2092
plt.scatter(yrg['SalePrice'],ypred)

# @@ Cell 2093
plt.scatter(yrg['SalePrice'],ytest)

# @@ Cell 2094
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    ridge = Ridge(alpha=1)
    ridge.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    y_predtr=regr.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2095
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2096
cvscores

# @@ Cell 2097
plt.scatter(y_test,y_pred)

# @@ Cell 2098
plt.scatter(y_train,y_predtr)

# @@ Cell 2099
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 2100
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2101
cvscores,trscores

# @@ Cell 2102
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,5],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2103
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2104
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,5],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2105
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,2],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2106
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,1],cv=5).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2107
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,1],cv=None,store_cv_values=True).fit(X, y)
clf.score(X,y)

clf.alpha_

# @@ Cell 2108
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,1],cv=None,store_cv_values=True).fit(X, y)
clf.score(X,y)

clf.cv_values

# @@ Cell 2109
from sklearn.linear_model import RidgeCV
X = xtrain
y = np.log(ytrain)

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,1],cv=None,store_cv_values=True).fit(X, y)
clf.score(X,y)

clf.cv_values_

# @@ Cell 2110
clf.cv_values_.shape

# @@ Cell 2111
np.mean(clf.cv_values_)

# @@ Cell 2112
np.mean(clf.cv_values_,axis=0)

# @@ Cell 2113
np.mean(clf.cv_values_,axis=1)

# @@ Cell 2114
np.mean(clf.cv_values_,axis=0)

# @@ Cell 2115
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=2)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2116
cvscores,trscores

# @@ Cell 2117
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=.001)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2118
cvscores,trscores

# @@ Cell 2119
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2120
cvscores,trscores

# @@ Cell 2121
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1.5)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2122
cvscores,trscores

# @@ Cell 2123
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=2)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2124
cvscores,trscores

# @@ Cell 2125
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=3)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2126
cvscores,trscores

# @@ Cell 2127
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=2)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2128
cvscores,trscores

# @@ Cell 2129
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2130
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 2131
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=40)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2132
cvscores,trscores

# @@ Cell 2133
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2134
cvscores,trscores

# @@ Cell 2135
cvscores,trscores

# @@ Cell 2136
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2137
plt.scatter(np.exp(y_test),np.exp(y_pred))

# @@ Cell 2138
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols].iloc[train_index], xtrain[topcols].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2139
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2140
cvscores,trscores

# @@ Cell 2141
cvscores.mean(,trscores

# @@ Cell 2142
cvscores.mean(),trscores

# @@ Cell 2143
np.mean(cvscores),trscores

# @@ Cell 2144
y_pred

# @@ Cell 2145
np.exp(y_pred)

# @@ Cell 2146
np.exp(y_pred)>50

# @@ Cell 2147
np.exp(y_pred)>1750000

# @@ Cell 2148
np.argwhere(np.exp(y_pred)>1750000)

# @@ Cell 2149
np.exp(y_pred)

# @@ Cell 2150
np.exp(y_pred).shape

# @@ Cell 2151
plt.scatter((y_test),(y_pred))

# @@ Cell 2152
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=.5)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2153
np.mean(cvscores),trscores

# @@ Cell 2154
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2155
plt.scatter((y_test),(y_pred))

# @@ Cell 2156
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=0)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2157
np.mean(cvscores),trscores

# @@ Cell 2158
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=.001)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2159
np.mean(cvscores),trscores

# @@ Cell 2160
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2161
plt.scatter((y_test),(y_pred))

# @@ Cell 2162
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=0)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2163
np.mean(cvscores),trscores

# @@ Cell 2164
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2165
plt.scatter((y_test),(y_pred))

# @@ Cell 2166
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=0.01)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2167
np.mean(cvscores),trscores

# @@ Cell 2168
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2169
plt.scatter((y_test),(y_pred))

# @@ Cell 2170
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2171
np.mean(cvscores),trscores

# @@ Cell 2172
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=2)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2173
np.mean(cvscores),trscores

# @@ Cell 2174
cvscores,trscores

# @@ Cell 2175
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=3)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2176
cvscores,trscores

# @@ Cell 2177
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=50000)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2178
cvscores,trscores

# @@ Cell 2179
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=5)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2180
cvscores,trscores

# @@ Cell 2181
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=6)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2182
cvscores,trscores

# @@ Cell 2183
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2184
cvscores,trscores

# @@ Cell 2185
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=100)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2186
cvscores,trscores

# @@ Cell 2187
mean(cvscores),trscores

# @@ Cell 2188
np.mean(cvscores),trscores

# @@ Cell 2189
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1000)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2190
np.mean(cvscores),trscores

# @@ Cell 2191
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=100)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2192
np.mean(cvscores),trscores

# @@ Cell 2193
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=50)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2194
np.mean(cvscores),trscores

# @@ Cell 2195
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=40)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2196
np.mean(cvscores),trscores

# @@ Cell 2197
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2198
np.mean(cvscores),trscores

# @@ Cell 2199
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2200
np.mean(cvscores),trscores

# @@ Cell 2201
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2202
np.mean(cvscores),trscores

# @@ Cell 2203
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=30)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2204
np.mean(cvscores),trscores

# @@ Cell 2205
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=40)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2206
np.mean(cvscores),trscores

# @@ Cell 2207
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=30)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2208
np.mean(cvscores),trscores

# @@ Cell 2209
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=30)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2210
np.mean(cvscores),trscores

# @@ Cell 2211
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2212
plt.scatter((y_test),(y_pred))

# @@ Cell 2213
cvscores,trscores

# @@ Cell 2214
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2215
cvscores,trscores

# @@ Cell 2216
cvscores,trscores,cvscores.mean()

# @@ Cell 2217
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2218
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2219
plt.scatter((y_test),(y_pred))

# @@ Cell 2220
rid.fit(xtrain,ytrain)
rid.predict(xtest)

# @@ Cell 2221
rid.fit(xtrain,ytrain)
ypred=rid.predict(xtest)

# @@ Cell 2222
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=np.exp(ytest)
ans.to_csv('data_ridge.csv',index=False)

# @@ Cell 2223
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=np.exp(ypred)
ans.to_csv('data_ridge.csv',index=False)

# @@ Cell 2224
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=np.exp(ypred)
ans.to_csv('data_ridg.csv',index=False)

# @@ Cell 2225
plt.plot(np.exp(yrg['Saleprice']),np.exp(ypred))

# @@ Cell 2226
plt.plot(np.exp(yrg['SalePrice']),np.exp(ypred))

# @@ Cell 2227
yrg['SalePrice']

# @@ Cell 2228
ypred

# @@ Cell 2229
plt.plot(np.exp(yrg['SalePrice']),ypred)

# @@ Cell 2230
plt.scatter(np.exp(yrg['SalePrice']),ypred)

# @@ Cell 2231
x=[0,60000]
y=[0,60000]

# @@ Cell 2232
plt.scatter(np.exp(yrg['SalePrice']),ypred)
plt.plot(x,y)

# @@ Cell 2233
x=[0,600000]
y=[0,600000]

# @@ Cell 2234
plt.scatter(np.exp(yrg['SalePrice']),ypred)
plt.plot(x,y)

# @@ Cell 2235
plt.scatter((y_test),(y_pred))

# @@ Cell 2236
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2237
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2238
np.exp(y_train)-np.exp(y_predtr)

# @@ Cell 2239
plt.scatter(np.exp(y_train),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2240
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2241
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2242
plt.scatter((y_train),(y_train)-(y_predtr))

# @@ Cell 2243
#Clearly nonlinear!
plt.scatter((y_train),(y_train)-(y_predtr))

# @@ Cell 2244
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter((y_predtr),(y_train)-(y_predtr))

# @@ Cell 2245
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2246
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2247
plt.scatter(np.exp(y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2248
xtrain.skew()

# @@ Cell 2249
ytrain.skew()

# @@ Cell 2250
xtrain.skew()

# @@ Cell 2251
ytrain.skew()

# @@ Cell 2252
np.log(ytrain).skew()

# @@ Cell 2253
np.log(xtrain).skew()

# @@ Cell 2254
(xtrain).skew()

# @@ Cell 2255
(xtrain).skew()[0]

# @@ Cell 2256
(xtrain).skew()

# @@ Cell 2257
(xtrain).skew().sort_values()

# @@ Cell 2258
np.abs((xtrain).skew()).sort_values()

# @@ Cell 2259
np.abs((xtrain).skew()).sort_values(ascending=True)

# @@ Cell 2260
np.abs((xtrain).skew()).sort_values(ascending=False)

# @@ Cell 2261
np.abs((xtrain).skew()).sort_values(ascending=False)[:10]

# @@ Cell 2262
np.abs((xtrain).skew()).sort_values(ascending=False)[:12]

# @@ Cell 2263
np.abs((xtrain).skew()).sort_values(ascending=False)[]

# @@ Cell 2264
np.abs((xtrain).skew()).sort_values(ascending=False)[5]

# @@ Cell 2265
np.abs((xtrain).skew()).sort_values(ascending=False)[10]

# @@ Cell 2266
np.abs((xtrain).skew()).sort_values(ascending=False)

# @@ Cell 2267
np.abs((xtrain[intcols]).skew()).sort_values(ascending=False)

# @@ Cell 2268
xtrain['MiscVal'].hist()

# @@ Cell 2269
xtrain['PoolArea'].hist()

# @@ Cell 2270
xtrain['LotArea'].hist()

# @@ Cell 2271
xtrain['LotArea']

# @@ Cell 2272
xtrain['BsmtFinSF2'].hist()

# @@ Cell 2273
xtrain['MasVnrArea'].hist()

# @@ Cell 2274
np.log(xtrain['MasVnrArea']).hist()

# @@ Cell 2275
np.log(1+xtrain['MasVnrArea']).hist()

# @@ Cell 2276
np.log(xtrain['MasVnrArea']).hist()

# @@ Cell 2277
xtrain['MasVnrArea']**1/3.hist()

# @@ Cell 2278
xtrain['MasVnrArea']**(1/3).hist()

# @@ Cell 2279
np.log(xtrain['MasVnrArea']).hist()

# @@ Cell 2280
np.power(xtrain['MasVnrArea'],1/3).hist()

# @@ Cell 2281
np.power(xtrain['MasVnrArea'],1/4).hist()

# @@ Cell 2282
xtrain[topcols]

# @@ Cell 2283
xtrain.columns

# @@ Cell 2284
xtrain.str.contains('MoSold')

# @@ Cell 2285
print(xtrain.filter(like='MoSold').columns)

# @@ Cell 2286
print(xtrain.filter(like='MsZoning').columns)

# @@ Cell 2287
print(xtrain.filter(like='MSZoning').columns)

# @@ Cell 2288
print(xtrain.filter(like='Foundation').columns)

# @@ Cell 2289
print(xtrain.filter(like='CentralAir').columns)

# @@ Cell 2290
topcols

# @@ Cell 2291
topcols[0]

# @@ Cell 2292
topcols[5]

# @@ Cell 2293
len(topcols)

# @@ Cell 2294
topcols[30]

# @@ Cell 2295
topcols[20]

# @@ Cell 2296
topcols[29]

# @@ Cell 2297
topcols

# @@ Cell 2298
topcols[28]

# @@ Cell 2299
topcols[0:20]

# @@ Cell 2300
topcols[0:25]

# @@ Cell 2301
topcols[0:24]

# @@ Cell 2302
topcols[0:23]

# @@ Cell 2303
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2304
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2305
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2306
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2307
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2308
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2309
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2310
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2311
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2312
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2313
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=100)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2314
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2315
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[topcols[0:23]].iloc[train_index], xtrain[topcols[0:23]].iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=15)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2316
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2317
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=15)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2318
cvscores,trscores,np.mean(cvscores)

# @@ Cell 2319
cvscores,trscores,np.mean(cvscores),np.std(cvcores)

# @@ Cell 2320
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2321
np.exp(.138)

# @@ Cell 2322
np.expn(.138**2)

# @@ Cell 2323
np.exp(.138**2)

# @@ Cell 2324
np.mean((y_train)-(y_predtr))

# @@ Cell 2325
np.std((y_train)-(y_predtr))

# @@ Cell 2326
plt.scatter(np.exp(y_train),np.exp(y_predtr))

# @@ Cell 2327
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter((y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2328
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter((y_predtr),(y_train)-(y_predtr))

# @@ Cell 2329
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter((y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2330
plt.scatter(np.exp(y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2331
np.exp(y_train)

# @@ Cell 2332
y_train

# @@ Cell 2333
ytrain

# @@ Cell 2334
ytrain.hist()

# @@ Cell 2335
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter(np.exp(y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2336
np.std(np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2337
plt.scatter((y_test),(y_pred))

# @@ Cell 2338
plt.scatter((y_test),(y_pred))
np.std(np.exp(y_test)-np.exp(y_pred))

# @@ Cell 2339
plt.scatter((y_test),(y_pred))
plt.scatter(np.exp(y_pred),np.exp(y_test)-np.exp(y_pred))
np.std(np.exp(y_test)-np.exp(y_pred))

# @@ Cell 2340
#plt.scatter((y_test),(y_pred))
plt.scatter(np.exp(y_pred),np.exp(y_test)-np.exp(y_pred))
np.std(np.exp(y_test)-np.exp(y_pred))

# @@ Cell 2341
plt.scatter((y_test),(y_pred))
#plt.scatter(np.exp(y_pred),np.exp(y_test)-np.exp(y_pred))
#np.std(np.exp(y_test)-np.exp(y_pred)

# @@ Cell 2342
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=((ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=15)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2343
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter(np.exp(y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2344
#Clearly nonlinear!
#Use log transform on skewed y variable
plt.scatter((y_predtr),(y_train)-(y_predtr))

# @@ Cell 2345
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=15)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2346
plt.scatter(np.exp(y_predtr),np.exp(y_train)-np.exp(y_predtr))

# @@ Cell 2347
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2348
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2349
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2350
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=100)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2351
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2352
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2353
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2354
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=5)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2355
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2356
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=1)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2357
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2358
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2359
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2360
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=20)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2361
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2362
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=50)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2363
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2364
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5)
cvscores=[]
trscores=[]
yltrain=(np.log(ytrain))
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
    y_train, y_test = yltrain.iloc[train_index], yltrain.iloc[test_index]
    rid = Ridge(alpha=10)
    rid.fit(X_train, y_train)
    y_pred=rid.predict(X_test)
    y_predtr=rid.predict(X_train)
    
    cvscores.append(np.sqrt(mean_squared_error((y_test), (y_pred))))
    
    trscores.append(np.sqrt(mean_squared_error((y_train), (y_predtr))))

# @@ Cell 2365
cvscores,trscores,np.mean(cvscores),np.std(cvscores)

# @@ Cell 2366
rid.fit(xtrain,ytrain)
ypred=rid.predict(xtest)

# @@ Cell 2367
ans=pd.DataFrame()
ans['Id']=test['Id']
ans['SalePrice']=np.exp(ypred)
ans.to_csv('data_ridge.csv',index=False)

# @@ Cell 2368
plt.scatter((ytest),(ypred))

# @@ Cell 2369
rid.fit(xtrain,np.log(ytrain))
ypred=rid.predict(xtest)

# @@ Cell 2370
plt.scatter((np.log(ytest),(ypred))

# @@ Cell 2371
plt.scatter(np.log(ytest),(ypred))

# @@ Cell 2372
rid.fit(xtrain,np.log(ytrain))
ytrainpred=rid.predict(xtrain)
ypred=rid.predict(xtest)

# @@ Cell 2373
plt.scatter(np.log(ytest),(ytrainpred))

# @@ Cell 2374
ytrain.shape,ytrainpred.shape

# @@ Cell 2375
plt.scatter(ytrain,ytrainpred)

# @@ Cell 2376
plt.scatter(np.log(ytrain),ytrainpred)

# @@ Cell 2377
ypred

