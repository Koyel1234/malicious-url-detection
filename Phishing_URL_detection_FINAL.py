#!/usr/bin/env python
# coding: utf-8

# In[139]:


## Environment phishingurldetection
## install all necessary libraries

#pip install numpy
#pip install pandas
#pip install seaborn
#pip install matplotlib
#pip install tldextract
#pip install sklearn
#pip install scorecardpy
#pip install xgboost
#pip install lightgbm
#pip install catboost
#pip install flask
#pip install pywebio


# ## Importing Libraries

# In[2]:


# common libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset
# This is the dataset which we are using to fit model.

# In[3]:


# loading dataset
data = pd.read_csv("Phishing_Legitimate_full.csv")


# In[4]:


# first 10 rows of dataset
data.head(10)


# In[5]:


#dimention of dataset
data.shape


# In[6]:


# show all columns
pd.set_option('display.max_columns' , 100)


# In[7]:


# show first 5 (by deafult) rows along with all columns
data.head()


# In[8]:


# resetting default ma_columns option to avoid overload on notebook
pd.reset_option("display.max_columns")


# Now before doing any EDA or checking any kind of characteristic of the dataset, first we will check which features we can extract from elements. Among all features we can use only those features to train model which we can extract from links.
# <br>
# <br>
# We'll use ***urldata.csv*** dataset, which contains links and only result that link is malicious or not ***(all malicious links are not phishing links, but we are using this dataset to show how to extract elements and store in a same csv)*** as we have to use a csv file containing links to validate our model.

# In[9]:


#dataset to show elements extraction
data_url = pd.read_csv("urldata.csv", index_col=0)


# In[10]:


# showing first 10 rows
data_url


# - one quick check
# - here it is clearly seen that the row contanning label 'benign' and 'malicious' in column 'label' has value 0 and 1 respectively in 'result' column
# - to see if there is any column 'result' which has value mismatch with column 'label'
# 
# 

# In[11]:


data_url.label.value_counts()


# In[12]:


data_url.result.value_counts()


# In[13]:


flag = False

for i in range(len(data_url['label'])):
    if data_url.iloc[i,1]=='benign' and data_url.iloc[i,2]!=0:
        flag=True
    if data_url.iloc[i,1]=='malicious' and data_url.iloc[i,2]!=1:
        flag=True   
        
print(flag)        


# **There is no mismatch in 'label' and 'result' columns.**<br><br>
# We can **drop 'label'** column to show csv with extracted elements.

# In[14]:


#drop 'label' column
data_url.drop('label', axis=1, inplace=True)


# In[15]:


#show first 5 and last 5 rows 
data_url


# ## Code to Extract Features from Given URL

# In[16]:


# libraries for element extraction from urls
from urllib import parse
from urllib.parse import urlparse
import tldextract 


# In[17]:


data_url['UrlLength'] = [len(i) for i in data_url['url']]
data_url['NumDash'] = [i.count('-') for i in data_url['url']]
data_url['NumDots'] = [i.count('.') for i in data_url['url']]
data_url['NumUnderscore'] = [i.count('_') for i in data_url['url']]
data_url['NumPercent'] = [i.count('%') for i in data_url['url']]
data_url['NumAmpersand'] = [i.count('&') for i in data_url['url']]
data_url['NumHash'] = [i.count('#') for i in data_url['url']]
data_url['NumDashInHostname'] = [(urlparse(i).netloc).count('-') for i in data_url['url']]
data_url['NumNumericChars'] = [sum(c.isdigit() for c in i) for i in data_url['url']]
data_url['AtSymbol'] = [0 if c[1].find('@')==-1 else 1 for c in data_url['url'].iteritems()]
data_url['TildeSymbol'] = [0 if c[1].find('~')==-1 else 1 for c in data_url['url'].iteritems()]
data_url['NoHttps'] = [0 if c[1].find('https')==-1 else 1 for c in data_url['url'].iteritems()]
data_url['IpAddress'] = np.where(data_url['url'].str.match('^(http|https)://\d+\.\d+\.\d+\.\d+\.*') == True, 1, 0)
data_url['SubdomainLevel'] = [len(tldextract.extract(i).subdomain.split('.')) for i in data_url['url']]
data_url['NumQueryComponents'] = [len(dict(parse.parse_qs(parse.urlsplit(i, allow_fragments=False).query))) for i in data_url['url']]
data_url['HostnameLength'] = [len(urlparse(i).netloc) for i in data_url['url']]
data_url['PathLength'] = [len(urlparse(i).path) for i in data_url['url']]
data_url['QueryLength'] = [len(urlparse(i).query) for i in data_url['url']]
data_url['DoubleSlashInPath'] = [0 if urlparse(c[1]).path.find('//')==-1 else 1 for c in data_url['url'].iteritems()]   


# In[18]:


# first preview of data
data_url


# This is our validation data. We'll leave it now in this form. We'll check accuracy of our model over this dataset after final model fitting.

# In[19]:


# now setting options max_columns, max_rows, max_colwidth to see the results more clearly.
pd.set_option('display.max_columns' , None)
pd.set_option('display.max_rows' , None)
pd.set_option('display.max_colwidth' , None)


# In[20]:


# let's see first 1000 rows
## part of benign urls
data_url.head(1000)


# In[21]:


# let's see last 1000 rows
## part of malicious urls
data_url.tail(1000)


# In[22]:


# resetting default max_columns option to avoid overload on notebook
pd.reset_option("display.max_columns")
pd.reset_option("display.max_rows")
pd.reset_option("display.max_colwidth")


# ### ***Datset for this to check***

# In[23]:


# our dataset of use will be
# other than the elements which can't be extracted for now, we have also dropped 'id' column 
data_in_use = data[['UrlLength','NumDash','NumDots','NumUnderscore','NumPercent','NumAmpersand','NumHash','NumDashInHostname','NumNumericChars','AtSymbol','TildeSymbol','NoHttps','IpAddress','SubdomainLevel','NumQueryComponents','HostnameLength','PathLength','QueryLength','DoubleSlashInPath','CLASS_LABEL']]


# In[24]:


# base dataset preview
data_in_use


# In[25]:


# dimention of base dataset
data_in_use.shape


# It has **19 features of url** and one column **"CLASS_LABEL"** contains its label as **'Phishing Url' or 'Non-Phishing Url' (values 0 or 1 respectively).**

# ### ***About Dataset***

# In[26]:


data_in_use.info()


# All columns are in numeric (int64) datatype and there is **no missing value in any column**.

# In[27]:


# summary
data_in_use.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.9,0.95,0.99])


# Now check total number of unique values in column.

# In[28]:


#checking for number of unique values in each column
for col in data_in_use:
    print(col,'------>',len(data_in_use[col].unique()))


# Now here we can see that all the columns contains discrete values. So to check total number of unique values, max present value in a class we can convert each column into category type and check summary.

# In[29]:


data_in_use=data_in_use.astype(str) # all columns will convert to object datatype


# In[30]:


data_in_use.describe()


# From here (the last column **CLASS_LABEL**) we can also conclude that there are **5000 phishing** url among 10000 urls with no missing value. So **dataset is completely balanced**.

# Change all columns to numeric for further analysis.

# In[31]:


data_in_use=data_in_use.astype(int)  # all columns will convert to int32 datatype


# - So here data contains **NO MISSING VALUE** 
# - Dataset is **BALANCED**

# Now we proceed to fit our baseline model which is **Logistic Regression** with default parameters.

# ## Baseline Model (Default Logistic Model)
# Now we will create logistic regression model with dafault hyperparameters...with all features present in our base dataset. This will be our baseline model. As the dataset is balanced, we will use **Accuracy** as measure of model.
# <br><br>
# As we are applying logistic regression, not going to apply scaling here, as scaling with not have any impact on results of logistic regression.

# In[32]:


# import libraries for model fitting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[33]:


# divide data into base independent(X) and dependent(y) features

X = data_in_use.iloc[:,0:19]
y = data_in_use['CLASS_LABEL']


# In[34]:


# show base independent and dependent data

print("data_in_use")
print('\n')

# Independent Features
print('Independent Features:')
print('Dimention:',X.shape)
print('Column Names:')
print(X.columns)
print('First few rows:')
pd.set_option('display.max_columns' , 20)
print(X.head())
pd.reset_option('display.max_columns')

print('\n')

# Dependent Features
print('Dependent Feature: CLASS_LABEL')
print('Dimention:',y.shape)
print('First few values:')
print(y.head())


# In[35]:


# divide data into base training and testing set
## we will use 20% as test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("data_in_use")
print('Training split : X_train_shape =', X_train.shape, ',   y_train_shape =', y_train.shape)
print('Testing split : X_test_shape =', X_test.shape, ',   y_test_shape =', y_test.shape)


# In[36]:


# fit logistic model with default hyperparameters

## model training
classifier = LogisticRegression(random_state=0)
logistic_model = classifier.fit(X_train, y_train)


## prediction
predictions_training = logistic_model.predict(X_train)
score=cross_val_score(LogisticRegression(random_state=0),X_train_new, y_train, cv=ssplit)
predictions_training_CV=score.mean()     
predictions_testing = logistic_model.predict(X_test)

## Confusion Matrix
cm = confusion_matrix(y_test, predictions_testing)
print("Confusion Matrix : \n", cm)

## Classification Report
print(classification_report(y_test, predictions_testing))

## probability of falling into a particular class from each observation
probs = logistic_model.predict_proba(X_test)
probs = probs[:,1]

## AUC (Area Under Curve) Score
auc_score = roc_auc_score(y_test, probs)
print("Logistic Model (baseline) ROC AUC = ",format(auc_score,"0.2f"))


## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')

## ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, linestyle='--', label='Logistic (Baseline) Model')
plt.title("ROC Curve for Logistic Regression (Baseline Model)")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# In[197]:


## Training and Tesing Accuracy
print("Training accuracy of Logistic Regression: ", format(accuracy_score(y_train,predictions_training),"0.2f"))
print("Training accuracy of Logistic Regression from Cross Validation: ", format(predictions_training_CV,"0.2f"))
print("Testing accuracy of Logistic Regression: ", format(accuracy_score(y_test, predictions_testing),"0.2f")) 


# - **Quite good training and testing accuracy**. Here **seems no such overfitting issue**.
# - Next we will do feature scaling which we skipped in baseline model. But now we are going to use some algorithms which will require feature scaling.
# - Next we will perform some **new feature creation** from **EDA, Weight of Evidence** techniques and **score** of being anomalous of an observation from **Isolation Forest**.
# - Then we'll **fit some different models, perform hyperparameter tuning**.
# - Finally apply **Voting Classifier** Algorithm **to get final result**.
# - We'll **apply all told process over base training data *X_train***.

# # Feature Creation

# ## ***From EDA of Extracted Features***

# In[37]:


X_train.describe()


# In[38]:


# Combine scaled training and testing data
X_tr_y_tr = pd.concat([X_train,y_train], axis=1)


# #### ***Stacked Bar Plot from Crosstab***

# In[39]:


for col in X_tr_y_tr:
    pal = ['royalblue','lightblue']
    crstb_2 = pd.crosstab(X_tr_y_tr['CLASS_LABEL'], X_tr_y_tr[col]).apply(lambda r: r/r.sum()*100, axis=1)
    ax_1=crstb_2.plot.bar(stacked = True)
    display(crstb_2)

    plt.legend(bbox_to_anchor=(1.2,0.8),title='c')
    plt.title('For {}'.format(col))

    for rec in ax_1.patches:
        height=rec.get_height()
        ax_1.text(rec.get_x()+rec.get_width() / 2,
             rec.get_y()+height / 2,
             "{:.0f}%".format(height),
             ha='center',
             va='bottom')
    
    plt.show() 


# #### ***Countplot for All Features as well as Target Variable***

# In[40]:


for col in X_tr_y_tr:
    
    fig, ax=plt.subplots(1,2)
    sns.countplot(X_tr_y_tr[X_tr_y_tr['CLASS_LABEL']==1][col],orient="v", ax=ax[0])
    sns.countplot(X_tr_y_tr[X_tr_y_tr['CLASS_LABEL']==0][col],orient="v", ax=ax[1])
    plt.show()


# In[41]:


for col in X_tr_y_tr:
    sns.countplot(col,hue='CLASS_LABEL',data=X_tr_y_tr)
    plt.show()


# #### ***Boxplot***

# In[42]:


X_tr_y_tr['CLASS_LABEL']=X_tr_y_tr['CLASS_LABEL'].astype(int)


# In[43]:


for col in X_train:
    sns.boxplot(X_train[col], X_tr_y_tr['CLASS_LABEL'], orient='h')
    plt.show()


# In[ ]:





# #### ***Pairplot***

# In[44]:


sns.pairplot(hue='CLASS_LABEL',data=X_tr_y_tr)


# #### ***Newly Created Features from EDA***

# In[45]:


# Newly created features from EDA. We'll add them as new columns in base scaled training data
c1 = np.where(X_train['NumDots']>=4,1,0)
c2 = np.where(X_train['SubdomainLevel']>=3,1,0)
c3 = np.where(X_train['NumDash']<=1,1,0)
c4 = np.where(X_train['NumQueryComponents']==0,1,0)
c5 = np.where(X_train['NumNumericChars']==0,1,0)


# In[46]:


# create new dataframe for new features
new_features = pd.concat([pd.DataFrame(c1,columns=["NumDots_GrtrEql_4_EDA"]),pd.DataFrame(c2,columns=["SubdomainLevel_GrtrEql_3_EDA"]),pd.DataFrame(c3,columns=["NumDash_LsEql_1_EDA"]),pd.DataFrame(c4,columns=["NumQueryComponents==0_EDA"]),pd.DataFrame(c5,columns=["NumNumericChars==0_EDA"])], axis=1)


# In[47]:


new_features


# In[ ]:





# ## ***From Weight of Imbalanced Method***

# In[51]:


import scorecardpy as sc


# In[52]:


# we are using here following combined X_train and y_train
X_tr_y_tr


# In[53]:


y = 'CLASS_LABEL'


# In[54]:


bins = sc.woebin(X_tr_y_tr,y)


# In[55]:


bins


# In[56]:


sc.woebin_plot(bins)


# In[57]:


X_tr_y_tr.columns


# In[58]:


woe_data = bins['UrlLength'].append(bins['NumDash']).append(bins['NumDots']).append(bins['NumUnderscore']).append(bins['NumPercent']).append(bins['NumAmpersand']).append(bins['NumHash']).append(bins['NumDashInHostname']).append(bins['NumNumericChars']).append(bins['AtSymbol']).append(bins['TildeSymbol']).append(bins['NoHttps']).append(bins['IpAddress']).append(bins['SubdomainLevel']).append(bins['NumQueryComponents']).append(bins['HostnameLength']).append(bins['PathLength']).append(bins['QueryLength']).append(bins['DoubleSlashInPath'])


# In[59]:


woe_data


# Now we will create new features taking weights of each intervals as new feature.

# In[60]:


new_features['UrlLength_WOE'] = [-0.462714 if i<35 else 0.492540 if 35<=i<60 else 0.085523 if 60<=i<70 else -0.571443 if 70<=i<130 else 0.211513 for i in X_train['UrlLength']]
new_features['NumDash_WOE'] = [0.506946 if i<2 else -0.216327 if 2<=i<3 else -1.194891 if 3<=i<6 else -3.685051 for i in X_train['NumDash']]
new_features['NumDots_WOE'] = [-1.319003 if i<2 else -0.019816 if 2<=i<3 else 0.292073 if 3<=i<4 else 1.422297 for i in X_train['NumDots']]
new_features['NumUnderscore_WOE'] = [0.081354 if i<1 else -0.192783 if 1<=i<2 else -0.869469 for i in X_train['NumUnderscore']]
new_features['NumPercent_WOE'] = [0.00000 for i in X_train['NumPercent']]
new_features['NumAmpersand_WOE'] = [0.114997 if i<1 else -1.146679 for i in X_train['NumAmpersand']]
new_features['NumHash_WOE'] = [0.000000 for i in X_train['NumHash']]
new_features['NumDashInHostname_WOE'] = [-0.096186 if i<1 else 1.000892 for i in X_train['NumDashInHostname']]
new_features['NumNumericChars_WOE'] = [0.707812 if i<2 else -0.494543 if 2<=i<5 else -1.830517 if 5<=i<8 else -3.127076 if 8<=i<9 else -0.777151 if 9<=i<18 else 1.319388 for i in X_train['NumNumericChars']]
new_features['AtSymbol_WOE'] = [0.000000 for i in X_train['AtSymbol']]
new_features['TildeSymbol_WOE'] = [0.000000 for i in X_train['TildeSymbol']]
new_features['NoHttps_WOE'] = [0.000000 for i in X_train['NoHttps']]
new_features['IpAddress_WOE'] = [0.000000 for i in X_train['IpAddress']]
new_features['SubdomainLevel_WOE'] = [0.062771 if i<1 else -0.057979 for i in X_train['SubdomainLevel']]
new_features['NumQueryComponents_WOE'] = [0.223358 if i<1 else -0.992823 if 1<=i<2 else -1.166971 for i in X_train['NumQueryComponents']]
new_features['HostnameLength_WOE'] = [-0.705058 if i<11 else -0.120158 if 11<=i<20 else 0.080934 if 20<=i<26 else 0.484893 if 26<=i<30 else 1.202948 for i in X_train['HostnameLength']]
new_features['PathLength_WOE'] = [-0.873511 if i<8 else -0.203832 if 8<=i<16 else 0.402728 if 16<=i<46 else -0.117753 if 46<=i<60 else -0.833557 for i in X_train['PathLength']]
new_features['QueryLength_WOE'] = [0.222194 if i<6 else -1.813158 if 6<=i<30 else -0.491447 for i in X_train['QueryLength']]
new_features['DoubleSlashInPath_WOE'] = [0.000000 for i in X_train['DoubleSlashInPath']]


# In[61]:


new_features.head()


# ## ***Score from Isolation Forest***

# In[62]:


from sklearn.ensemble import IsolationForest

# we will use all 19 base features
x = X_train.values

clf = IsolationForest(contamination= 0.5, random_state=0)
predictions = clf.fit_predict(x)
print(predictions)
dec_fun = clf.decision_function(x)
dec_fun


# In[63]:


# dec_fun is numpy array, coverting it to dataframe to add with X_train_new
df = pd.DataFrame(dec_fun, columns = ["ISF_Score"])


# In[64]:


df


# In[65]:


new_features=pd.concat([new_features,df],axis=1)


# In[66]:


# All Newly Created Features
new_features


# **Here, till now all new features are created based on base 19 features only.** Lets add this **new_features** dataframe with **X_train** dataframe.

# In[67]:


## to avoid creating NaN values after concatenating
X_train.reset_index(drop=True,inplace=True)
new_features.reset_index(drop=True,inplace=True)


# In[68]:


X_train_new =pd.concat([X_train, new_features], axis=1)


# In[69]:


X_train_new


# # Feature Selection

# ## ***Variance Threshold***

# In[70]:


# import necessary library
from sklearn.feature_selection import VarianceThreshold

# we're setting variance threshold at 0
var_thres=VarianceThreshold(threshold=0)

# fitting this variance threshold in all columns of Newly Created  training data
var_thres.fit(X_train_new)


# In[71]:


# get the number of constant columns (whoose variance is 0)

constant_columns = [column for column in X_train_new.columns if column not in X_train_new.columns[var_thres.get_support()]]
print(len(constant_columns))


# In[72]:


# get boolean value 'False' for columns having constant nature
var_thres.get_support()


# In[73]:


# get the column names which remain in the Newly Created  training data
X_train_new.columns[var_thres.get_support()]


# In[74]:


# columns with constant nature
for feature in constant_columns:
    print(feature)


# In[75]:


# dropping constant columns
X_train_new.drop(constant_columns, axis=1, inplace=True)


# In[76]:


# all the columns in the training data
X_train_new.columns


# In[77]:


len(X_train_new.columns)


# **37 features are going for next Feature Selection Method.**

# ## ***Correlation Method***

# In[78]:


X_train_new.corr()


# In[79]:


# adjust the figure size
from matplotlib.pyplot import figure
figure(figsize=(10,10), dpi=80)

#better visual for correlation
sns.heatmap(X_train_new.corr())


# Clearly, we can see that some boxes have completely white and back colours that means correlation is Perfect Positive (in case of white coloured boxes) or Perfect Negative (in case of black coloured boxes). We can remove them.

# In[80]:


# function to get all features highly correlated (both positive and negative) with another for a given threshold

def correlation(dataset, threshold):
    col_corr = set()  #set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  #getting the name of the column
                col_corr.add(colname)
    return col_corr                        


# In[81]:


corr_features = correlation(X_train_new, 0.95)
len(set(corr_features))


# There are 1 feature which are highly correlated (can be explained by another feature) by amount of more than 95%. We will remove it to avoid complexity in model.

# In[82]:


# corresponding correlated features
corr_features


# In[83]:


# dropping highly correlated features
X_train_new.drop(corr_features, axis=1, inplace=True)


# In[84]:


# remaining columns
X_train_new.columns


# In[85]:


len(X_train_new.columns)


# **36 features are going for next criteria of feature selection.**

# ## ***Mutual Information***
# It internally uses KNN classifier. So I am using scaled version of X_train_new dataset.

# In[86]:


## Importing Libraries
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

## Copy data in another dataframe to retrieve latter
d=X_train_new.copy()


## Apply scaling
scaling = MinMaxScaler()
d_scaled=scaling.fit_transform(d)
d_scaled=pd.DataFrame(d_scaled)

## Calculate mutual information
mutual_info = mutual_info_classif(d_scaled, y_train, random_state=0)
mutual_info


# In[87]:


## Storing original data into X_train_new
X_train_new = d.copy()


# In[88]:


X_train_new.columns


# In[89]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train_new.columns
mutual_info.sort_values(ascending = False)


# In[90]:


# adjust the figure size
from matplotlib.pyplot import figure
figure(figsize=(10,10), dpi=80)
mutual_info.sort_values(ascending = False).plot.bar(figsize=(15,5))


# In[91]:


X_train_new.drop(["TildeSymbol","SubdomainLevel","DoubleSlashInPath"], axis=1, inplace=True)


# In[92]:


X_train_new.columns


# In[93]:


X_train_new.head()


# In[94]:


len(X_train_new.columns)


# Dropping "TildeSymbol","SubdomainLevel","DoubleSlashInPath" for having very low mutual information.

# ## ***ExtraTreesClassifier***
# It internally uses Decision Tree Classifier. So instead of using scaled features, we will use original variables. Because scaling leads to some amount of information loose.

# In[95]:


## Import Libraries
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# In[96]:


## Fit ExtraTreesClassifier with best set of parameters
model = ExtraTreesClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=3, criterion='gini',random_state=0)
model.fit(X_train_new,y_train)
print("model.feature_importances_:\n")
print(model.feature_importances_)
print("\n")
print("X_train_new.columns:\n")
print(X_train_new.columns)
ranked_features = pd.Series(model.feature_importances_, index = X_train_new.columns)

print("\n")
print("Visual Representation of Feature Importance from Model (Red line indicates value=0.005):")

## Visual Representaion
figure(figsize=(10,10), dpi=100)
ranked_features.sort_values().plot(kind='barh')
plt.axvline(x=0.005,c='r')
plt.show()


# Clearly "NoHttps","SubdomainLevel_GrtrEql_3_EDA", "NumHash" and "AtSymbol" have very less importance (setting cutoff at 0.005). So, we'll drop them.

# In[97]:


X_train_new.drop(["NoHttps","SubdomainLevel_GrtrEql_3_EDA", "NumHash", "AtSymbol" ] , axis=1, inplace=True)


# In[98]:


X_train_new.head()


# ## Final Training data

# In[99]:


# setting options to see all columns
pd.set_option("display.max_columns",None)


# In[100]:


X_train_new


# In[101]:


# finally training data has following columns
X_train_new.columns


# In[102]:


# resetting options 
pd.reset_option("display.max_columns")


# ## Preparing Test Data

# In[103]:


# setting options to see all columns, full column width
pd.set_option("display.max_columns",None)


# In[104]:


X_test.head()


# In[105]:


X_test.shape


# In[106]:


# resetting options 
pd.reset_option("display.max_columns")


# In[107]:


# We will work on X_test_new to keep base test dataset unchanged 
X_test_new = X_test.copy()


# In[108]:


X_test_new.head()


# In[109]:


X_test_new.shape


# ## Creating New Features

# In[110]:


X_train_new.columns


# In[111]:


X_test_new = X_test_new[['UrlLength', 'NumDash', 'NumDots', 'NumUnderscore', 'NumPercent',
       'NumAmpersand', 'NumDashInHostname', 'NumNumericChars', 'IpAddress',
       'NumQueryComponents', 'HostnameLength', 'PathLength', 'QueryLength']]


# In[112]:


X_test_new['NumDots_GrtrEql_4_EDA'] = np.where(X_test['NumDots']>=4,1,0)
X_test_new['NumDash_LsEql_1_EDA'] = np.where(X_test['NumDash']<=1,1,0)
X_test_new['NumQueryComponents==0_EDA'] = np.where(X_test['NumQueryComponents']==0,1,0)
X_test_new['NumNumericChars==0_EDA'] = np.where(X_test['NumNumericChars']==0,1,0)
X_test_new['UrlLength_WOE'] = [-0.462714 if i<35 else 0.492540 if 35<=i<60 else 0.085523 if 60<=i<70 else -0.571443 if 70<=i<130 else 0.211513 for i in X_test['UrlLength']]
X_test_new['NumDash_WOE'] = [0.506946 if i<2 else -0.216327 if 2<=i<3 else -1.194891 if 3<=i<6 else -3.685051 for i in X_test['NumDash']]
X_test_new['NumDots_WOE'] = [-1.319003 if i<2 else -0.019816 if 2<=i<3 else 0.292073 if 3<=i<4 else 1.422297 for i in X_test['NumDots']]
X_test_new['NumUnderscore_WOE'] = [0.081354 if i<1 else -0.192783 if 1<=i<2 else -0.869469 for i in X_test['NumUnderscore']]
X_test_new['NumAmpersand_WOE'] = [0.114997 if i<1 else -1.146679 for i in X_test['NumAmpersand']]
X_test_new['NumDashInHostname_WOE'] = [-0.096186 if i<1 else 1.000892 for i in X_test['NumDashInHostname']]
X_test_new['NumNumericChars_WOE'] = [0.707812 if i<2 else -0.494543 if 2<=i<5 else -1.830517 if 5<=i<8 else -3.127076 if 8<=i<9 else -0.777151 if 9<=i<18 else 1.319388 for i in X_test['NumNumericChars']]
X_test_new['SubdomainLevel_WOE'] = [0.062771 if i<1 else -0.057979 for i in X_test['SubdomainLevel']]
X_test_new['HostnameLength_WOE'] = [-0.705058 if i<11 else -0.120158 if 11<=i<20 else 0.080934 if 20<=i<26 else 0.484893 if 26<=i<30 else 1.202948 for i in X_test['HostnameLength']]
X_test_new['PathLength_WOE'] = [-0.873511 if i<8 else -0.203832 if 8<=i<16 else 0.402728 if 16<=i<46 else -0.117753 if 46<=i<60 else -0.833557 for i in X_test['PathLength']]
X_test_new['QueryLength_WOE'] = [0.222194 if i<6 else -1.813158 if 6<=i<30 else -0.491447 for i in X_test['QueryLength']]


# In[113]:


x = X_test.values
predictions=clf.predict(x)

df_test = pd.DataFrame(clf.decision_function(x), columns = ["ISF_Score"])
df_test


# In[114]:


clf.decision_function(x)


# In[116]:


## to avoid 'id' mismatch
X_test_new.reset_index(drop=True,inplace=True)
df_test.reset_index(drop=True,inplace=True)


# In[117]:


X_test_new=pd.concat([X_test_new,df_test],axis=1)


# In[212]:


X_validation.head()


# In[118]:


## to see all columns
pd.set_option("display.max_columns", None)


# In[119]:


X_test_new


# In[120]:


## resetting to default
pd.reset_option("display.max_columns")


# # Model Fit with Optimized Hyperparameter
# As this is **balanced dataset**, we will use **Repeated Randon Train-Test Split** every step.

# In[121]:


# import library for RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[184]:


# We'll also check cross validation score from training set
# Import library for Repeated Random Cross Validation
from sklearn.model_selection import ShuffleSplit
ssplit = ShuffleSplit(n_splits=10,test_size=0.30)


# In[ ]:


# Import library to calculate cross validation score
from sklearn.model_selection import cross_val_score


# ## Decision Tree Classifier

# In[122]:


# import library for DecisionTree
from sklearn.tree import DecisionTreeClassifier


# In[123]:


# make parameter Randomized for DecisionTree
params_DTC = {'max_leaf_nodes':list(range(2, 100)), 'min_samples_split':[2,3,4], 'criterion': ['entropy','gini']}
print(params_DTC)

# fitting Randomized_search on DecisionTree and getting the best hyperparameters
Randomized_search_DTC = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), params_DTC, verbose=1, cv=5)

best_params_DTC = Randomized_search_DTC.fit(X_train_new, y_train)

print(best_params_DTC.best_params_)

print(best_params_DTC.best_estimator_)


# In[188]:


predictions_training_DTC = Randomized_search_DTC.predict(X_train_new)
score=cross_val_score(DecisionTreeClassifier(max_leaf_nodes=93, min_samples_split=4, random_state=0),X_train_new, y_train, cv=ssplit)
predictions_training_DTC_CV=score.mean()
predictions_testing_DTC = Randomized_search_DTC.predict(X_test_new)

print("Training accuracy of DecisionTree: ", format(accuracy_score(y_train,predictions_training_DTC),"0.2f"))
print("Training accuracy of DecisionTree after Cross Validation: ", format(predictions_training_DTC_CV,"0.2f"))
print("Testing accuracy of DecisionTree: ", format(accuracy_score(y_test,predictions_testing_DTC),"0.2f"))


# In[124]:


## Confusion Matrix
cm_DTC = confusion_matrix(y_test, predictions_testing_DTC)
print("Confusion Matrix : \n", cm_DTC)

## Classification Report
print(classification_report(y_test, predictions_testing_DTC))

## probability of falling into a particular class from each observation
probs_DTC = best_params_DTC.predict_proba(X_test_new)
probs_DTC = probs_DTC[:,1]

## AUC (Area Under Curve) Score
auc_score_DTC = roc_auc_score(y_test, probs_DTC)
print("Decision Tree Model ROC AUC = ",format(auc_score_DTC,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')

## ROC Curve of Decision Tree
fpr_DTC, tpr_DTC, __DTC = roc_curve(y_test, probs_DTC)

plt.plot(fpr_DTC, tpr_DTC, linestyle='--', label='Decision Tree')
plt.title("ROC Curve for Decision Tree Classifier")
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # Random Forest Classifier

# In[125]:


# import library for RandomForest
from sklearn.ensemble import RandomForestClassifier


# In[126]:


# make parameter Randomized for DecisionTree
params_RFC = {
    'criterion': ['entropy','gini'],
    
    'max_depth': [int(x) for x in np.linspace(10,100,5)],
    
    'max_features': ['auto','sqrt','log2'],
    
    'min_samples_leaf': [2,5,8],
    
    'min_samples_split': [2,5,10],
    
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=500, num=5)]
}
print(params_RFC)

# fitting Randomized_search on RandomForestClassifier and getting the best hyperparameters
Randomized_search_RFC = RandomizedSearchCV(RandomForestClassifier(random_state=0), params_RFC, verbose=1, cv=5)

best_params_RFC = Randomized_search_RFC.fit(X_train_new, y_train)

print(best_params_RFC.best_params_)

print(best_params_RFC.best_estimator_)


# In[189]:


predictions_training_RFC = Randomized_search_RFC.predict(X_train_new)
score=cross_val_score(RandomForestClassifier(max_depth=32, max_features='sqrt', min_samples_leaf=2,
                       min_samples_split=5, n_estimators=300, random_state=0),X_train_new, y_train, cv=ssplit)
predictions_training_RFC_CV=score.mean()
predictions_testing_RFC = Randomized_search_RFC.predict(X_test_new)

print("Training accuracy of Random Forest Classifier: ", format(accuracy_score(y_train,predictions_training_RFC),"0.2f"))
print("Training accuracy of Random Forest Classifier after Cross Validation: ", format(predictions_training_RFC_CV,"0.2f"))
print("Testing accuracy of Random Forest Classifier: ", format(accuracy_score(y_test,predictions_testing_RFC),"0.2f"))


# In[127]:


## Confusion Matrix
cm_RFC = confusion_matrix(y_test, predictions_testing_RFC)
print("Confusion Matrix : \n", cm_RFC)

## Classification Report
print(classification_report(y_test, predictions_testing_RFC))

## probability of falling into a particular class from each observation
probs_RFC = best_params_RFC.predict_proba(X_test_new)
probs_RFC = probs_RFC[:,1]

## AUC (Area Under Curve) Score
auc_score_RFC = roc_auc_score(y_test, probs_RFC)
print("Random Forest Classifier ROC AUC = ",format(auc_score_RFC,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')


## ROC Curve from Random Forest
fpr_RFC, tpr_RFC, __RFC = roc_curve(y_test, probs_RFC)
plt.plot(fpr_RFC, tpr_RFC, linestyle='--', label='Random Forest')
plt.title("ROC Curve for Random Forest Classifier")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # SVM

# In[128]:


# import library for DecisionTree
from sklearn.svm import SVC


# In[129]:


## we have to apply scaled data here
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()

print("Scaled Training Data:\n")
## Scaling for Training Data
X_train_new_scaled = scaling.fit_transform(X_train_new)
X_train_new_scaled = pd.DataFrame(X_train_new_scaled, columns=X_train_new.columns)
print(X_train_new_scaled)

print("\n")

print("Scaled Testing Data:\n")
## Scaling for Testing Data
X_test_new_scaled = scaling.transform(X_test_new)
X_test_new_scaled = pd.DataFrame(X_test_new_scaled, columns=X_test_new.columns)
print(X_test_new_scaled)


# In[130]:


# make parameter Randomized for Support Vector Machine
params_SVM = {
    'C':[0.1, 1, 10, 100, 1000],
    'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
    'kernel':['rbf']
}
print(params_SVM)

# fitting Randomized_search on RandomizedForestClassifier and getting the best hyperparameters
Randomized_search_SVM = RandomizedSearchCV(SVC(probability=True, random_state=0), params_SVM, verbose=1, cv=5)

best_params_SVM = Randomized_search_SVM.fit(X_train_new_scaled, y_train)

print(best_params_SVM.best_params_)

print(best_params_SVM.best_estimator_)


# In[191]:


predictions_training_SVM = Randomized_search_SVM.predict(X_train_new_scaled)
score=cross_val_score(SVC(C=1000, gamma=0.1, probability=True, random_state=0),X_train_new, y_train, cv=ssplit)
predictions_training_SVM_CV=score.mean()     
predictions_testing_SVM = Randomized_search_SVM.predict(X_test_new_scaled)


print("Training accuracy of Support Vector Machine: ", format(accuracy_score(y_train,predictions_training_SVM),"0.2f"))
print("Training accuracy of Support Vector Machine after Cross Validation: ", format(predictions_training_SVM_CV,"0.2f"))
print("Testing accuracy of Support Vector Machine: ", format(accuracy_score(y_test,predictions_testing_SVM),"0.2f"))


# In[131]:


## Confusion Matrix
cm_SVM = confusion_matrix(y_test, predictions_testing_SVM)
print("Confusion Matrix : \n", cm_SVM)

## Classification Report
print(classification_report(y_test, predictions_testing_SVM))

## probability of falling into a particular class from each observation
probs_SVM = best_params_SVM.predict_proba(X_test_new)
probs_SVM = probs_SVM[:,1]

## AUC (Area Under Curve) Score
auc_score_SVM = roc_auc_score(y_test, probs_SVM)
print("Support Vector Machine ROC AUC = ",format(auc_score_SVM,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')

## ROC Curve of Support Vector Machine
fpr_SVM, tpr_SVM, __SVM = roc_curve(y_test, probs_SVM)
plt.plot(fpr_SVM, tpr_SVM, linestyle='--', label='Support Vector Machine')
plt.title("ROC Curve for Support Vector Machine")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # XgBoost

# In[140]:


# import library for XgBoost
from xgboost import XGBClassifier


# In[141]:


# make parameter Randomized for XgBoost
params_XGB = {
    
    'max_depth': [i for i in range(2,10,1)],
    'n_estimators': [i for i in range(60,220,40)],
    'learning_rate':[0.1,0.01,0.05]
}
print(params_XGB)

# fitting Randomized_search on RandomForestClassifier and getting the best hyperparameters
Randomized_search_XGB = RandomizedSearchCV(XGBClassifier(objective = 'binary:logistic', nthread=4, use_label_encoder=False, seed=42), params_XGB, verbose=1, cv=5)

best_params_XGB = Randomized_search_XGB.fit(X_train_new, y_train)

print(best_params_XGB.best_params_)

print(best_params_XGB.best_estimator_)


# In[200]:


predictions_training_XGB = Randomized_search_XGB.predict(X_train_new)
score=cross_val_score(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=140, n_jobs=4, nthread=4, num_parallel_tree=1,
              random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              seed=42, subsample=1, tree_method='exact',
              use_label_encoder=False, validate_parameters=1, verbosity=None),X_train_new, y_train, cv=ssplit)
predictions_training_XGB_CV=score.mean()      
predictions_testing_XGB = Randomized_search_XGB.predict(X_test_new)

print("Training accuracy of XgBoost: ", format(accuracy_score(y_train,predictions_training_XGB),"0.2f"))
print("Training accuracy of XgBoost after Cross Validation: ", format(predictions_training_XGB_CV,"0.2f"))
print("Testing accuracy of XgBoost: ", format(accuracy_score(y_test,predictions_testing_XGB),"0.2f"))


# In[142]:


## Confusion Matrix
cm_XGB = confusion_matrix(y_test, predictions_testing_XGB)
print("Confusion Matrix : \n", cm_XGB)

## Classification Report
print(classification_report(y_test, predictions_testing_XGB))

## probability of falling into a particular class from each observation
probs_XGB = best_params_XGB.predict_proba(X_test_new)
probs_XGB = probs_XGB[:,1]

## AUC (Area Under Curve) Score
auc_score_XGB = roc_auc_score(y_test, probs_XGB)
print("XgBoost Model ROC AUC = ",format(auc_score_XGB,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')


## ROC Curve of XgBoost Model
fpr_XGB, tpr_XGB, __XGB = roc_curve(y_test, probs_XGB)
plt.plot(fpr_XGB, tpr_XGB, linestyle='--', label='XgBoost')
plt.title("ROC Curve for XgBoost Model")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # lightgbm

# In[143]:


# import library for lightgbm
import lightgbm as lgb


# In[144]:


# make parameter Randomized for lightgbm
params_LGB = {
    'num_leaves': [31, 127],
    
    'reg_alpha': [0.1, 0.5],
    
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    
    'lambda_l1': [0, 1, 1.5],
    
    'lambda_l2': [0, 1]
}
print(params_LGB)

# fitting Randomized_search on RandomForestClassifier and getting the best hyperparameters
Randomized_search_LGB = RandomizedSearchCV(lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',num_boost_round=2000,learning_rate=0.01,random_state=0), params_LGB, verbose=1, cv=5)

best_params_LGB = Randomized_search_LGB.fit(X_train_new, y_train)

print(best_params_LGB.best_params_)

print(best_params_LGB.best_estimator_)


# In[199]:


predictions_training_LGB = Randomized_search_LGB.predict(X_train_new)
score=cross_val_score(lgb.LGBMClassifier(lambda_l1=0, lambda_l2=0, learning_rate=0.01,
               min_data_in_leaf=50, num_boost_round=2000, objective='binary',
               random_state=0, reg_alpha=0.1),X_train_new, y_train, cv=5)
predictions_training_LGB_CV=score.mean()
predictions_testing_LGB = Randomized_search_LGB.predict(X_test_new)

print("Training accuracy of lightgbm: ", format(accuracy_score(y_train,predictions_training_LGB),"0.2f"))
print("Training accuracy of lightgbm after Cross Validation: ", format(predictions_training_LGB_CV,"0.2f"))
print("Testing accuracy of lightgbm: ", format(accuracy_score(y_test,predictions_testing_LGB),"0.2f"))


# In[146]:


## Confusion Matrix
cm_LGB = confusion_matrix(y_test, predictions_testing_LGB)
print("Confusion Matrix : \n", cm_LGB)

## Classification Report
print(classification_report(y_test, predictions_testing_LGB))

## probability of falling into a particular class from each observation
probs_LGB = best_params_LGB.predict_proba(X_test_new)
probs_LGB = probs_LGB[:,1]

## AUC (Area Under Curve) Score
auc_score_LGB = roc_auc_score(y_test, probs_LGB)
print("LightGBM Model ROC AUC = %.3f",format(auc_score_LGB,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')


## ROC Curve
fpr_LGB, tpr_LGB, __LGB = roc_curve(y_test, probs_LGB)
plt.plot(fpr_LGB, tpr_LGB, linestyle='--', label='ROC Curve of LightGBM Model')
plt.title("ROC Curve of LightGBM Model")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # catboost

# In[147]:


# import library for CatBoost
import catboost as cb


# In[148]:


# make parameter Randomized for CatBoost
params_CTB = {
    'depth':[3,1,2,6,4,5,7,8,9,10],
    'iterations':[250,100,500,1000],
    'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
    'l2_leaf_reg':[3,1,5,10,100],
    'border_count':[32,5,10,20,100,200],
    'thread_count':[4]
}
print(params_CTB)

# fitting Randomized_search on RandomForestClassifier and getting the best hyperparameters
Randomized_search_CTB = RandomizedSearchCV(cb.CatBoostClassifier(random_state=0), params_CTB, verbose=1, cv=5)

best_params_CTB = Randomized_search_CTB.fit(X_train_new, y_train)

print(best_params_CTB.best_params_)

print(best_params_CTB.best_estimator_)


# In[201]:


predictions_training_CTB = Randomized_search_CTB.predict(X_train_new)
score=cross_val_score(cb.CatBoostClassifier(thread_count= 4, learning_rate= 0.03, l2_leaf_reg= 1, iterations= 100, depth= 10, border_count= 100,random_state=0),X_train_new, y_train, cv=5)
predictions_training_CTB_CV=score.mean()
predictions_testing_CTB = Randomized_search_CTB.predict(X_test_new)

print("Training accuracy of CatBoost: ", format(accuracy_score(y_train,predictions_training_CTB),"0.2f"))
print("Training accuracy of CatBoost after Cross Validation: ", format(predictions_training_CTB_CV,"0.2f"))
print("Testing accuracy of CatBoost: ", format(accuracy_score(y_test,predictions_testing_CTB),"0.2f"))


# In[149]:


## Confusion Matrix
cm_CTB = confusion_matrix(y_test, predictions_testing_CTB)
print("Confusion Matrix : \n", cm_CTB)

## Classification Report
print(classification_report(y_test, predictions_testing_CTB))

## probability of falling into a particular class from each observation
probs_CTB = best_params_CTB.predict_proba(X_test_new)
probs_CTB = probs_CTB[:,1]

## AUC (Area Under Curve) Score
auc_score_CTB = roc_auc_score(y_test, probs_CTB)
print("CatBoost Model ROC AUC = ",format(auc_score_CTB,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')

## ROC Curve
fpr_CTB, tpr_CTB, __CTB = roc_curve(y_test, probs_CTB)
plt.plot(fpr_CTB, tpr_CTB, linestyle='--', label='ROC Curve of CatBoost Model')
plt.title("ROC Curve of CatBoost Model")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# # Voting Classifier 

# In[150]:


# import library
from sklearn.ensemble import VotingClassifier


# In[156]:


# group/ensemble of models
## we'll not include Support Vector Machine as ROC curve from it has 50% area as AUC score

estimator = []
estimator.append(('LR', LogisticRegression()))  # from baseline model
estimator.append(('DTC', DecisionTreeClassifier(max_leaf_nodes=96, random_state=0)))

estimator.append(('RFC', RandomForestClassifier(max_depth=100, min_samples_leaf=2, min_samples_split=5,
                       n_estimators=400, random_state=0)))

estimator.append(('XGB', XGBClassifier(n_estimators= 180, max_depth= 6, learning_rate= 0.1, random_state=0)))


estimator.append(('LGB', lgb.LGBMClassifier(lambda_l1=1, lambda_l2=1, learning_rate=0.01,
               min_data_in_leaf=50, num_boost_round=2000, num_leaves=127,
               objective='binary', random_state=0, reg_alpha=0.5)))

estimator.append(('CTB', cb.CatBoostClassifier(thread_count= 4, learning_rate= 0.03, l2_leaf_reg= 1, iterations= 1000, depth= 7, border_count= 200, random_state=0)))


# In[157]:


estimator


# ## ***Soft Voting***

# In[226]:


# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators = estimator, voting = 'soft', verbose=1)
vot_soft.fit(X_train_new, y_train)

y_pred_train_soft = vot_soft.predict(X_train_new)
y_pred_test_soft = vot_soft.predict(X_test_new)


# In[227]:


# using accuracy_score metric to predict accuracy
training_accuracy_soft = accuracy_score(y_train, y_pred_train_soft)
score=cross_val_score(VotingClassifier(estimators = estimator, voting = 'soft', verbose=1),X_train_new, y_train, cv=5)
training_accuracy_soft_CV=score.mean()
testing_accuracy_soft = accuracy_score(y_test, y_pred_test_soft)

print("Training accuracy after applying Voting(soft) Classifier: ", format(training_accuracy_soft,"0.2f"))
print("Training accuracy of Voting (Soft) Classifier after Cross Validation: ", format(training_accuracy_soft_CV,"0.2f"))
print("Testing accuracy after applying Voting(soft) Classifier: ", format(testing_accuracy_soft,"0.2f"))


# In[161]:


## Confusion Matrix
cm_VOT_soft = confusion_matrix(y_test, y_pred_test_soft )
print("Confusion Matrix : \n", cm_VOT_soft)

## Classification Report
print(classification_report(y_test, y_pred_test_soft))

## probability of falling into a particular class from each observation
probs_VOT_soft = vot_soft.predict_proba(X_test_new_scaled)
probs_VOT_soft = probs_VOT_soft[:,1]

## AUC (Area Under Curve) Score
auc_score_VOT_soft = roc_auc_score(y_test, probs_VOT_soft)
print("Voting (Soft) Classifier ROC AUC = %.3f",format(auc_score_VOT_soft,"0.2f"))

## no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill Model')

## ROC Curve
fpr_VOT_soft, tpr_VOT_soft, __VOT_soft = roc_curve(y_test, probs_VOT_soft)
plt.plot(fpr_VOT_soft, tpr_VOT_soft, linestyle='--', label='Voting (soft) Classifier')
plt.title("ROC Curve of Voting (Soft) Classifier Model")

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show legend
plt.legend()
# show the plot
plt.show()


# In[202]:


combined_result = {'Model_Name':['Logistic (Baseline)','Decision Tree','Random Forest','Support Vector Machine','XgBoost','LightGBM','CatBoost','Voting(Soft)'],
                   'Training_Accuracy':[format(accuracy_score(y_train,predictions_training),"0.2f"), format(accuracy_score(y_train,predictions_training_DTC),"0.2f"), format(accuracy_score(y_train,predictions_training_RFC),"0.2f"), format(accuracy_score(y_train,predictions_training_SVM),"0.2f"), format(accuracy_score(y_train,predictions_training_XGB),"0.2f"),format(accuracy_score(y_train,predictions_training_LGB),"0.2f"),format(accuracy_score(y_train,predictions_training_CTB),"0.2f"), format(training_accuracy_soft,"0.2f")],
                   'Training_Accuracy_from_CV':[format(predictions_training_CV,"0.2f"),format(predictions_training_DTC_CV,"0.2f"),format(predictions_training_RFC_CV,"0.2f"),format(predictions_training_SVM_CV,"0.2f"),format(predictions_training_XGB_CV,"0.2f"),format(predictions_training_LGB_CV,"0.2f"),format(predictions_training_CTB_CV,"0.2f"),format(training_accuracy_soft_CV,"0.2f")],
                   'Testing_Accuracy':[format(accuracy_score(y_test,predictions_testing),"0.2f"), format(accuracy_score(y_test,predictions_testing_DTC),"0.2f"), format(accuracy_score(y_test,predictions_testing_RFC),"0.2f"), format(accuracy_score(y_test,predictions_testing_SVM),"0.2f"), format(accuracy_score(y_test,predictions_testing_XGB),"0.2f"),format(accuracy_score(y_test,predictions_testing_LGB),"0.2f"),format(accuracy_score(y_test,predictions_testing_CTB),"0.2f"), format(testing_accuracy_soft,"0.2f")],
                   'ROC_AUC_Score':[format(auc_score,"0.2f"),format(auc_score_DTC,"0.2f"),format(auc_score_RFC,"0.2f"),format(auc_score_SVM,"0.2f"),format(auc_score_XGB,"0.2f"),format(auc_score_LGB,"0.2f"),format(auc_score_CTB,"0.2f"),format(auc_score_VOT_soft,"0.2f")]
            
                  }


# In[203]:


combined_result = pd.DataFrame.from_dict(combined_result)


# In[204]:


combined_result


# No overfitting or underfitting issue.

# ### We will use Soft Voting Classifier as Final Model, as it will give us average probability for a set of features to predict 

# # Check Model on Validation Data

# In[220]:


X_validation=data_url.copy()
X_validation=X_validation.drop(['url','result'],axis=1)
X_validation_new=X_validation.copy()

X_validation_new = X_validation_new[['UrlLength', 'NumDash', 'NumDots', 'NumUnderscore', 'NumPercent',
       'NumAmpersand', 'NumDashInHostname', 'NumNumericChars', 'IpAddress',
       'NumQueryComponents', 'HostnameLength', 'PathLength', 'QueryLength']]

X_validation_new['NumDots_GrtrEql_4_EDA'] = np.where(X_validation['NumDots']>=4,1,0)
X_validation_new['NumDash_LsEql_1_EDA'] = np.where(X_validation['NumDash']<=1,1,0)
X_validation_new['NumQueryComponents==0_EDA'] = np.where(X_validation['NumQueryComponents']==0,1,0)
X_validation_new['NumNumericChars==0_EDA'] = np.where(X_validation['NumNumericChars']==0,1,0)
X_validation_new['UrlLength_WOE'] = [-0.462714 if i<35 else 0.492540 if 35<=i<60 else 0.085523 if 60<=i<70 else -0.571443 if 70<=i<130 else 0.211513 for i in X_validation['UrlLength']]
X_validation_new['NumDash_WOE'] = [0.506946 if i<2 else -0.216327 if 2<=i<3 else -1.194891 if 3<=i<6 else -3.685051 for i in X_validation['NumDash']]
X_validation_new['NumDots_WOE'] = [-1.319003 if i<2 else -0.019816 if 2<=i<3 else 0.292073 if 3<=i<4 else 1.422297 for i in X_validation['NumDots']]
X_validation_new['NumUnderscore_WOE'] = [0.081354 if i<1 else -0.192783 if 1<=i<2 else -0.869469 for i in X_validation['NumUnderscore']]
X_validation_new['NumAmpersand_WOE'] = [0.114997 if i<1 else -1.146679 for i in X_validation['NumAmpersand']]
X_validation_new['NumDashInHostname_WOE'] = [-0.096186 if i<1 else 1.000892 for i in X_validation['NumDashInHostname']]
X_validation_new['NumNumericChars_WOE'] = [0.707812 if i<2 else -0.494543 if 2<=i<5 else -1.830517 if 5<=i<8 else -3.127076 if 8<=i<9 else -0.777151 if 9<=i<18 else 1.319388 for i in X_validation['NumNumericChars']]
X_validation_new['SubdomainLevel_WOE'] = [0.062771 if i<1 else -0.057979 for i in X_validation['SubdomainLevel']]
X_validation_new['HostnameLength_WOE'] = [-0.705058 if i<11 else -0.120158 if 11<=i<20 else 0.080934 if 20<=i<26 else 0.484893 if 26<=i<30 else 1.202948 for i in X_validation['HostnameLength']]
X_validation_new['PathLength_WOE'] = [-0.873511 if i<8 else -0.203832 if 8<=i<16 else 0.402728 if 16<=i<46 else -0.117753 if 46<=i<60 else -0.833557 for i in X_validation['PathLength']]
X_validation_new['QueryLength_WOE'] = [0.222194 if i<6 else -1.813158 if 6<=i<30 else -0.491447 for i in X_validation['QueryLength']]

x = X_validation.values
predictions=clf.predict(x)
df_validation = pd.DataFrame(clf.decision_function(x), columns = ["ISF_Score"])

## to avoid 'id' mismatch
X_validation_new.reset_index(drop=True,inplace=True)
df_validation.reset_index(drop=True,inplace=True)
X_validation_new=pd.concat([X_validation_new,df_validation],axis=1)
X_validation_new.head()


# In[229]:


y_pred_validation_soft = vot_soft.predict(X_validation_new)
validation_accuracy_soft = accuracy_score(data_url['result'], y_pred_validation_soft)


# In[230]:


print("Accuracy of Validation set from Voting(soft) Classifier: ", format(validation_accuracy_soft,"0.2f"))


# # Saving Model and Isolation Score Calculating Function as Pickle Files

# In[205]:


import pickle


# In[206]:


## Save the trained model as pickle string
with open('Phishing_Model.pickle','wb') as f:
    pickle.dump(vot_soft,f)


# In[207]:


## Save Isolation Score Calculating function into pickle file
with open('Isolation_Forest_Score.pickle','wb') as file:
    pickle.dump(clf,file)


# # Creating requirements.txt

# In[ ]:


#pip freeze>requirements.txt

