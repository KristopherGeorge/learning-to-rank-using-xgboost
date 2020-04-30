import pandas as pd
import os
import numpy as np


os.getcwd()

train = pd.read_csv("MSLR-WEB10K/Fold1/train.txt", 
                    header=None, 
                    sep=" ")

test = pd.read_csv("MSLR-WEB10K/Fold1/test.txt", 
                   header=None, 
                   sep=" ")


print('done')
train.head()


# let's inspect the first rows
np.random.seed(0xc00f33e)
df_samp = train.sample(4)


df_samp

# you can see column 1 represents the query id!
# and the the row will represent a document for it
# and then column 0 represents the relevancy of the document
# for the query!
# so the relevancy scores
train[0].value_counts()

# columns 2 till the end represent the features and have
# a feature  name: feature value dictionary style!
# so we need to 

# extract qid which is positions 4 onwards! i.e.
[x[4:] for x in df_samp[1]]

# and now for a feature
[x.split(':') for x in df_samp[2]]
[x.split(':')[1] for x in df_samp[2]]

def extract_qid(qid_str):
    return qid_str[4:]

def extract_val(feat):
    return feat.split(':')[1]


# reminder - map: works only on a pandas series
from sklearn.datasets import load_iris
data = load_iris()
data = pd.DataFrame(data['data'], 
                    columns = data['feature_names'])
data.head()                    

def cm_to_mm(var):
    return var * 10

# will error out
data.map(cm_to_mm)
data[['sepal length (cm)']].map(cm_to_mm)
# map works only on a pandas series! not on a pandas data frame!
data['sepal length (cm)'].map(cm_to_mm)


# introduce apply! works on the entire data-frame!
# applies function to each column!
data.apply(cm_to_mm)


### now Applymap -> works at the column + element level!!!!
### that's the difference between apply -> which works at the column level!



# so clean up the qid! could use map or apply here
df_samp[1].apply(extract_qid)
df_samp[1].map(extract_qid)


df_samp[1] = df_samp[1].apply(extract_qid)


cols = df_samp.columns[2:-1]
# the below will fail because split only works at the element
# level! hence why we must use apply and then map!!!!!
df_samp[cols].apply(extract_val)
df_samp[cols] = df_samp[cols].applymap(extract_val)


#'138' is NaN for every row - so we'll drop it!
df_samp[138]



def df_transform(df):
    df[1] = df[1].apply(extract_qid)
    df[df.columns[2:-1]] = df[df.columns[2:-1]].applymap(extract_val)
    df = df.drop(138, axis=1)
    return df.sort_values([1, 0]).reset_index(drop = True)
    
train_df = df_transform(train)

train_df.head()

sum(train_df[1] == '1')
train_df[0].value_counts()
train_df[1].value_counts().sort_values().head(30)

train_df[train_df[1] == '13429']

# i.e. here is a query 13429, and we have 9 documents for it
# and only 1 has a relevancy score of 2 whilst the rest are really
# terible with a relavancy score of 0!

test_df = df_transform(test)

####
test_df[0].value_counts()
test_df[1].value_counts().sort_values().head(20)
test_df[test_df[1] == '20728']

# you can see how our test data set is! for each query
# we include all documents that we're considering for it!
# and now we rank these documents! note that to train this
# data you need to have the relavancy for each document...
# this is possible to manufacture!

# so now you have Y label = column 0, query / group id = column 1
# and the rest are features!

# you want to save pickle versions of these data sets to be 
# able to load them quickly


train_df.to_pickle('MSLR-WEB10K/Fold1/train.pkl')
test_df.to_pickle('MSLR-WEB10K/Fold1/test.pkl')

#### let's set up for a linear regression model


X_test = test_df[test_df.columns[2:]]
y_test = test_df[0]

X = train_df[train_df.columns[2:]]
y = train_df[0]

import sklearn
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y)
reg.score(X,y) # gives the R ^ 2 of model! so this is a shit model!

# so I get the fit function...what does the score give you?

preds = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(preds, y_test)




### plot the results -> don't know how to plot things in
### crappy python, and don't care to be honest for the time
### being going to excel at doing things in R!

plot_data = pd.DataFrame(data= {"true score": y_test, 
                                "predicted score": preds})


plot_data.sample(10)


import altair as alt
alt.renderers.enable('notebook')

alt.Chart(plot_data.sample(1000)).encode(x="true score", y="predicted score").mark_point().interactive()
# predicted scores are continous while the true scores
# are discrete -> hence why linear regression is itself a fraught 
# method to begin with to handle this problem
# it's a ordinal regression 
# but could be modelled as a multinomial problem!


