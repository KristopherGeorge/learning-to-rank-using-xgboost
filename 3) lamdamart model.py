import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import time


train_df = pd.read_pickle('MSLR-WEB10K/Fold1/train.pkl')
test_df = pd.read_pickle('MSLR-WEB10K/Fold1/test.pkl')

train_df = train_df.sort_values(1)
train_df.head()
train_grp = train_df[1]
train_truth = train_df[0]
train_X = train_df.loc[:,2:]
train_X.columns = ['x__' + str(x) for x in range(1, 137, 1)]
train_X = train_X.apply(pd.to_numeric)
training_data = xgb.DMatrix(train_X,
                            label = train_truth)
train_grp = train_grp.map(pd.to_numeric)
training_data.set_group(list(train_grp))
print('done')

group_info = {}
for i in train_grp:        
    if i in group_info.keys():
        group_info[i] = group_info[i] + 1
    else:
        group_info[i] = 1
group_info       
gid = [group_info[key] for key in group_info]
gid 

training_data.set_group(gid)
print('done')

def get_DMatrix(df):
    df = df.sort_values(1)
    grp = df[1]

    # I don't know best way to do this
    # need to keep the ordering of the data
    # that the group info comes in unchanged
    # I'm using this approach because I know
    # it will do this
    grp_info = {}
    for i in grp:        
        if i in grp_info.keys():
            grp_info[i] = grp_info[i] + 1
        else:
            grp_info[i] = 1       
    grp_info_cnts = [grp_info[key] for key in grp_info]
 
    y = df[0]
    X = df.loc[:,2:]
    X.columns = ['x__' + str(x) for x in range(1, 137, 1)]
    X = X.apply(pd.to_numeric)
    df_DMatrix = xgb.DMatrix(X,
                             label = y)
    grp = grp.map(pd.to_numeric)
    df_DMatrix.set_group(grp_info_cnts)
    print(f'nrows group = {len(grp)}')
    print(f'nrows Y label = {len(grp)}')
    print(f'shape of X = {X.shape}')
    return df_DMatrix


train_data = get_DMatrix(train_df)
test_data = get_DMatrix(test_df)


print('done')

params_lm2 = [('objective','rank:ndcg'),
              ('max_depth',2), 
              ('eta',0.1), 
              ('num_boost_round',4), 
              ('seed',404)]

start_lm2 = time.time()           
model_lm2 = xgb.train(params_lm2, 
                      train_data)
end_lm2 = time.time()
print(end_lm2-start_lm2)


start_lm2p = time.time()
pred_lm2 = model_lm2.predict(test_data)
end_lm2p = time.time()
print(end_lm2p-start_lm2p)



params_lm6 = [('objective','rank:ndcg'),
              ('max_depth',6), 
              ('eta',0.1), 
              ('num_boost_round',4), 
              ('seed',404)]

start_lm6 = time.time()           
model_lm6 = xgb.train(params_lm6, train_data)
end_lm6 = time.time()
print(end_lm6-start_lm6)


start_lm6p = time.time()
pred_lm6 = model_lm6.predict(test_data)
end_lm6p = time.time()
print(end_lm6p-start_lm6p)



pd.DataFrame(pred_lm6).to_csv("MSLR-WEB10K/Fold1/pred_lm6.txt", 
                              header=None, 
                              sep=" ")
pd.DataFrame(pred_lm2).to_csv("MSLR-WEB10K/Fold1/pred_lm2.txt", 
                             header=None, 
                             sep=" ")

params = [('objective','rank:ndcg'),
          ('max_depth',2), 
          ('eta',0.1), 
          ('num_boost_round',4)]


model = xgb.train(params, train_data)
predictions = model.predict(test_data)                             







##### now for lamdaRank models
##### now for lamdaRank models
##### now for lamdaRank models
##### now for lamdaRank models
##### now for lamdaRank models

params_lr2 = [('objective','rank:pairwise'),
              ('max_depth',2), 
              ('eta',0.1), 
              ('num_boost_round',4), 
              ('seed',404)]

start_lr2 = time.time()           
model_lr2 = xgb.train(params_lr2, train_data)
end_lr2 = time.time()
print(end_lr2-start_lr2)


start_lr2p = time.time()
pred_lr2 = model_lr2.predict(test_data)
end_lr2p = time.time()
print(end_lr2p-start_lr2p)


params_lr6 = [('objective','rank:pairwise'),
              ('max_depth',6), 
              ('eta',0.1), 
              ('num_boost_round',4), 
              ('seed',404)]

start_lr6 = time.time()           
model_lr6 = xgb.train(params_lr6, 
                      train_data)
end_lr6 = time.time()
print(end_lr6-start_lr6)

start_lr6p = time.time()
pred_lr6 = model_lr6.predict(test_data)
end_lr6p = time.time()
print(end_lr6p-start_lr6p)


pd.DataFrame(pred_lr6).to_csv("MSLR-WEB10K/Fold1/pred_lr6.txt", 
                              header = None, 
                              sep = " ")
pd.DataFrame(pred_lr2).to_csv("MSLR-WEB10K/Fold1/pred_lr2.txt", 
                              header = None, 
                              sep = " ")

