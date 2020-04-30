import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train_df = pd.read_pickle('MSLR-WEB10K/Fold1/train.pkl')
test_df = pd.read_pickle('MSLR-WEB10K/Fold1/test.pkl')


# get X vars
X = train_df[train_df.columns[2:]]
# get Y var - remember column 1 is the grouping var
y = train_df[0]
reg = LinearRegression().fit(X, y)

X_test = test_df[test_df.columns[2:]]
y_test = test_df[0]

preds = reg.predict(X_test)


### simple mean squared error
from sklearn.metrics import mean_squared_error
mean_squared_error(preds, y_test)

# pairwise difference -> the number of swaps
# required to make to ensure the predicted list 
# matches the true ordering!

# this is done at a query level!

# use the bubble sort algorithm!
preds_df = pd.DataFrame({'qid': test_df[1], 
                         'truth' :test_df[0], 
                         'predicted' : preds })

preds_df.qid.value_counts().sort_values().head(20)

preds_df.qid.value_counts().sort_values().head(20)
preds_df.head()
# 26668    
preds_df[preds_df.qid == '22468']

preds_df.query('truth > 0').qid.value_counts().sort_values().head(100)

preds_df[preds_df.qid == '13918']
preds_df[preds_df.qid == '13918'].truth.value_counts()

# wow so of the documents only one was relevant! so you can see that you only need

preds_df.\
    assign(n = 1).\
    groupby(['qid', 'truth'], as_index = False)['n'].\
    sum().\
    assign(n2 = 1).\
    groupby(['qid'], as_index = False).\
    aggregate('sum').\
    query('n2 >= 4 and n < 30')


preds_df[preds_df.qid == '13663']
# excellent case to validate!


groups = preds_df.groupby('qid')

print(groups.size().min())
print(groups.size().max())



qid_13663 = preds_df[preds_df.qid =='13663']
qid_13663

qid_13663.sort_values(by = "predicted", ascending = False)


def bubble_sort(vals_list):
    swaps = 0
    n=len(vals_list)
    sorted = 0
    while sorted == 0:
        swaps_this_pass = 0
        for i in range(0, n-1):

            if vals_list[i]>vals_list[i+1]:
                vals_list[i], vals_list[i+1] = vals_list[i+1], vals_list[i]
                swaps_this_pass = swaps_this_pass + 1

        if swaps_this_pass==0:
            sorted=1
        swaps = swaps + swaps_this_pass        
        n = n-1 #ith pass of bubble sort puts the nth value in order.
            #so there is no need to consider this. 
    return(swaps)

truth_by_pred = qid_13663.sort_values(by = "predicted").truth.tolist()
bubble_sort(truth_by_pred)



def bubble_swaps(group):
    truth_by_pred = group.sort_values(by = "predicted").truth.tolist()
    swaps = bubble_sort(truth_by_pred)
    return swaps

pairwise = groups.apply(bubble_swaps)
pairwise.sample(10)

pairwise.mean()

# insane number of swaps needed => and I don't know if
# the above is working properly!


import math

def dcg(ordered_data):
    numerator = [(2 ** x) - 1 for x in ordered_data]
    denominator = [math.log(x +1, 2) for x in range(1, len(ordered_data)+1)]
    return sum([x/y for x, y in zip(numerator, denominator)])

truth_by_pred = qid_13663.sort_values(by = "predicted", ascending = False).truth.tolist()
truth_by_pred

optimal_ordering = sort(truth_by_pred)

dcg(truth_by_pred)




def nDCG(unordered_group):
    pred_ord = unordered_group.\
                    sort_values(by = "predicted", 
                                ascending = False).\
                    truth.\
                    tolist()

    opt_ord = unordered_group.\
                    sort_values(by = "truth", 
                                ascending = False).\
                    truth.\
                    tolist() 
    
    pred_dcg = dcg(pred_ord)
    opt_dcg = dcg(opt_ord)

    if opt_dcg == 0:
        return 0
    else:
        return pred_dcg / opt_dcg



groups = preds_df.groupby('qid')
dcg_score = groups.apply(nDCG)
dcg_score.sample(10)

sum(dcg_score == 0) / len(dcg_score)
# so only 3% of the cases do we have a ndcg of zero!


##### note that if a query has only 1 document for it
##### we would want to remove it from the analysis!
##### i.e. we would want to return nothing for it
##### also -> you could argue, we would want to 
##### normalise the results at nDCG@p where p = 5 for all
##### cases i.e. filter predictions to be top 5, and for that
##### list get the predicted cases


def nDCG_at_p(unordered_group,p = 5):
    pred_df = unordered_group.\
                    sort_values(by = "predicted", 
                                ascending = False).\
                    reset_index(drop = True).loc[0:(p-1), :]

    n = pred_df.shape[0]
    if n < p:
        result = np.nan
    else:
        pred_ord = pred_df.truth.tolist()

        opt_ord = pred_df.\
                        sort_values(by = "truth", 
                                    ascending = False).\
                        truth.\
                        tolist() 

        pred_dcg = dcg(pred_ord)
        opt_dcg = dcg(opt_ord)


        if opt_dcg == 0:
            result = 0
        else:
            result = pred_dcg / opt_dcg
    
    return result


nDCG_at_p(preds_df.query('qid == "10003"'), p = 10)

#### and you would want to filter our maybe only queries that 
#### have at least 10 documents so we'll be saying for cases where
#### we've made at least 10 recommendations, this is what we're seeing.

#### then we could summarise the results by taking the mean!

nDCG_at_5 = groups.apply(nDCG_at_p)
print('done')

np.nanmean(nDCG_at_5)


### that's it, and it's quite straight forward, although my
### coding will be quite avg!

