import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math


test = pd.read_pickle('MSLR-WEB10K/Fold1/test.pkl')

# you want a linear regression one as well!


pred_base = pd.read_csv("MSLR-WEB10K/Fold1/pred_linearModel.txt", header=None, sep=" ")[1]
pred_lm6 = pd.read_csv("MSLR-WEB10K/Fold1/pred_lm6.txt", header=None, sep=" ")[1]
pred_lm2 = pd.read_csv("MSLR-WEB10K/Fold1/pred_lm2.txt", header=None, sep=" ")[1]
pred_lr6 = pd.read_csv("MSLR-WEB10K/Fold1/pred_lr6.txt", header=None, sep=" ")[1]
pred_lr2 = pd.read_csv("MSLR-WEB10K/Fold1/pred_lr2.txt", header=None, sep=" ")[1]


len(pred_base)
len(pred_lm6)

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

def bubble_swaps(group):
    truth_by_pred = group.sort_values(by = "predicted").truth.tolist()
    swaps = bubble_sort(truth_by_pred)
    return swaps


""" this returns 0 if all of the ordered data is undesirable"""
def ndcg_p(ordered_data, p):
    """normalised discounted cumulative gain"""
    if sum(ordered_data)==0:
        return 0
    else:
        indexloop = range(0, p)
        DCG_p = 0
        for index in indexloop:
            current_ratio=(2**(ordered_data[index])-1)*(math.log((float(index)+2), 2)**(-1))
            DCG_p = DCG_p + current_ratio
        sorted_data= sorted(ordered_data,reverse=True)
        n = len(ordered_data)
        indexloop = range(0, n)
        iDCG_p = 0
        for index in indexloop:
            current_ratio=(2**(sorted_data[index])-1)*((math.log((index+2), 2))**(-1))
            iDCG_p = iDCG_p + current_ratio
        return(DCG_p/iDCG_p)
    
    
def compute_at_p(ordered_data, p): 
    """
    returns -1 if number of items in list is less than p
    else returns ndcg@p
    """
    n = len(ordered_data)
    if n<p:
        result = np.nan
    else:
        result = ndcg_p(ordered_data, p)
    
    return result


def sort_ndcg(unordered_group, p):
    ordered_group = unordered_group.sort_values(by = "predicted").truth.tolist()
    return compute_at_p(ordered_group, p)


print(mean_squared_error(pred_base, y_test))
print(mean_squared_error(pred_lm6, y_test))
print(mean_squared_error(pred_lm2, y_test))
print(mean_squared_error(pred_lr6, y_test))
print(mean_squared_error(pred_lr2, y_test))


preds_df_base = pd.DataFrame({'qid': test[1], 
                             'truth' :y_test, 
                            'predicted' : pred_base })
groups_base = preds_df_base.groupby('qid')
pairwise_base = groups_base.apply(bubble_swaps)
np.mean(pairwise_base)
ndcg_base_100 = groups_base.apply(sort_ndcg, p=100)
np.nanmean(ndcg_base_100)




preds_df_lm6 = pd.DataFrame({'qid': test[1], 
                             'truth' :y_test, 
                            'predicted' : pred_lm6 })
groups_lm6 = preds_df_lm6.groupby('qid')
pairwise_lm6 = groups_lm6.apply(bubble_swaps)
np.mean(pairwise_lm6)
ndcg_lm6_100 = groups_lm6.apply(sort_ndcg, p=100)
np.nanmean(ndcg_lm6_100)


preds_df_lm2 = pd.DataFrame({'qid': test[1], 
                             'truth' :y_test, 
                             'predicted' : pred_lm2 })
groups_lm2 = preds_df_lm2.groupby('qid')
pairwise_lm2 = groups_lm2.apply(bubble_swaps)
print(np.mean(pairwise_lm2))
ndcg_lm2_100 = groups_lm2.apply(sort_ndcg, p=100)
print(np.nanmean(ndcg_lm2_100))



# so you can see that, the linear model isn't doing
# as well when it comes to predicting the ranking information
# and this is what we were expecting, so I'm pleased with this
# result...Also, it's clear that the lamdaMart model that uses
# nDCG directly model 6 does best! very very happy with this result!

print(f'linear regression = {np.nanmean(ndcg_base_100)}')
print(f'rank:pairwise = {np.nanmean(ndcg_lm6_100)}')
print(f'rank:ndcg = {np.nanmean(ndcg_lm2_100)}')




ndcg_lm2_500 = groups_lm2.apply(sort_ndcg, p=500)
print(np.nanmean(ndcg_lm2_500))

ndcg_lm6_500 = groups_lm6.apply(sort_ndcg, p=500)
print(np.nanmean(ndcg_lm6_500))