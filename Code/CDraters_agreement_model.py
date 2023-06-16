# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:22:59 2022

@author: iis
"""

import scipy.stats as st
import pandas as pd
from glob import glob
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



dat_path = "J:\\Summer-Students\\Processed_data\\Raters_raw\\"
out_data_path = 'J:\\Summer-Students\\Processed_data\\'
out_path = 'J:\\Summer-Students\\Output\\'

key = pd.read_excel(dat_path+'key\\135 CD reads - key randomized.xlsx')

# DataFrame for key + predictions from each rater --> save to processed data
df_raters = pd.DataFrame(
    columns = list(key.columns), 
    index = key.index)
df_raters.loc[:,'id_ResearchPACS'] = key.loc[:,'id_ResearchPACS']
df_raters.loc[:,'CD'] = key.loc[:,'CD']


# DataFrame of predictive performance of each rater
perf_mat = pd.DataFrame(columns = ['rater', 'acc', 'sens', 'spec'])

dir_arr =  glob(dat_path + '*.xlsx')

for i in range(len(dir_arr)):
    rater_path = dir_arr[i]
    raterid = 'rater'+str(i)
    
    dat_rater = pd.read_excel(rater_path)
    pred = dat_rater.iloc[:,1]
    
    df_raters.loc[:, raterid] = pred
    
    #all(dat_rater.id_ResearchPACS == df_raters.id_ResearchPACS)
    mismatch = [j for j in range(len(dat_rater.id_ResearchPACS)) if dat_rater.id_ResearchPACS[j] != df_raters.id_ResearchPACS[j]]
    print(mismatch)
    
    cm = confusion_matrix(key.CD, pred)
    total = sum(sum(cm))
    perf_mat.loc[i, 'rater'] = raterid
    perf_mat.loc[i, 'acc'] = accuracy = (cm[0,0]+cm[1,1])/total
    perf_mat.loc[i, 'sens'] = sens = cm[0,0]/(cm[0,0]+cm[0,1])
    perf_mat.loc[i, 'spec'] = spec = cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    

    
arr = df_raters.iloc[:, [2,3,4]]    
agg = irr.aggregate_raters(arr) # returns a tuple (data, categories)
kappa = irr.fleiss_kappa(agg[0], method='fleiss')
print("Kappa: " + str(kappa))


df_raters.to_excel(out_data_path + "Reads - key and 3_raters.xlsx")     
perf_mat.to_excel(out_path + "Radiologist raters performance.xlsx")



