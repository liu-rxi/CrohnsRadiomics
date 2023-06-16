# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:43:17 2022

@author: iis
"""

import scipy.stats as st
import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import plot_roc_curve, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from statsmodels.stats import inter_rater as irr


## Returns list of 5 performance metrics given a set of predictions and truth
def eval_perf(y, y_pred, y_score):
    CM = confusion_matrix(y, y_pred)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]

    Population = TN+FN+TP+FP
    Prevalence = (TP+FP) / Population
    Accuracy   = (TP+TN) / Population
    Sensitivity= TP / (TP+FN)
    Specificity= TN / (TN+FP)
    PPV        = TP / (TP+FP)
    NPV        = TN / (TN+FN)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y, y_score) 
    roc_auc = auc(fpr, tpr)
    
    # Perf_table[repeat_idx,:]=[Accuracy, Sensitivity, Specificity, PPV, NPV, roc_auc]
    return([Accuracy, Sensitivity, Specificity, PPV, NPV, roc_auc])


dat_path = "J:\\Summer-Students\\Processed_data\\Raters_raw\\"
out_data_path = 'J:\\Summer-Students\\Processed_data\\'
out_path = 'J:\\Summer-Students\\Output\\'

dat = pd.read_excel(dat_path+'Towbin CD Reads.xlsx')
key = pd.read_excel(dat_path+'key\\135 CD reads - key randomized.xlsx')

# DataFrame for key + predictions from each rater --> save to processed data
df_raters = pd.DataFrame(
    columns = list(key.columns), 
    index = key.index)
df_raters.loc[:,'id_ResearchPACS'] = key.loc[:,'id_ResearchPACS']
df_raters.loc[:,'CD'] = key.loc[:,'CD']



dir_arr =  glob(dat_path + '*.xlsx')


# =============================================================================
# Rater performance and Kappa
# =============================================================================
perf_mat = pd.DataFrame(columns = ['rater', 'acc', 'sens', 'spec', 'ppv', 'npv'])

for i in range(len(dir_arr)):
    rater_path = dir_arr[i]
    raterid = 'rater'+str(i)
    
    dat_rater = pd.read_excel(rater_path)
    pred = dat_rater.iloc[:,1]
    
    df_raters.loc[:, raterid] = pred
    
    #all(dat_rater.id_ResearchPACS == df_raters.id_ResearchPACS)
    # mismatch = [j for j in range(len(dat_rater.id_ResearchPACS)) if dat_rater.id_ResearchPACS[j] != df_raters.id_ResearchPACS[j]]
    # print(mismatch)
    
    cm = confusion_matrix(key.CD, pred)
    total = sum(sum(cm))
    perf_mat.loc[i, 'rater'] = raterid
    perf_mat.loc[i, 'acc'] = accuracy = (cm[0,0]+cm[1,1])/total
    perf_mat.loc[i, 'sens'] = sens = cm[0,0]/(cm[0,0]+cm[0,1])
    perf_mat.loc[i, 'spec'] = spec = cm[1,1]/(cm[1,0]+cm[1,1])
    perf_mat.loc[i, 'ppv'] = cm[0,0]/(cm[0,0] + cm[1,0])
    perf_mat.loc[i, 'npv'] = cm[1,1]/(cm[1,1] + cm[0,1])

    if "Towbin" in rater_path:
        towbin_path = rater_path

#Majority vote performance
i=i+1
vote = sum(np.asarray(df_raters.iloc[:,[2,3,4]]).transpose())
consensus = (vote >= 2).astype(int)

cm = confusion_matrix(key.CD, consensus)
total = sum(sum(cm))
perf_mat.loc[i, 'rater'] = "majority"
perf_mat.loc[i, 'acc'] = accuracy = (cm[0,0]+cm[1,1])/total
perf_mat.loc[i, 'sens'] = sens = cm[0,0]/(cm[0,0]+cm[0,1])
perf_mat.loc[i, 'spec'] = spec = cm[1,1]/(cm[1,0]+cm[1,1])
perf_mat.loc[i, 'ppv'] = cm[0,0]/(cm[0,0] + cm[1,0])
perf_mat.loc[i, 'npv'] = cm[1,1]/(cm[1,1] + cm[0,1])

print(perf_mat)
perf_mat.to_excel(out_path + "Radiologist rater performance.xlsx")

 
#Kappa
agg = irr.aggregate_raters(df_raters.iloc[:,[2,3,4]]) # returns a tuple (data, categories)
print("Kappa: ", irr.fleiss_kappa(agg[0], method='fleiss'))



## Placeholder - then, write this perf mat
            
# =============================================================================
# Modeling Towbin's measurements 
# =============================================================================
dat_rater = pd.read_excel(towbin_path) 
x = dat_rater[['max_thickness (mm)']] #median 4mm, IQR 2-6mm
y = df_raters.CD

n_reps = 2000
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True)

perf_model = pd.DataFrame(columns = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'roc_auc'])
thresh = []
for j in range(n_reps):
    
    y_pred = -100*np.ones(len(y))
    y_prob = -100*np.ones(len(y))
    
    for train_i, test_i in skf.split(x, y):
        #print("Train: ", train_i, " Test:", test_i)
        x_train, x_test = x.loc[train_i], x.loc[test_i]
        y_train, y_test = y.loc[train_i], y.loc[test_i]
        
    
        model = LogisticRegression(solver='liblinear', penalty = 'l2', random_state=0).fit(x_train, y_train)
        #model.score(x_train, y_train) 
        #print(classification_report(y_test, model.predict(x_test)))
        
        thresh.append((-model.intercept_/model.coef_)[0][0])
        y_pred[test_i] = model.predict(x_test)
        y_prob[test_i] = model.predict_proba(x_test)[:,1]
        
    perf = eval_perf(y, y_pred, y_prob)
    perf_model.loc[len(perf_model.index)] = perf



    
perf_mean = np.mean(perf_model,axis=0)
perf_std = np.std(perf_model,axis=0)
perf_median = perf_model.apply(np.median, axis = 0)
perf_ci_low = perf_model.apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_mat.index)*[2.5])
perf_ci_high = perf_model.apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_mat.index)*[2.5])

for i in range(len(x_test.index)):
    x_test.iloc[i] = (i+2)*0.3
model.predict(x_test)
# perf_str = [
#     ' + '.join(x1_colnames),
#     perf_mean[5], perf_std[5], '%0.3f (%0.3f-%0.3f)' % (perf_mean[5], perf_mean[5]-1.97*perf_std[5], perf_mean[5]+1.97*perf_std[5]),
#     perf_mean[3], perf_std[3], '%0.3f (%0.3f-%0.3f)' % (perf_mean[3], perf_mean[3]-1.97*perf_std[3], perf_mean[3]+1.97*perf_std[3]),
#     perf_mean[4], perf_std[4], '%0.3f (%0.3f-%0.3f)' % (perf_mean[4], perf_mean[4]-1.97*perf_std[4], perf_mean[4]+1.97*perf_std[4]),
#     perf_mean[1], perf_std[1], '%0.3f (%0.3f-%0.3f)' % (perf_mean[1], perf_mean[1]-1.97*perf_std[1], perf_mean[1]+1.97*perf_std[1]),
#     perf_mean[2], perf_std[2], '%0.3f (%0.3f-%0.3f)' % (perf_mean[2], perf_mean[2]-1.97*perf_std[2], perf_mean[2]+1.97*perf_std[2]),
#     perf_mean[0], perf_std[0], '%0.3f (%0.3f-%0.3f)' % (perf_mean[0], perf_mean[0]-1.97*perf_std[0], perf_mean[0]+1.97*perf_std[0]),
#     ]   

thick_perf = pd.DataFrame(
    columns = ['auc mean', 'auc_95', 'acc mean', 'acc_95', 'sens mean', 'sens_95', 
    'spec mean', 'spec_95', 'ppv mean', 'ppv_95', 'npv mean', 'npv_95'
    ])
thick_perf.loc[0,:] =[
    perf_mean[5], '%0.3f (%0.3f-%0.3f)' % (perf_median[5], perf_ci_low[5], perf_ci_high[5]),
    perf_mean[0], '%0.3f (%0.3f-%0.3f)' % (perf_median[0], perf_ci_low[0], perf_ci_high[0]),
    perf_mean[1], '%0.3f (%0.3f-%0.3f)' % (perf_median[1], perf_ci_low[1], perf_ci_high[1]),
    perf_mean[2], '%0.3f (%0.3f-%0.3f)' % (perf_median[2], perf_ci_low[2], perf_ci_high[2]),
    perf_mean[3], '%0.3f (%0.3f-%0.3f)' % (perf_median[3], perf_ci_low[3], perf_ci_high[3]),
    perf_mean[4], '%0.3f (%0.3f-%0.3f)' % (perf_median[4], perf_ci_low[4], perf_ci_high[4]),
    ]   

print(thick_perf)
pd.DataFrame(thick_perf).to_excel(out_path + "Thickness model 1k.xlsx")

plot_roc_curve(model, x_test, y_test)

thresh_df = pd.DataFrame(thresh)
thresh_df.describe()
