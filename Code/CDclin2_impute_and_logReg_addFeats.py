# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:16:25 2022

@author: iis
"""

import copy
import os
import numpy as np
import pandas as pd
import time
# from sklearnex import patch_sklearn
# patch_sklearn()

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn import preprocessing
import matplotlib.pyplot as plt



dat_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_135.xlsx'
out_dir = 'J:\\Summer-Students\\Output\\Clinical FE 3\\'

run_svm = True
write_data = True
exclude_normal = False
n_reps = 400


time_now = time.strftime("%Y_%m_%d-%H_%M")

Norm_path = "J:\\Summer-Students\\new_normal"
Abnorm_path = "J:\\Summer-Students\\new_cd"
os.listdir(Norm_path) + os.listdir(Abnorm_path)

# ##ensure the clinical data is mapped to outcomes in the same order 
df_clin = pd.read_excel(dat_path) #.iloc[:, 2:]
# df_clin = df_clin[df_clin.CD == 0].append(
#     df_clin[df_clin.CD == 1])
# df_clin.index = range(len(df_clin.index))


if exclude_normal:
    fname_tag = "excludeNormal"
    #Exclude subjects '0'-'15' as no lab values    
    exclusions = ['IRC318H-0'+ str(i).zfill(2) for i in range(1,16)]
    df_clin = df_clin.loc[[df_clin.id[i] not in exclusions for i in range(len(df_clin.index))], :]
    df_clin.index = range(len(df_clin.index))
else:
    fname_tag = "includeNormal"
    
cols_categ = ['CD', 'race', 'ethnicity']
df_clin[cols_categ] = df_clin[cols_categ].astype('category')


# =============================================================================
# Strategy 1: Impute missing CRP, ESR, calprotectin, and lactoferrin
# =============================================================================

## Imputation strategy 1: median imputation
df_med = copy.deepcopy(df_clin)

med_crp = np.median(df_med.CRP.dropna())
med_esr = np.median(df_med.ESR.dropna())
med_feCal = np.median(df_med.fecal_calprotectin.dropna())
# med_feLac = np.median(df_med.fecal_lactoferrin.dropna())

df_med.loc[df_med['CRP'].isna(), 'CRP'] = med_crp
df_med.loc[df_med['ESR'].isna(), 'ESR'] = med_esr

tmp_fc = -100*np.ones(len(df_med))
tmp_fc[df_med.fecal_calprotectin <= 50]  = 0
tmp_fc[df_med.fecal_calprotectin.isna()] = 1
tmp_fc[[a and b for a, b in zip(df_med.fecal_calprotectin > 50, df_med.fecal_calprotectin <= 200)]]  = 1
tmp_fc[df_med.fecal_calprotectin >200]  = 2

df_med.loc[:, "fc_cat"] = tmp_fc
df_med.loc[df_med['fecal_calprotectin'].isna(), 'fecal_calprotectin'] = med_feCal
# df_med.loc[df_med['fecal_lactoferrin'].isna(), 'fecal_lactoferrin'] = med_feLac


# =============================================================================
# Strategy 2: Recode thresholds, including missing data
# =============================================================================

#     # CRP
# # ESR
# df_thresh = copy.deepcopy(df_clin)

# #CRP 
# df_thresh.loc[df_med.CRP <1] = 0
# df_thresh.loc[df_med.CRP <1] = 0

# # fecal_calprotectin as 0=0-50, 1 = missing or 51-200, 2=200+ 
# df_thresh.loc[df_thresh.fecal_calprotectin <= 50, "FC_cat"]  = 0
# df_thresh.loc[df_thresh.fecal_calprotectin.isna(), "FC_cat"] = 1
# df_thresh.loc[[a and b for a, b in zip(df_thresh.fecal_calprotectin > 50, df_thresh.fecal_calprotectin <= 200)],
#     "FC_cat"]  = 1
# df_thresh.loc[df_thresh.fecal_calprotectin >200, "FC_cat"]  = 2


# =============================================================================
# Add clinical features based on clinical thresholds:
# =============================================================================


if write_data:
    df_med.to_excel(
        ('J:\\Summer-Students\\Processed_data\\clinical_data_%s_impute_addthresh.xlsx' % fname_tag),
        index=False)

# ## Imputation strategy 2: MICE
# import miceforest as mf

# df_mice = copy.deepcopy(df_clin)

# kernel = mf.ImputationKernel(
#   data=df_mice,
#   save_all_iterations=True,
#   random_state=1991
# )

# # Run the MICE algorithm for 3 iterations on each of the datasets
# kernel.mice(3,verbose=True)
# # Make a multiple imputed dataset with our new data
# new_data_imputed = kernel.impute_new_data(new_data)
# # Return a completed dataset
# new_completed_data = new_data_imputed.complete_data(0)



# =============================================================================
# Evaluate performance via 5-fold CV
# =============================================================================

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

# parameters for stratified kfold and bootstrapped ci
df_clin.CD = df_clin.CD.astype(int)
n_fold = 5

## 5-fold stratified cross validation
y = df_med.CD
# x1_colnames = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', 'ht', 'wt']

x1_colnames = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', "fc_cat", 'wt', 'ht']
#x1_colnames = ['age','female','BMI','CRP','ESR','fecal_calprotectin']
#x2_colnames = ['age','female', 'BMI','CRP','ESR','fecal_calprotectin','fecal_lactoferrin']

x1 = df_med[x1_colnames]
#x2 = df_med[x2_colnames]


## evaluate model with x1 parameters
perf_reg = pd.DataFrame(columns = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'roc_auc'])
coef_mat = pd.DataFrame(columns = x1_colnames)

perf_svm = pd.DataFrame(columns = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'roc_auc'])

x=x1
y=y
print("Training clinical logistic reg and SVM, %s reps", n_reps)

for i in range(n_reps):
    print(str(i), ", ", end="")
    
    reg_pred = -100*np.ones(len(y))
    reg_score = -100*np.ones(len(y))
    
    svm_pred = -100*np.ones(len(y))
    svm_score = -100*np.ones(len(y))
    
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)

    for train_i, test_i in skf.split(x1, y):
        
        ## Divide test/training data
        x_train, x_test = x.loc[train_i], x.loc[test_i]
        y_train, y_test = y.loc[train_i], y.loc[test_i]
        
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_norm = scaler.transform(x_train)
        x_test_norm = scaler.transform(x_test)
    
        ## Train L2 logistic regression model with these 6 variables
        
        # c = 1
        tuned_parameters = [{'C':  np.logspace(-2,2,5) }]
        lr_clin = LogisticRegression(max_iter=1000)
        lr_clin_cv = GridSearchCV(lr_clin, tuned_parameters, cv=n_fold)
        lr_clin_cv.fit(x_train_norm, y_train)
        
        model = LogisticRegression(
            solver='liblinear', penalty = 'l2',
            C = lr_clin_cv.best_params_["C"]).fit(x_train_norm, y_train)
        #model.score(x_train, y_train) 
        coef_mat.loc[len(coef_mat.index)] = np.exp(model.coef_)[0].tolist()
        reg_score[test_i] = model.decision_function(x_test_norm)  
        reg_pred[test_i] = model.predict(x_test_norm)
        

        
        if run_svm:
            # ## Train SVM without feature selection after normalizing

            
            tuned_parameters = [{'kernel': ['linear'], 'C':  np.logspace(-3,3,7) }]  #    [1,2,3,4,5,6]    # np.logspace(-3,3,7) 
            clf_cv = GridSearchCV(svm.SVC(), 
                                  tuned_parameters, 
                                  scoring='accuracy',    #  scoring='roc_auc'
                                  cv=n_fold)  
            clf_cv.fit(x_train_norm, y_train)
     
            clf = svm.SVC(kernel='linear', C= clf_cv.best_params_['C'])
            clf.fit(x_train_norm, y_train)
            
            svm_score[test_i] = clf.decision_function(x_test_norm)  
            svm_pred[test_i] = clf.predict(x_test_norm)
    
    #eval perf after adding all CV-segments together
    perf_reg.loc[len(perf_reg.index)] = eval_perf(y, reg_pred, reg_score)
    if run_svm: 
        perf_svm.loc[len(perf_svm.index)] = eval_perf(y, svm_pred, svm_score)

        




        

# =============================================================================
# Output performance metrics    
# =============================================================================
perf_mean = np.mean(perf_reg,axis=0)
perf_std = np.std(perf_reg,axis=0)
perf_median = perf_reg.apply(np.median, axis = 0)
perf_ci_low = perf_reg.apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_reg.index)*[2.5])
perf_ci_high = perf_reg.apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_reg.index)*[2.5])
Path(out_dir).mkdir(parents=True, exist_ok=True)


df_perf = pd.DataFrame(columns = [ #performance vars w label
    'model', 'auc mean', 'auc_95', 'acc mean', 'acc_95', 'sens mean', 'sens_95', 
    'spec mean', 'spec_95', 'ppv mean', 'ppv_95', 'npv mean', 'npv_95'
    ])

#logistic
df_perf.loc[len(df_perf.index), :] = [
    'logistic: ' + ' + '.join(x1_colnames),
    perf_mean[5], '%0.3f (%0.3f-%0.3f)' % (perf_median[5], perf_ci_low[5], perf_ci_high[5]),
    perf_mean[0], '%0.3f (%0.3f-%0.3f)' % (perf_median[0], perf_ci_low[0], perf_ci_high[0]),
    perf_mean[1], '%0.3f (%0.3f-%0.3f)' % (perf_median[1], perf_ci_low[1], perf_ci_high[1]),
    perf_mean[2], '%0.3f (%0.3f-%0.3f)' % (perf_median[2], perf_ci_low[2], perf_ci_high[2]),
    perf_mean[3], '%0.3f (%0.3f-%0.3f)' % (perf_median[3], perf_ci_low[3], perf_ci_high[3]),
    perf_mean[4], '%0.3f (%0.3f-%0.3f)' % (perf_median[4], perf_ci_low[4], perf_ci_high[4]),
    ]  


# =============================================================================
# If run SVM, eval SVM perf and write to "+SVM" file. Otherwise, to logistic reg
# =============================================================================
if run_svm:
    perf_mean = np.mean(perf_svm,axis=0)
    perf_std = np.std(perf_svm,axis=0)
    perf_median = perf_svm.apply(np.median, axis = 0)
    perf_ci_low = perf_svm.apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_reg.index)*[2.5])
    perf_ci_high = perf_svm.apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(perf_reg.index)*[2.5])



    #logistic
    df_perf.loc[len(df_perf.index), :] = [
        'SVM: ' + ' + '.join(x1_colnames),
        perf_mean[5], '%0.3f (%0.3f-%0.3f)' % (perf_median[5], perf_ci_low[5], perf_ci_high[5]),
        perf_mean[0], '%0.3f (%0.3f-%0.3f)' % (perf_median[0], perf_ci_low[0], perf_ci_high[0]),
        perf_mean[1], '%0.3f (%0.3f-%0.3f)' % (perf_median[1], perf_ci_low[1], perf_ci_high[1]),
        perf_mean[2], '%0.3f (%0.3f-%0.3f)' % (perf_median[2], perf_ci_low[2], perf_ci_high[2]),
        perf_mean[3], '%0.3f (%0.3f-%0.3f)' % (perf_median[3], perf_ci_low[3], perf_ci_high[3]),
        perf_mean[4], '%0.3f (%0.3f-%0.3f)' % (perf_median[4], perf_ci_low[4], perf_ci_high[4]),
        ]  
    df_perf.to_excel(out_dir + fname_tag + " Clinical logisticReg + SVM performance FE %s.xlsx" % n_reps, index=False)

else:
    df_perf.to_excel(out_dir + fname_tag + " Clinical logisticReg performance FE %s.xlsx" % n_reps, index=False)

#print(perf_reg)
#print(perf_str)


# =============================================================================
# Output model coefficients and CI
# =============================================================================
coef_out = pd.DataFrame(columns = [ #coeformance vars w label
    'mask', 'age mean', 'age 95', 'female mean', 'female 95', 'BMI_z mean', 'BMI_z_95', 
    'CRP mean', 'CRP_95', 'ESR mean', 'ESR_95', 'fecal_calprotectin mean', 'fecal_calprotectin_95'
    ])
coef_mean = np.mean(coef_mat,axis=0)
coef_std = np.std(coef_mat,axis=0)
coef_median = coef_mat.apply(np.median, axis = 0)
coef_ci_low = coef_mat.apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(coef_mat.index)*[2.5])
coef_ci_high = coef_mat.apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(coef_mat.index)*[2.5])


coef_out.loc[len(coef_out.index), :] = [
    ' + '.join(x1_colnames),
    coef_mean[5], '%0.3f (%0.3f-%0.3f)' % (coef_median[5], coef_ci_low[5], coef_ci_high[5]),
    coef_mean[0], '%0.3f (%0.3f-%0.3f)' % (coef_median[0], coef_ci_low[0], coef_ci_high[0]),
    coef_mean[1], '%0.3f (%0.3f-%0.3f)' % (coef_median[1], coef_ci_low[1], coef_ci_high[1]),
    coef_mean[2], '%0.3f (%0.3f-%0.3f)' % (coef_median[2], coef_ci_low[2], coef_ci_high[2]),
    coef_mean[3], '%0.3f (%0.3f-%0.3f)' % (coef_median[3], coef_ci_low[3], coef_ci_high[3]),
    coef_mean[4], '%0.3f (%0.3f-%0.3f)' % (coef_median[4], coef_ci_low[4], coef_ci_high[4]),
    ]  

coef_out.to_excel(out_dir + "Clinical logistic reg odds ratios FE %s.xlsx"% n_reps, index=False)
#print(coef_mat)
#print(coef_str)
