#from scipy import stats
import numpy as np
import pandas as pd
import time
from pathlib import Path
import itertools
from sklearnex import patch_sklearn
import os
patch_sklearn()

from sklearn import svm
#from scipy.misc import comb
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.ensemble import StackingClassifier
from mlxtend.classifier import StackingCVClassifier


# =============================================================================
# Setup: setup parameters you want here
#  1. single feature
#  2. clinical variable set
#  3. 
#
# =============================================================================

feat_flag = "wallCore"  # 1 mask to run
#feat_flag = ["wallCore", "fat"] #will recognize list

clin_setup = "exp"      #orig, noFC (-fecal_calprotectin), exp (+fc_cat)
clin_model = "LinearSVM" #LR, LinearSVM
meta_model = "LinearSVM" #LR, LinearSVM, or RBFSVM

#model details
total_exp = 100 # repitions for bootstrapping - use low val for now, then increase to 1k for publication
fs_flag = True
param_type = '02mm'

# create out path based on parameters
fname_prefix = "%s - clin_%s_%s, meta_%s" % (feat_flag, clin_setup, clin_model, meta_model)
if fs_flag == False: 
    fname_prefix = fname_prefix + " noFS"
out_path = 'J:\\Summer-Students\\Output\\Stack, %s\\' % fname_prefix 
model_list = [feat_flag, "clinical", "stacked score", "stacked aug"]
n_meta =2

if clin_setup == "noFC": #
    clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_imputed.xlsx'
    clin_colns = ['age','female','BMI_z','CRP','ESR', "wt", "ht"]
elif clin_setup == "orig":
    clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_imputed.xlsx'
    clin_colns = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', "wt", "ht"]
elif clin_setup == "origLog":
    clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_imputed_log2.xlsx'
    clin_colns = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', "wt", "ht"]
elif clin_setup == "exp":
    clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_impute_addThresh.xlsx'
    clin_colns = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', "fc_cat", "wt", "ht"]
elif clin_setup == "expLog":
    clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_impute_addThresh_log2.xlsx'
    clin_colns = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin', "fc_cat", "wt", "ht"]

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


# =============================================================================
# Read data and organize
# =============================================================================

dat_path = 'J:\\Summer-Students\\Processed_data\\'
time_now = time.strftime("%Y_%m_%d-%H_%M")

abn_name = 'Crohns'
norm_name = 'Normal'

## test all possible pairs
#feat_flags = list(["wall", "wallCore", "fat", "fatCore"])
#to_mix = list(itertools.combinations(feat_flags, 2)) #all pairwise combinations

# note - currently there is no explicit matching of ID - it's all implicit positional
# if there are significant changes to the order of subjects, ensure that the
# end result of (norm + abnorm) subjects lines up with the above clinical data file
df_clin = pd.read_excel(clin_path)
df_clin.female = df_clin.female.astype(int)
clin_features = np.array(df_clin[clin_colns])

Norm_wall_features_pd = pd.read_pickle(dat_path + param_type + norm_name+'_wall_features.pkl')  
Norm_wallCore_features_pd = pd.read_pickle(dat_path + param_type + norm_name+'_wallCore_features.pkl') 
Norm_fat_features_pd = pd.read_pickle(dat_path + param_type + norm_name+'_fat_features.pkl')
Norm_fatCore_features_pd = pd.read_pickle(dat_path + param_type + norm_name+'_fatCore_features.pkl')

Abnorm_wall_features_pd_1 = pd.read_pickle(dat_path + param_type + abn_name+'_wall_features.pkl')  
Abnorm_wallCore_features_pd_1 = pd.read_pickle(dat_path + param_type + abn_name+'_wallCore_features.pkl') 
Abnorm_fat_features_pd_1 = pd.read_pickle(dat_path + param_type + abn_name+'_fat_features.pkl')
Abnorm_fatCore_features_pd_1 = pd.read_pickle(dat_path + param_type + abn_name+'_fatCore_features.pkl')

## Append prefix to feature names:
Norm_wall_features_pd.columns = ["wall "+x for x in Norm_wall_features_pd.columns.tolist()]    
Norm_wallCore_features_pd.columns = ["wallCore "+x for x in Norm_wallCore_features_pd.columns.tolist()]    
Norm_fat_features_pd.columns = ["fat "+x for x in Norm_fat_features_pd.columns.tolist()]    
Norm_fatCore_features_pd.columns = ["fatCore "+x for x in Norm_fatCore_features_pd.columns.tolist()]    

Abnorm_wall_features_pd_1.columns = ["wall "+x for x in Abnorm_wall_features_pd_1.columns.tolist()]    
Abnorm_wallCore_features_pd_1.columns = ["wallCore "+x for x in Abnorm_wallCore_features_pd_1.columns.tolist()]    
Abnorm_fat_features_pd_1.columns = ["fat "+x for x in Abnorm_fat_features_pd_1.columns.tolist()]    
Abnorm_fatCore_features_pd_1.columns = ["fatCore "+x for x in Abnorm_fatCore_features_pd_1.columns.tolist()]    


# Initialize final output dfs for performance, feature ranking, and misclassification
df_perf = pd.DataFrame(columns = [ #performance vars w label
    'mask', 'auc mean', 'auc_95', 'acc mean', 'acc_95', 'sens mean', 'sens_95', 
    'spec mean', 'spec_95', 'ppv mean', 'ppv_95', 'npv mean', 'npv_95'
    ])

df_feat = pd.DataFrame() # top 10 features

pt_names = os.listdir("J:\\Summer-Students\\new_normal\\") + os.listdir("J:\\Summer-Students\\new_cd")
df_misclas = pd.DataFrame({"pt_id": pt_names})


print("Training " + fname_prefix)


# =============================================================================
#   Set data up for modeling
# =============================================================================
if type(feat_flag) is list:
    print("2 mask not yet implemented")
else: ##   get data and labels per these parameters;   Note: texture feature start at index 14 of 107
        
    if feat_flag == 'wall':    # whole bowel, use all features
            feature_norm0 = Norm_wall_features_pd.to_numpy()
            feature_abnorm0 = Abnorm_wall_features_pd_1.to_numpy()
            feature_names0 = Norm_wall_features_pd
    elif feat_flag == 'wallCore': #use only texture features
            feature_norm0 = Norm_wallCore_features_pd.to_numpy()[:,14:]
            feature_abnorm0 = Abnorm_wallCore_features_pd_1.to_numpy()[:,14:]     
            feature_names0 = list(Norm_wallCore_features_pd)[14:]
    elif feat_flag == 'fat': #use all features
            feature_norm0 = Norm_fat_features_pd.to_numpy()
            feature_abnorm0 = Abnorm_fat_features_pd_1.to_numpy()
            feature_names0 = list(Norm_fat_features_pd)  
    elif feat_flag == 'fatCore': #use only texture  features
            feature_norm0 = Norm_fatCore_features_pd.to_numpy()[:,14:]
            feature_abnorm0 = Abnorm_fatCore_features_pd_1.to_numpy()[:,14:]  
            feature_names0 = list(Norm_fatCore_features_pd)[14:]

## consolidate data for models (normal + abnormal)
x0 = np.concatenate((feature_norm0, feature_abnorm0),axis=0)
y = np.concatenate((np.zeros(feature_norm0.shape[0]), np.ones(feature_abnorm0.shape[0])), axis=0)
feature_names0 = [sub.replace('original_', '') for sub in feature_names0] 

#  models to track being concurrently trained in each loop
#  repeat configuration - list of Perf_tables, for 5 models - 3 layer1, 2 layer2
Perf_tables = [-100*np.ones([total_exp, 6]) for i in range(len(model_list))]
n_feats = []
misclassified_i = [np.zeros(len(y)) for i in range(len(model_list))]
score_avgs = [np.zeros(len(y)) for i in range(len(model_list))] # avg scores for BAGGED model --> ROC curve

#Track FS for Layer0 models
feature_ranking = [np.zeros(len(feature_names0)), np.zeros(len(clin_colns))] 
feature_selection_freq = [np.zeros(len(feature_names0)), np.zeros(len(clin_colns))] 

#Track weights for meta models
meta_coefs = [np.zeros(n_meta), np.zeros(n_meta+len(feature_names0)+len(clin_colns))]

# =============================================================================
#   For (total_exp) number of repititions, 5-f with holdout (train, test)
#       On train, for all individual models (...)
#           5-f feature selection, then
#           5-f model tuning, then
#           5-f meta model
#       Then, evaluation on test
# =============================================================================
print('Model training')
for repeat_idx in range(total_exp): 
    
    print(str(repeat_idx) + ", ", end="")
    ##  single configuration
    n_fold = 5
    y0_pred = -100*np.ones(len(y))
    y0_score = -100*np.ones(len(y))

    clin_pred = -100*np.ones(len(y))
    clin_score = -100*np.ones(len(y))
    
    metaScore_pred = -100*np.ones(len(y))
    metaScore_score = -100*np.ones(len(y))
    
    metaAug_pred = -100*np.ones(len(y))
    metaAug_score = -100*np.ones(len(y))
    
    #loo = LeaveOneOut()
    #kf = KFold(n_splits=n_fold)
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    CV_idx = 1
    

    #  K Fold CV  or LOOCV loop
    for i,j in skf.split(x0, y):

        x0_train_raw = x0[i,:] #of mask 1
        x0_test_raw = x0[j,:] 
        clin_train = clin_features[i,]
        clin_test = clin_features[j]
        y_train = y[i]
        y_test = y[j]
        
        ## feature normalization
        scaler0 = preprocessing.StandardScaler().fit(x0_train_raw)
        x0_train_norm = scaler0.transform(x0_train_raw)
        x0_test_norm = scaler0.transform(x0_test_raw)
        
        scaler2 = preprocessing.StandardScaler().fit(clin_train)
        clin_train = scaler2.transform(clin_train)
        clin_test = scaler2.transform(clin_test)
        
    # =============================================================================
    #    Feature selection, for radiomic layer 0 models    
    # =============================================================================
        if fs_flag:  # True = do LASSO
            
            # 5-fold cv for mask 1
            para_grid={"C":  np.logspace(-3,3,7)}             # define grid
            lr = LogisticRegression(penalty = 'l1', solver='liblinear', max_iter=1000)
            lr_cv=GridSearchCV(lr, 
                               para_grid, 
                               scoring='accuracy',    #  scoring='roc_auc'
                               cv=n_fold)     
            lr_cv.fit(x0_train_norm, y_train)    
            lasso0 = LogisticRegression(penalty = 'l1', solver='liblinear', 
                                       C= lr_cv.best_params_['C'])
            lasso0.fit(x0_train_norm, y_train)
            
            #record lasso selection frequency    
            importance0 = np.abs(lasso0.coef_) #coefs for model 0
            importance_idx0 = (importance0>0)
            idx_sf0 = np.where(importance_idx0==1)[1] #numerical index of selected feats for model 0
            
            tmp_freq0 = np.zeros(len(feature_names0))
            tmp_freq0[idx_sf0] = 1 #to track with the SVM coefficients
            
            n_feats.append(sum(sum(importance_idx0)))

            # select features
            x0_train = np.take(x0_train_norm, idx_sf0, axis=1)
            x0_test = np.take(x0_test_norm, idx_sf0, axis=1)       
        
            #for multi modeling
            tmp_freq = tmp_freq0
            feature_selection_freq[0] = feature_selection_freq[0] + tmp_freq   
            idx_sf = idx_sf0
                
        else:
            x0_train = x0_train_norm
            x0_test =  x0_test_norm
            idx_sf = np.array(range(len(feature_names0)))
        
        
    # =============================================================================
    #   Layer 0 models: train
    # =============================================================================        
        ## model 0: clf0, for mask 1
        tuned_parameters = [{'kernel': ['linear'], 'C':  np.logspace(-3,3,7) }] 
        clf0_cv = GridSearchCV(svm.SVC(), 
                              tuned_parameters, 
                              scoring='accuracy',    #  scoring='roc_auc'
                              cv=n_fold)  
        clf0_cv.fit(x0_train, y_train)

        clf0 = svm.SVC(kernel='linear',C= clf0_cv.best_params_['C'])
        clf0.fit(x0_train, y_train)
        
           
        
        ## model 1: clinical - LinearSVM or LR
        if clin_model == "LinearSVM":
            tuned_parameters = [{'C':  np.logspace(-2,2,5) }]
            lr_clin = LogisticRegression()
            lr_clin_cv = GridSearchCV(lr_clin, tuned_parameters, cv=n_fold)
            lr_clin_cv.fit(clin_train, y_train)
            
            clf_clin = LogisticRegression(
                solver='liblinear', 
                penalty = 'l2', 
                C = lr_clin_cv.best_params_["C"])
            clf_clin.fit(clin_train, y_train)
        elif clin_model == "LR":
            tuned_parameters = [{'C':  np.logspace(-2,2,5) }]
            lr_clin = LogisticRegression()
            lr_clin_cv = GridSearchCV(lr_clin, tuned_parameters, cv=n_fold)
            lr_clin_cv.fit(clin_train, y_train)
            
            clf_clin = LogisticRegression(
                solver='liblinear', 
                penalty = 'l2', 
                C = lr_clin_cv.best_params_["C"])
            clf_clin.fit(clin_train, y_train)
        else:
            print("Improper clinical model selected, %s" % clin_model)
            quit()

    # =============================================================================
    #   Meta-classifiers - scores only, and feed-forward/augmented
    # =============================================================================
        ## Setup input data for meta models
        # MODEL SCORE: scale scores for SVM metaclassifier
        tmp_score_train = np.column_stack(( # scores must be scaled before metaclassifier input
            clf0.decision_function(x0_train), 
            clf_clin.decision_function(clin_train)))

        score_scaler = preprocessing.StandardScaler().fit(tmp_score_train) 
        x_score_train = score_scaler.transform(tmp_score_train)

        # MODEL AUG: score + input
        x_aug_train = np.concatenate((
            x_score_train, x0_train, clin_train), axis=1)
        
        ## Setup input data for meta models
        if meta_model == "LR":
            tuned_parameters = {'C':  np.logspace(-3,3,7)}

            # MODEL SCORE
            lr_meta_cv = GridSearchCV(LogisticRegression(max_iter=1000),tuned_parameters,cv=n_fold)
            lr_meta_cv.fit(x_score_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            #use_c = 1
            # print("meta score c: ", str(use_c))
            clf_meta_score = LogisticRegression(
                solver='liblinear', penalty = 'l2', C= use_c, max_iter=5000)
            clf_meta_score.fit(x_score_train, y_train)
            
            
            # MODEL SVM
            tuned_parameters = {'C':  np.logspace(-4,1,5)}
            lr_meta_cv = GridSearchCV(LogisticRegression(max_iter=1000),tuned_parameters,cv=n_fold)
            lr_meta_cv.fit(x_aug_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            #use_c = 0.01
            # print("meta aug c: ", str(use_c))
            clf_meta_aug = LogisticRegression(
                solver='liblinear', penalty = 'l2', C= use_c, max_iter=5000)
            clf_meta_aug.fit(x_aug_train, y_train)
            
        elif meta_model == "LinearSVM":
            tuned_parameters = {'kernel': ['linear'], 'C':  np.logspace(-2,3,6)}

            # MODEL SCORE
            lr_meta_cv = GridSearchCV(svm.SVC(), tuned_parameters, cv=n_fold)
            lr_meta_cv.fit(x_score_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            #use_c = 1
            #print("meta pred c: ", str(use_c))
            clf_meta_score = svm.SVC(kernel = 'linear', C= use_c)
            clf_meta_score.fit(x_score_train, y_train)
                
            # MODEL AUG
            lr_meta_cv = GridSearchCV(svm.SVC(), tuned_parameters, cv=n_fold)
            lr_meta_cv.fit(x_aug_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            #use_c = 1
            # print("meta pred c: ", str(use_c))
            clf_meta_aug = svm.SVC(kernel = 'linear', C= use_c)
            clf_meta_aug.fit(x_aug_train, y_train)
     
        elif meta_model == "RBFSVM":
            tuned_parameters = {'C':  np.logspace(-1,2,4), 'gamma': np.logspace(-2,1,4)}

            # MODEL SCORE
            lr_meta_cv = GridSearchCV(svm.SVC(), tuned_parameters, cv=n_fold)
            lr_meta_cv.fit(x_score_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            use_gamma = lr_meta_cv.best_params_["gamma"]
            # print("C: %s, gam: %s" % (use_c, use_gamma))
            clf_meta_score = svm.SVC(kernel = 'rbf', C= use_c, gamma= use_gamma)
            clf_meta_score.fit(x_score_train, y_train)
            

            # MODEL AUG

            lr_meta_cv = GridSearchCV(svm.SVC(), tuned_parameters, cv=n_fold)
            lr_meta_cv.fit(x_aug_train, y_train)
            use_c = lr_meta_cv.best_params_["C"]
            use_gamma = lr_meta_cv.best_params_["gamma"]
            # print("C: %s, gam: %s" % (use_c, use_gamma))
            clf_meta_aug = svm.SVC(kernel = 'rbf', C= use_c, gamma= use_gamma)
            clf_meta_aug.fit(x_aug_train, y_train)
            
    # =============================================================================
    #   Evaluate perf
    # =============================================================================
        y0_score[j] = clf0.decision_function(x0_test)  
        y0_pred[j] = clf0.predict(x0_test)
        clin_score[j] = clf_clin.decision_function(clin_test)  
        clin_pred[j] = clf_clin.predict(clin_test)
        
        x_pred_test = np.column_stack((y0_pred[j], clin_pred[j]))
        x_score_test = score_scaler.transform(
            np.column_stack((y0_score[j], clin_score[j])))
        x_aug_test = np.concatenate((x_score_test, x0_test, clin_test), axis=1)
       

        metaScore_score[j] = clf_meta_score.decision_function(x_score_test)
        metaScore_pred[j] = clf_meta_score.predict(x_score_test)
        metaAug_score[j] = clf_meta_aug.decision_function(x_aug_test)
        metaAug_pred[j] = clf_meta_aug.predict(x_aug_test)
        
        ## Feature ranking - can't rank for RBF SVM
        if meta_model != "RBFSVM":
            #model 0
            tmp_ranking = np.zeros(len(feature_names0))
            tmp_ranking[idx_sf] = np.abs(clf0.coef_)/np.max(np.abs(clf0.coef_))
            feature_ranking[0] = feature_ranking[0] + tmp_ranking
            #clin model
            tmp_ranking = np.abs(clf_clin.coef_[0])/np.max(np.abs(clf_clin.coef_[0]))
            feature_ranking[1] = feature_ranking[1] + tmp_ranking
            
            #meta models
            tmp_ranking = np.abs(clf_meta_score.coef_)/np.max(np.abs(clf_meta_score.coef_))
            meta_coefs[0] = meta_coefs[0] + tmp_ranking
            
            tmp_ranking = np.zeros(len(meta_coefs[1]))
            i_aug = list(range(n_meta)) + list(idx_sf + n_meta) + list(range(-len(clin_colns), 0))
            tmp_ranking[i_aug] = np.abs(clf_meta_aug.coef_)/np.max(np.abs(clf_meta_aug.coef_))
            meta_coefs[1] = meta_coefs[1] + tmp_ranking
            
        CV_idx = CV_idx + 1
        #  end of CV loop
    
    Perf_tables[0][repeat_idx,:] = eval_perf(y, y0_pred, y0_score)
    Perf_tables[1][repeat_idx,:] = eval_perf(y, clin_pred, clin_score)
    Perf_tables[2][repeat_idx,:] = eval_perf(y, metaScore_pred, metaScore_score)
    Perf_tables[3][repeat_idx,:] = eval_perf(y, metaAug_pred, metaAug_score)

    # count misclassified by clf_meta_aug
    misclassified_i[0][y0_pred != y] += 1
    misclassified_i[1][clin_pred != y] += 1
    misclassified_i[2][metaScore_pred != y] += 1
    misclassified_i[3][metaAug_pred != y] += 1

    #get PR curves for models
    score_avgs[0] += y0_score/total_exp
    score_avgs[1] += clin_score/total_exp
    score_avgs[2] += metaScore_score/total_exp
    score_avgs[3] += metaAug_score/total_exp

    # end of N repeat loop


# ## performance    
# # ACC 
# print('Accuracy = %0.3f' % np.mean(Perf_table[:,0],axis=0))
# # Sen
# print('Sensitivity = %0.3f' % np.mean(Perf_table[:,1],axis=0) )
# # Spe
# print('Specificity = %0.3f' % np.mean(Perf_table[:,2],axis=0))
# # PPV and NPV
# print('PPV = %0.3f' % np.mean(Perf_table[:,3],axis=0))
# print('NPV = %0.3f' % np.mean(Perf_table[:,4],axis=0))
# # Compute ROC curve and ROC area for each class
# print('AUC = %0.3f' % np.mean(Perf_table[:,5],axis=0))

Path(out_path).mkdir(parents=True, exist_ok=True)

for i in range(len(model_list)):
    Perf_table = Perf_tables[i]
    
    perf_mean = np.mean(Perf_table,axis=0)
    # perf_std = np.std(Perf_table,axis=0)
    
    
    # print(perf_mean)
    # print(perf_std) 
    perf_median = pd.DataFrame(Perf_table).apply(np.median, axis = 0)
    perf_ci_low = pd.DataFrame(Perf_table).apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    perf_ci_high = pd.DataFrame(Perf_table).apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    
    # df_perf.loc[len(df_perf)] = [
    #     '%s, %s' % (feat_flag, param_type),
    #     perf_mean[5], perf_std[5], '%0.3f (%0.3f-%0.3f)' % (perf_mean[5], perf_mean[5]-1.97*perf_std[5], perf_mean[5]+1.97*perf_std[5]),
    #     perf_mean[3], perf_std[3], '%0.3f (%0.3f-%0.3f)' % (perf_mean[3], perf_mean[3]-1.97*perf_std[3], perf_mean[3]+1.97*perf_std[3]),
    #     perf_mean[4], perf_std[4], '%0.3f (%0.3f-%0.3f)' % (perf_mean[4], perf_mean[4]-1.97*perf_std[4], perf_mean[4]+1.97*perf_std[4]),
    #     perf_mean[1], perf_std[1], '%0.3f (%0.3f-%0.3f)' % (perf_mean[1], perf_mean[1]-1.97*perf_std[1], perf_mean[1]+1.97*perf_std[1]),
    #     perf_mean[2], perf_std[2], '%0.3f (%0.3f-%0.3f)' % (perf_mean[2], perf_mean[2]-1.97*perf_std[2], perf_mean[2]+1.97*perf_std[2]),
    #     perf_mean[0], perf_std[0], '%0.3f (%0.3f-%0.3f)' % (perf_mean[0], perf_mean[0]-1.97*perf_std[0], perf_mean[0]+1.97*perf_std[0]),
    #     ]
    # df_perf.loc[len(df_perf)] = [
    #     '%s, %s' % (feat_flag, param_type),
    #     perf_mean[5], perf_std[5], '%0.3f (%0.3f-%0.3f)' % (perf_mean[5], perf_ci_low[5], perf_ci_high[5]),
    #     perf_mean[3], perf_std[3], '%0.3f (%0.3f-%0.3f)' % (perf_mean[3], perf_ci_low[3], perf_ci_high[3]),
    #     perf_mean[4], perf_std[4], '%0.3f (%0.3f-%0.3f)' % (perf_mean[4], perf_ci_low[4], perf_ci_high[4]),
    #     perf_mean[1], perf_std[1], '%0.3f (%0.3f-%0.3f)' % (perf_mean[1], perf_ci_low[1], perf_ci_high[1]),
    #     perf_mean[2], perf_std[2], '%0.3f (%0.3f-%0.3f)' % (perf_mean[2], perf_ci_low[2], perf_ci_high[2]),
    #     perf_mean[0], perf_std[0], '%0.3f (%0.3f-%0.3f)' % (perf_mean[0], perf_ci_low[0], perf_ci_high[0]),
    #     ]
    df_perf.loc[len(df_perf)] = [
        model_list[i],
        perf_mean[5], '%0.3f (%0.3f-%0.3f)' % (perf_median[5], perf_ci_low[5], perf_ci_high[5]),
        perf_mean[0], '%0.3f (%0.3f-%0.3f)' % (perf_median[0], perf_ci_low[0], perf_ci_high[0]),
        perf_mean[1], '%0.3f (%0.3f-%0.3f)' % (perf_median[1], perf_ci_low[1], perf_ci_high[1]),
        perf_mean[2], '%0.3f (%0.3f-%0.3f)' % (perf_median[2], perf_ci_low[2], perf_ci_high[2]),
        perf_mean[3], '%0.3f (%0.3f-%0.3f)' % (perf_median[3], perf_ci_low[3], perf_ci_high[3]),
        perf_mean[4], '%0.3f (%0.3f-%0.3f)' % (perf_median[4], perf_ci_low[4], perf_ci_high[4]),
        ]
    # Plot of a ROC curve for a specific class
    #plt.figure()
    #plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic curve')
    #plt.legend(loc="lower right")
    #plt.show()
    
    
    ## new columns for each type for 1) top features and 2) their weights (8 cols)
    ##   rank and print feature selection and ranking names
    if model_list[i] == feat_flag:
        i_feat_rank = feature_ranking[i].argsort()[::-1][:20] #0
        tmp_names = feature_names0
        tmp_df = feature_ranking[i]

    elif model_list[i] == "clinical":
        i_feat_rank = feature_ranking[i].argsort()[::-1][:20] #0    #Track FS for Layer0 models
        tmp_names = clin_colns
        tmp_df = feature_ranking[i]

    elif model_list[i] == 'stacked score':
        i_feat_rank = meta_coefs[0].argsort()[::-1]
        tmp_names = ["%s_pred" % feat_flag, "clin_pred"]
        tmp_df = meta_coefs[0]

    elif model_list[i] == 'stacked aug':
        i_feat_rank = meta_coefs[1].argsort()[::-1]
        tmp_names = ["%s_pred" % feat_flag, "clin_pred"] + feature_names0 + clin_colns
        tmp_df = meta_coefs[1]

    top_rank_features = np.take(tmp_names, i_feat_rank)
    top_rank_features_weight = np.take(tmp_df, i_feat_rank)
    top_rank = pd.DataFrame(
        zip( top_rank_features, top_rank_features_weight),
        columns=['%s_feat' % model_list[i], '%s_wt' % model_list[i]])
    
    if not df_feat.empty:
        df_feat = pd.concat([df_feat, top_rank], axis=1)
    else:
        df_feat = top_rank
    
    #misclassifications
    tmp_misclas =  pd.DataFrame({"%s error"%model_list[i]: 100*misclassified_i[i]/total_exp})
    df_misclas = pd.concat([df_misclas,tmp_misclas], axis = 1)


# =============================================================================
# Output performance, feature rank, misclassifications
# =============================================================================
n_feat_col = {"n_%s_feats"%feat_flag: ["%s (%s-%s)" % (np.median(n_feats), np.percentile(n_feats, 25), np.percentile(n_feats, 75))]}
df_feat = pd.concat([df_feat, pd.DataFrame(n_feat_col)], axis=1)
df_perf.to_excel(out_path + "%s %s perf %s %s.xlsx" % (fname_prefix, param_type, time_now, total_exp), index=False)
df_feat.to_excel(out_path + "%s %s features %s %s.xlsx" % (fname_prefix, param_type, time_now, total_exp), index=False)
df_misclas.to_excel(out_path + "%s %s miss_rate %s %s %s.xlsx" % (fname_prefix, param_type, model_list[i], time_now, total_exp), index=False)


# misclassification extensions: Kappa with human majority
df_human = pd.read_excel("J:\\Summer-Students\\Processed_data\\Reads - key and 3_raters - sorted.xlsx")
df_human.loc["majority"] = np.round(np.mean(df_human.iloc[:, 2:5], axis=1))

metaaug_majority = 

# =============================================================================
# Plot ROC curves + raters
# =============================================================================
rater_FPR = 1 - np.array([0.871, 0.729, 0.771])
rater_TPR = [0.892, 0.954, 0.969]

plt.figure(0).clf()
plt.scatter(rater_FPR, rater_TPR, label="Individual raters", marker = "^")
plt.scatter([1-0.881], [0.969], label="Majority of raters", marker = "^")



for i in range(len(model_list)):
    
    if i != 2:
        pl_fpr, pl_tpr, _ = roc_curve(y, score_avgs[i])
        pl_auc = round(roc_auc_score(y, score_avgs[i]), 3)
        
        plt.plot(pl_fpr, pl_tpr, label = "%s AUC: %s" % (model_list[i], pl_auc))

plt.grid(linewidth = 0.3)

plt.plot([0,1],[0,1], "--", alpha=0.3, color = "black")


plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Stacked Model Receiver Operating Characteristic")
plt.legend()

plt.savefig(out_path + "%s %s STACKED_FIG %s %s.png" % (fname_prefix, param_type, time_now, total_exp), dpi=300)


