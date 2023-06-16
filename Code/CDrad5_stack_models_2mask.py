# =============================================================================
# Written for pair of independent feature selection --> SVM w all currently
# =============================================================================

#from scipy import stats
import numpy as np
import pandas as pd
import time
from pathlib import Path
import itertools
import os
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn import svm
#from scipy.misc import comb
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
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

def eval_perf(y, y_pred, y_score):
    CM = confusion_matrix(y, y_pred)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]

    Population = TN+FN+TP+FP
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
out_path = 'J:\\Summer-Students\\Output\\Stack, 2 masks\\'
clin_path = 'J:\\Summer-Students\\Processed_data\\clinical_data_includeNormal_imputed.xlsx'

time_now = time.strftime("%Y_%m_%d-%H_%M")


param_type = '02mm'
total_exp = 1000 # repitions for bootstrapping - use low val for now, then increase to 1k for publication

abn_name = 'Crohns'
norm_name = 'Normal'

# test all possible pairs
feat_flags = list(["wall", "wallCore", "fat", "fatCore"])
to_mix = list(itertools.combinations(feat_flags, 2)) #all pairwise combinations
#mix = to_mix[3]
mix = ["wallCore", "fatCore"]



# note - currently there is no explicit matching of ID - it's all implicit positional
# if there are significant changes to the order of subjects, ensure that the
# end result of (norm + abnorm) subjects lines up with the above clinical data file
df_clin = pd.read_excel(clin_path)
df_clin.female = df_clin.female.astype(int)
clin_colns = ['age','female','BMI_z','CRP','ESR','fecal_calprotectin']
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



df_perf = pd.DataFrame(columns = [ #performance vars w label
    'mask', 'auc mean', 'auc_95', 'acc mean', 'acc_95', 'sens mean', 'sens_95', 
    'spec mean', 'spec_95', 'ppv mean', 'ppv_95', 'npv mean', 'npv_95'
    ])

df_feat = pd.DataFrame() # top 10 features



#for mix in to_mix:
fname_prefix = '+'.join(mix) #'LinearSVM'
print("Training " + fname_prefix)


# =============================================================================
#     Select appropriate features for given masks
# =============================================================================
### FOR MASK 1
##   get data and labels per these parameters;   Note: texture feature start at index 14 of 107
if mix[0] == 'wall':    # whole bowel, use all features
        feature_norm0 = Norm_wall_features_pd.to_numpy()
        feature_abnorm0 = Abnorm_wall_features_pd_1.to_numpy()
        feature_names0 = Norm_wall_features_pd
elif mix[0] == 'wallCore': #use only texture features
        feature_norm0 = Norm_wallCore_features_pd.to_numpy()[:,14:]
        feature_abnorm0 = Abnorm_wallCore_features_pd_1.to_numpy()[:,14:]     
        feature_names0 = list(Norm_wallCore_features_pd)[14:]
elif mix[0] == 'fat': #use all features
        feature_norm0 = Norm_fat_features_pd.to_numpy()
        feature_abnorm0 = Abnorm_fat_features_pd_1.to_numpy()
        feature_names0 = list(Norm_fat_features_pd)  
elif mix[0] == 'fatCore': #use only texture  features
        feature_norm0 = Norm_fatCore_features_pd.to_numpy()[:,14:]
        feature_abnorm0 = Abnorm_fatCore_features_pd_1.to_numpy()[:,14:]  
        feature_names0 = list(Norm_fatCore_features_pd)[14:]

### FOR MASK 2
if mix[1] == 'wall':    # whole bowel, use all features
        feature_norm1 = Norm_wall_features_pd.to_numpy()
        feature_abnorm1 = Abnorm_wall_features_pd_1.to_numpy()
        feature_names1 = Norm_wall_features_pd
elif mix[1] == 'wallCore': #use only texture features
        feature_norm1 = Norm_wallCore_features_pd.to_numpy()[:,14:]
        feature_abnorm1 = Abnorm_wallCore_features_pd_1.to_numpy()[:,14:]     
        feature_names1 = list(Norm_wallCore_features_pd)[14:]
elif mix[1] == 'fat': #use all features
        feature_norm1 = Norm_fat_features_pd.to_numpy()
        feature_abnorm1 = Abnorm_fat_features_pd_1.to_numpy()
        feature_names1 = list(Norm_fat_features_pd)  
elif mix[1] == 'fatCore': #use only texture  features
        feature_norm1 = Norm_fatCore_features_pd.to_numpy()[:,14:]
        feature_abnorm1 = Abnorm_fatCore_features_pd_1.to_numpy()[:,14:]  
        feature_names1 = list(Norm_fatCore_features_pd)[14:]

##   format feature for sklearn models   start with normal
x0 = np.concatenate((feature_norm0, feature_abnorm0),axis=0)

x1 = np.concatenate((feature_norm1, feature_abnorm1),axis=0)
y = np.concatenate((np.zeros(feature_norm0.shape[0]), np.ones(feature_abnorm0.shape[0])), axis=0)

feature_names0 = [sub.replace('original_', '') for sub in feature_names0] 
feature_names1 = [sub.replace('original_', '') for sub in feature_names1] 


#   repeat configuration - list of Perf_tables, for 5 models - 3 layer1, 2 layer2
#  models to track being concurrently trained in each loop
model_list = [mix[0], mix[1], "clinical", "Meta, score", "Meta, aug"]
Perf_tables = [-100*np.ones([total_exp, 6]) for i in range(len(model_list))]

misclassified_i = [np.zeros(len(y)) for i in range(len(model_list))]

feature_ranking = np.zeros(len(feature_names0)+len(feature_names1))
feature_selection_freq = np.zeros(len(feature_names0)+len(feature_names1))

# Perf_table1= -100*np.ones([total_exp, 6])
# feature_ranking1 = np.zeros(len(feature_names1))
# feature_selection_freq1 = np.zeros(len(feature_names1))


# =============================================================================
#   For (total_exp) number of repititions, 5-f with holdout (train, test)
#       On train, for all individual models (...)
#           5-f feature selection, then
#           5-f model tuning, then
#           5-f meta model
#       Then, evaluation on test
# =============================================================================

for repeat_idx in range(total_exp): 
    
    print(repeat_idx)
    ##  single configuration
    n_fold = 5
    y0_pred = -100*np.ones(len(y))
    y0_score = -100*np.ones(len(y))
    
    y1_pred = -100*np.ones(len(y))
    y1_score = -100*np.ones(len(y))
    
    clin_pred = -100*np.ones(len(y))
    clin_score = -100*np.ones(len(y))
    
    metaPred_pred = -100*np.ones(len(y))
    metaPred_score = -100*np.ones(len(y))
    
    metaAug_pred = -100*np.ones(len(y))
    metaAug_score = -100*np.ones(len(y))
    
    #loo = LeaveOneOut()
    #kf = KFold(n_splits=n_fold)
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    CV_idx = 1
    
    print('Model training')

    #  K Fold CV  or LOOCV loop
    for i,j in skf.split(x0, y):

        x0_train_raw = x0[i,:] #of mask 1
        x0_test_raw = x0[j,:] 
        x1_train_raw = x1[i,:] #of mask 2
        x1_test_raw = x1[j,:] 
        clin_train = clin_features[i, :]
        clin_test = clin_features[j, :]
        y_train = y[i]
        y_test = y[j]
        
        ## feature normalization - for 2svms, not clinical data
        scaler0 = preprocessing.StandardScaler().fit(x0_train_raw)
        x0_train_norm = scaler0.transform(x0_train_raw)
        x0_test_norm = scaler0.transform(x0_test_raw)
        
        scaler1 = preprocessing.StandardScaler().fit(x1_train_raw)
        x1_train_norm = scaler1.transform(x1_train_raw)
        x1_test_norm = scaler1.transform(x1_test_raw)
        
        scaler2 = preprocessing.StandardScaler().fit(clin_train)
        clin_train = scaler2.transform(clin_train)
        clin_test = scaler2.transform(clin_test)
        
# =============================================================================
#         Feature selection         
# =============================================================================
        fs_flag = 1   #  set to 0 to disenable lasso   
        if fs_flag==1:  
            
            ## 5-fold cv for mask 1
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
            importance0 = np.abs(lasso0.coef_)
            importance_idx0 = (importance0>0)
            idx_sf0 = np.where(importance_idx0==1)[1]
            
            x0_train = np.take(x0_train_norm, idx_sf0, axis=1)
            x0_test = np.take(x0_test_norm, idx_sf0, axis=1)       
            
            #   record lasso selection frequency    
            tmp_freq0 = np.zeros(len(feature_names0))
            tmp_freq0[idx_sf0] = 1
            
            
            ## 5-fold cv for mask 2
            para_grid={"C":  np.logspace(-3,3,7)} 
            lr = LogisticRegression(penalty = 'l1', solver='liblinear', max_iter=1000)
            lr_cv=GridSearchCV(lr, 
                               para_grid, 
                               scoring='accuracy',    #  scoring='roc_auc'
                               cv=n_fold)  
            lr_cv.fit(x1_train_norm, y_train)    
            
            lasso1 = LogisticRegression(penalty = 'l1', solver='liblinear', 
                                       C= lr_cv.best_params_['C'])
            lasso1.fit(x1_train_norm, y_train)
            importance1 = np.abs(lasso1.coef_)
            importance_idx1 = (importance1>0)
            idx_sf1 = np.where(importance_idx1==1)[1]
            
            x1_train = np.take(x1_train_norm, idx_sf1, axis=1)
            x1_test = np.take(x1_test_norm, idx_sf1, axis=1)
            
            #   record lasso selection frequency    
 
            
            tmp_freq1 = np.zeros(len(feature_names1))
            tmp_freq1[idx_sf1] = 1
            
            tmp_freq = np.concatenate((tmp_freq0, tmp_freq1))
            feature_selection_freq = feature_selection_freq + tmp_freq   
            
            idx_sf = np.concatenate((idx_sf0, idx_sf1 + len(feature_names0)))
                
        else:
            print("FEATURE SELECTION REQUIRED - ABORT")
            exit()
            # X_train = X_train_norm
            # X_test =  X_test_norm
            # idx_sf = np.array(range(len(feature_names)))
        
        

    # =============================================================================
    #   Tune parameters of models, then fit model to training data
    #       For all models
    #   Then, train stacking/meta-classifier
    #   Finally, evaluate performance of all single models + meta
    # =============================================================================
    
        scores_train = []
        
        ## model 0: clf0
        tuned_parameters = [{'kernel': ['linear'], 'C':  np.logspace(-3,3,7) }] 
        clf0_cv = GridSearchCV(svm.SVC(), 
                              tuned_parameters, 
                              scoring='accuracy',    #  scoring='roc_auc'
                              cv=n_fold)  
        clf0_cv.fit(x0_train, y_train)

        clf0 = svm.SVC(kernel='linear',C= clf0_cv.best_params_['C'])
        clf0.fit(x0_train, y_train)
        
           
        ## model 1: clf1
        tuned_parameters = [{'kernel': ['linear'], 'C':  np.logspace(-3,3,7) }] 
        clf1_cv = GridSearchCV(svm.SVC(), 
                              tuned_parameters, 
                              scoring='accuracy',    #  scoring='roc_auc'
                              cv=n_fold)  
        clf1_cv.fit(x1_train, y_train)

        clf1 = svm.SVC(kernel='linear',C= clf1_cv.best_params_['C'])
        clf1.fit(x1_train, y_train)

        
        ## model 2: clinical - l2 logistic regression
        tuned_parameters = [{'C':  np.logspace(-2,2,5) }]
        lr_clin = LogisticRegression()
        lr_clin_cv = GridSearchCV(lr_clin, tuned_parameters, cv=n_fold)
        lr_clin_cv.fit(clin_train, y_train)
        
        clf_clin = LogisticRegression(
            solver='liblinear', 
            penalty = 'l2', 
            C = lr_clin_cv.best_params_["C"])
        clf_clin.fit(clin_train, y_train)

    # =============================================================================
    #   Aggregate models: meta-classifier - scores only, and feed-forward/augmented
    # =============================================================================

        #elastic net? saga, elasticnet, l1_ratio 
        #====== placeholder for code ======
        
        #mix data into meta model
        x_pred_train = np.column_stack((
            clf0.decision_function(x0_train), 
            clf1.decision_function(x1_train), 
            clf_clin.decision_function(clin_train))
            )
        x_aug_train = np.concatenate((
            x_pred_train, x0_train, x1_train, clin_train), axis=1)
       
        # l2 - predictions only 
        tuned_parameters = {'C':  np.logspace(-2,2,5)}
        lr_meta_cv = GridSearchCV(LogisticRegression(max_iter=1000),
                                  tuned_parameters,
                                  cv=n_fold)
        lr_meta_cv.fit(x_pred_train, y_train)
        use_c = lr_meta_cv.best_params_["C"]
        #use_c = 1
        # print("meta pred c: ", str(use_c))

        clf_meta_pred = LogisticRegression(
            solver='liblinear', penalty = 'l2', C= use_c, max_iter=1000)
        clf_meta_pred.fit(x_pred_train, y_train)
        
        
        # l2 - feed forward and augment
 
        tuned_parameters = {'C':  np.logspace(-2,2,5)}
        lr_meta_cv = GridSearchCV(LogisticRegression(max_iter=1000),
                                  tuned_parameters,
                                  cv=n_fold)
        lr_meta_cv.fit(x_aug_train, y_train)
        use_c = lr_meta_cv.best_params_["C"]
        #use_c = 0.01
        # print("meta aug c: ", str(use_c))
        clf_meta_aug = LogisticRegression(
            solver='liblinear', penalty = 'l2', C= use_c, max_iter=5000)
        clf_meta_aug.fit(x_aug_train, y_train)
     
        

    # =============================================================================
    #   Evaluate perf
    # =============================================================================

        
        y0_score[j] = clf0.decision_function(x0_test)  
        y0_pred[j] = clf0.predict(x0_test)
        y1_score[j] = clf1.decision_function(x1_test)  
        y1_pred[j] = clf1.predict(x1_test)
        clin_score[j] = clf_clin.decision_function(clin_test)  
        clin_pred[j] = clf_clin.predict(clin_test)
        
        ## Scale predictions and performances, then use as feature
        x_pred_test = np.column_stack((y0_score[j], y1_score[j], clin_score[j]))
        x_aug_test = np.concatenate((x_pred_test, x0_test, x1_test, clin_test), axis=1)
       
        metaPred_score[j] = clf_meta_pred.decision_function(x_pred_test)
        metaPred_pred[j] = clf_meta_pred.predict(x_pred_test)
        metaAug_score[j] = clf_meta_aug.decision_function(x_aug_test)
        metaAug_pred[j] = clf_meta_aug.predict(x_aug_test)
        
        # #  feature ranking
        # tmp_ranking = np.zeros(len(feature_names0) + len(feature_names1))
        # tmp_ranking[idx_sf] = np.abs(clf.coef_)/np.max(np.abs(clf.coef_))
        # feature_ranking = feature_ranking + tmp_ranking
        
        CV_idx = CV_idx + 1
        #  end of CV loop
    
    Perf_tables[0][repeat_idx,:] = eval_perf(y, y0_pred, y0_score)
    Perf_tables[1][repeat_idx,:] = eval_perf(y, y1_pred, y1_pred)
    Perf_tables[2][repeat_idx,:] = eval_perf(y, clin_pred, clin_score)
    Perf_tables[3][repeat_idx,:] = eval_perf(y, metaPred_pred, metaPred_score)
    Perf_tables[4][repeat_idx,:] = eval_perf(y, metaAug_pred, metaAug_score)

    misclassified_i[0][y0_pred != y] += 1
    misclassified_i[1][y1_pred != y] += 1
    misclassified_i[2][clin_pred != y] += 1
    misclassified_i[3][metaPred_pred != y] += 1
    misclassified_i[4][metaAug_pred != y] += 1
    
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

pt_names = os.listdir("J:\\Summer-Students\\new_normal\\") + os.listdir("J:\\Summer-Students\\new_cd")

for i in range(len(model_list)):
    Perf_table = Perf_tables[i]
    
    perf_mean = np.mean(Perf_table,axis=0)
    # perf_std = np.std(Perf_table,axis=0)
    
    
    # print(perf_mean)
    # print(perf_std) 
    perf_median = pd.DataFrame(Perf_table).apply(np.median, axis = 0)
    perf_ci_low = pd.DataFrame(Perf_table).apply(lambda x: np.percentile(x, 2.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    perf_ci_high = pd.DataFrame(Perf_table).apply(lambda x: np.percentile(x, 97.5), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    

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
    idx_top_rank = feature_ranking.argsort()[::-1][:10]
    top_rank_features = np.take(feature_names0 + feature_names1, idx_top_rank)
    top_rank_features_weight = np.take(feature_ranking, idx_top_rank)
    top_rank = pd.DataFrame(
        zip( top_rank_features, top_rank_features_weight),
        columns=['%s_%s_feat' % (fname_prefix, param_type), '%s_%s_wt' % (fname_prefix, param_type)])
    
    if not df_feat.empty:
        df_feat = pd.concat([df_feat, top_rank], axis=1)
    else:
        df_feat = top_rank
    
    #write final tables to pkl
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    df_perf.to_excel(out_path + "%s %s perf %s %s.xlsx" % (fname_prefix, param_type, time_now, total_exp))
    df_feat.to_excel(out_path + "%s %s features %s %s.xlsx" % (fname_prefix, param_type, time_now, total_exp))
    

    # =============================================================================
    # Output misclassifications
    # =============================================================================
    misclassified_perc = 100*misclassified_i[i]/total_exp
    misclas_df = pd.DataFrame({
        "pt_id": pt_names, 
        "error_percent": misclassified_perc})
    
    misclas_df.to_excel(out_path + "%s %s miss_rate %s %s %s.xlsx" % (fname_prefix, param_type, model_list[i], time_now, total_exp))

