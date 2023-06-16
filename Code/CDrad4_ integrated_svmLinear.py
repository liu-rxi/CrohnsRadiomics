
#from scipy import stats
import numpy as np
import pandas as pd
import time
from pathlib import Path
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

dat_path = 'J:\\Summer-Students\\Processed_data\\'
out_path = 'J:\\Summer-Students\\Output\\Integrated\\'

time_now = time.strftime("%Y_%m_%d-%H_%M")

##   load bowel project features
print('Load data')

param_type = '05mm'
total_exp = 400 # repitions for bootstrapping - use low val for now, then increase to 1k for publication

integration = ['wallCore', 'fatCore'] #'fat', 'wall'
fname_prefix = '+'.join(integration) #'LinearSVM'
abn_name = 'Crohns'
norm_name = 'Normal'

# feat_flags = list(["wall", "wallCore", "fat", "fatCore"])

# fname_prefix = 'Original' #'' #choose what to run + name the file
# fname_prefix = 'Revised' #''
# if fname_prefix == "Original":
#     abn_name = 'Abnormals'
#     norm_name = 'Normals'
# elif fname_prefix == "Revised":
#     abn_name = 'AbnormalsRevised'
#     norm_name = 'NormalsRevised'
# else:
#     print("mismatched fname_prefix option, placeholder, other options maybe")
    


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

# Group_type = 'Abnormals_new'
# Abnorm_wall_features_pd_2 = pd.read_pickle(dat_path + Group_type+'_wall_features.pkl')  
# Abnorm_wallCore_features_pd_2 = pd.read_pickle(dat_path + Group_type+'_wallCore_features.pkl') 
# Abnorm_fat_features_pd_2 = pd.read_pickle(dat_path + Group_type+'_fat_features.pkl')

## Keeping all outputs

# df_perf = pd.DataFrame(columns = [ #performane vars w label
#     'mask', 'auc', 'auc_sd', 'auc_95', 'ppv', 'ppv_sd', 'ppv_95', 'npv', 'npv_sd', 'npv_95',
#     'sens', 'sens_sd', 'sens_95', 'spec', 'spec_sd', 'spec_95', 'acc', 'acc_sd', 'acc_95'
#     ])
df_perf = pd.DataFrame(columns = [ #performance vars w label
    'mask', 'auc mean', 'auc_95', 'acc mean', 'acc_95', 'sens mean', 'sens_95', 
    'spec mean', 'spec_95', 'ppv mean', 'ppv_95', 'npv mean', 'npv_95'
    ])

df_feat = pd.DataFrame() # top 10 features

# df_perf = pd.DataFrame(columns = [
#     'mask', 'auc', 'auc_sd', 'auc_95', 'ppv', 'ppv_sd', 'ppv_95', 'npv', 'npv_sd', 'npv_95',
#     'sens', 'sens_sd, sens_95', 'spec', 'spec_sd', 'spec_95', 'acc', 'acc_sd', 'acc_95'
#     ])


# =============================================================================
#   Parameters for this run:
    # feat_flag = which masks (wall, wallCore, fat, fatCore)
    # feat_type = all features, or texture featuers only (all, texture)
# =============================================================================
#feat_flag = 'wall'   # 1: bowel wall feature/  2: wallCore  /  3:fat / 4: fatCore
# out_path = out_path + time_now + "_" + fname_prefix + "_" + param_type + "\\"


feature_norm = np.ndarray(shape=[Norm_wall_features_pd.shape[0],0])
feature_abnorm = np.ndarray(shape=[Abnorm_wall_features_pd_1.shape[0],0])
feature_names = []

#pool features together for variable selection
for feat_flag in integration:
    
    ##   get data and labels per these parameters;   Note: texture feature start at index 14 of 107
    if feat_flag== 'wall':    # whole bowel, use all features
            add_norm = Norm_wall_features_pd.to_numpy()
            add_abnorm = Abnorm_wall_features_pd_1.to_numpy()
            add_featname = Norm_wall_features_pd
    elif feat_flag== 'wallCore': #use only texture features
            add_norm = Norm_wallCore_features_pd.to_numpy()[:,14:]
            add_abnorm = Abnorm_wallCore_features_pd_1.to_numpy()[:,14:]     
            add_featname = list(Norm_wallCore_features_pd)[14:]
    elif feat_flag== 'fat': #use all features
            add_norm = Norm_fat_features_pd.to_numpy()
            add_abnorm = Abnorm_fat_features_pd_1.to_numpy()
            add_featname = list(Norm_fat_features_pd)  
    elif feat_flag== 'fatCore': #use only texture  features
            add_norm = Norm_fatCore_features_pd.to_numpy()[:,14:]
            add_abnorm = Abnorm_fatCore_features_pd_1.to_numpy()[:,14:]  
            add_featname = list(Norm_fatCore_features_pd)[14:]

    feature_norm  = np.concatenate((feature_norm, add_norm), axis = 1) #add features
    feature_abnorm  = np.concatenate((feature_abnorm, add_abnorm), axis = 1)
    feature_names = feature_names + list(add_featname)
        
##   format feature for sklearn models   start with normal
X = np.concatenate((feature_norm, feature_abnorm),axis=0)
y = np.concatenate((np.zeros(feature_norm.shape[0]), np.ones(feature_abnorm.shape[0])), axis=0)

feature_names = [sub.replace('original_', '') for sub in feature_names] 


#   repeat configuration
Perf_table= -100*np.ones([total_exp, 6])
feature_ranking = np.zeros(len(feature_names))
feature_selection_freq = np.zeros(len(feature_names))

for repeat_idx in range(total_exp): 
    
    print(repeat_idx)
    ##  single configuration
    n_fold = 5
    y_pred = -100*np.ones(len(y))
    y_score = -100*np.ones(len(y))
    
    #loo = LeaveOneOut()
    #kf = KFold(n_splits=n_fold)
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    CV_idx = 1
    
    print('Model training')
    #  K Fold CV  or LOOCV loop
    for i,j in skf.split(X, y):
    #for i,j in loo.split(X):
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    #    print('CV fold')
    #    print(CV_idx)
    #    print(i)
    #    print(j)
        X_train_raw = X[i,:]
        X_test_raw = X[j,:] 
        y_train = y[i]
        y_test = y[j]
        
        ## feature normalization
        scaler = preprocessing.StandardScaler().fit(X_train_raw)
        X_train_norm = scaler.transform(X_train_raw)
        X_test_norm = scaler.transform(X_test_raw)
        
        ##  feature selection 
        fs_flag = 1   #  set to 0 to disenable lasso   
        if fs_flag==1:  
            #  LASSO as LR
            lr = LogisticRegression(penalty = 'l1', solver='liblinear', max_iter=1000)
            # define grid
            para_grid={"C":  np.logspace(-3,3,7)} # l1 lasso   np.logspace(-5,5,7)   [0.01, 0.1, 1,2,3,4,5]
            # 5-fold cv
            lr_cv=GridSearchCV(lr, 
                               para_grid, 
                               scoring='accuracy',    #  scoring='roc_auc'
                               cv=n_fold)     
            
            lr_cv.fit(X_train_norm,y_train)    
            
            lasso = LogisticRegression(penalty = 'l1', solver='liblinear', 
                                       C= lr_cv.best_params_['C'])
            lasso.fit(X_train_norm,y_train)
            importance = np.abs(lasso.coef_)
            importance_idx = (importance>0)
                    
            idx_sf = np.where(importance_idx==1)[1]  
            X_train = np.take(X_train_norm, idx_sf, axis=1)
            X_test = np.take(X_test_norm, idx_sf, axis=1)       
            #   record lasso selection frequency    
            tmp_freq = np.zeros(len(feature_names))
            tmp_freq[idx_sf] = 1
            feature_selection_freq = feature_selection_freq + tmp_freq   
                
        else:
            X_train = X_train_norm
            X_test =  X_test_norm
            idx_sf = np.array(range(len(feature_names)))
        
        
        #  model optimization penalty parameter
        # Set the parameters by wallCore-validation
        tuned_parameters = [{'kernel': ['linear'], 'C':  np.logspace(-3,3,7) }]  #    [1,2,3,4,5,6]    # np.logspace(-3,3,7) 
        clf_cv = GridSearchCV(svm.SVC(), 
                              tuned_parameters, 
                              scoring='accuracy',    #  scoring='roc_auc'
                              cv=n_fold)  
        
        clf_cv.fit(X_train, y_train)
    #    print("Best parameters set found on development set:")
#        print(clf_cv.best_params_)    
        
        #  fit a SVM model
        clf=svm.SVC(kernel='linear',C= clf_cv.best_params_['C'])
        clf.fit(X_train,y_train)
        
        #  feature ranking
        tmp_ranking = np.zeros(len(feature_names))
        tmp_ranking[idx_sf] = np.abs(clf.coef_)/np.max(np.abs(clf.coef_))
        feature_ranking = feature_ranking + tmp_ranking
        
        #  make a classification
        y_score[j] = clf.decision_function(X_test)  
        y_pred[j] = clf.predict(X_test)
        
        CV_idx = CV_idx + 1
        #  end of CV loop
        
    #  calculate performance
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
    
    Perf_table[repeat_idx,:]=[Accuracy, Sensitivity, Specificity, PPV, NPV, roc_auc]
     

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
    '%s, %s' % (fname_prefix, param_type),
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
top_rank_features = np.take(feature_names, idx_top_rank)
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

df_perf.to_excel(out_path + "%s %s perf %s.xlsx" % (fname_prefix, param_type, time_now))
df_feat.to_excel(out_path + "%s %s features %s.xlsx" % (fname_prefix, param_type, time_now))

