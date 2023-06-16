# -*- coding: utf-8 -*-

##   install pyradiomics  ---  pip install pyradiomics

import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from collections import OrderedDict
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import nrrd
import os
import SimpleITK as sitk
#import radiomics
from radiomics import featureextractor  #, getFeatureClasses
from pathlib import Path



# Norm_path = os.getcwd() +'\\Data\\Normal' #  '\\Data\\Normal'
# Abnorm_path = os.getcwd() +'\\Data\\Abnormal'
#Abnorm_path_new = '\\Data\\Bowel Abnormals new'


out_path = 'J:\\Summer-Students\\Processed_data\\'
# Norm_path = "J:\\Summer-Students\\Normal"
# Abnorm_path = "J:\\Summer-Students\\CD+"
Norm_path = "J:\\Summer-Students\\new_normal"
Abnorm_path = "J:\\Summer-Students\\new_cd"

##CHOOSE A PARAMETER FILE
# param_fname = 'Params_CD6.yaml'
# param_fname = 'Params_CD7.yaml'
# param_fname = 'Params_CD8.yaml'
# param_fname = 'Params_CD9.yaml'
param_dir = 'J:\\Summer-Projects-Codes\\Settings\\CD_norm\\' 
param_fname = '09mm.yaml'

param_root = param_fname.split(".")[0]

##CHOOSE A GROUP
# group_flag = 1      ##    0: normal;  1: abnormal;  2: new added abnormals
groups = [0,1] #run both

for group_flag in groups:
    if group_flag ==0:
        Data_path = Norm_path
        Group_type = 'Normal'
        # Group_type = 'Normals'
    
    elif group_flag ==1:
        Data_path = Abnorm_path
        Group_type = 'Crohns'
        # Group_type = 'Abnormals'
    
        
        
    Dir_arr = os.listdir(Data_path)
    #Dir_arr = Dir_arr[0:3]   #  for coding one subject
    
    
    # prepare the settings and initialize a radiomics extractor
    
    print("Extract pyrad features from: ", Data_path, ", using: ", param_fname)
    
    params = param_dir + param_fname
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    
    All_wall_features = {}
    All_wallCore_features = {}
    All_fat_features = {}
    All_fatCore_features = {}
    
    
    
    for sub_folder in Dir_arr:         #   loop over all subjects in normal/abnormal groups                 
        print('Subject:'+sub_folder)
        Sub_path = Data_path + '\\' + sub_folder + "\\"  
        processed_data = np.load(Sub_path+'masks\\processed_data.npy')
    
        sub_wall_features = {}     #  combine features from two slices
        sub_wallCore_features = {}
        sub_fat_features = {}
        sub_fatCore_features = {}
        
        # get spacing dimensinos from original nrrd, not .seg.nrrd
        for file in os.listdir(Sub_path):
            if file.endswith(".nrrd") and (".seg." not in file):
                img_data, img_header = nrrd.read(Sub_path + file)
    
                # data_spacing = [0.7, 0.7, 5]
                data_spacing = [img_header['space directions'][0,0], 
                                img_header['space directions'][1,1], 
                                img_header['space directions'][2,2]] 
    
        ##  iterate for each slice of the subject
        for slice_i in list([0,5]):   #  loop over two slices, stored in 0,:,: and 5,:,: of processed_data
            ##    get original img slice
            img_slice = processed_data[slice_i,:,:]   #/np.max(processed_data[0,:,:])   #   normalize  ##
            sitk_img = sitk.GetImageFromArray(img_slice)
            sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_img = sitk.JoinSeries(sitk_img)        
            
            ##   get wall mask and extract features        
            mask_wall_slice = processed_data[slice_i+1,:,:]
            sitk_wall_mask = sitk.GetImageFromArray(mask_wall_slice)
            sitk_wall_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_wall_mask = sitk.JoinSeries(sitk_wall_mask)
            sitk_wall_mask = sitk.Cast(sitk_wall_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT
            mask_wall_features = extractor.execute(sitk_img, sitk_wall_mask) 
            #    shorten features and combine features from two slices        
            tmp_wall_OD = OrderedDict((k, mask_wall_features[k]) for k in mask_wall_features if 'diagnostics' not in k)
            if slice_i==0:
                sub_wall_features = tmp_wall_OD
            else:
                for key, value in sub_wall_features.items():
                    if key in tmp_wall_OD:
                        sub_wall_features[key] = [ 0.5*(value + tmp_wall_OD[key]) ]    #  mean features of two slice
    #                    sub_wall_features[key] = list([value, tmp_wall_OD[key]])    #  concatenate features of two slice           
    #        print(sub_wall_features['original_shape_VoxelVolume'])          
            
            
            ##   get wallCore-sectional small ROI mask and extract features
            mask_wallCore_slice = processed_data[slice_i+2,:,:]
            sitk_wallCore_mask = sitk.GetImageFromArray(mask_wallCore_slice)
            sitk_wallCore_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_wallCore_mask = sitk.JoinSeries(sitk_wallCore_mask)
            sitk_wallCore_mask = sitk.Cast(sitk_wallCore_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT
            mask_wallCore_features = extractor.execute(sitk_img, sitk_wallCore_mask) 
            #    shorten features and combine features from two slices        
            tmp_wallCore_OD = OrderedDict((k, mask_wallCore_features[k]) for k in mask_wallCore_features if 'diagnostics' not in k)
            if slice_i==0:
                sub_wallCore_features = tmp_wallCore_OD
            else:
                for key, value in sub_wallCore_features.items():
                    if key in tmp_wallCore_OD:
                        sub_wallCore_features[key] = [ 0.5*(value + tmp_wallCore_OD[key]) ]         #  mean features of two slice 
            
            
            ##   get fat mask and extract features
            mask_fat_slice = processed_data[slice_i+3,:,:]
            sitk_fat_mask = sitk.GetImageFromArray(mask_fat_slice)
            sitk_fat_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_fat_mask = sitk.JoinSeries(sitk_fat_mask)
            sitk_fat_mask = sitk.Cast(sitk_fat_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT
            mask_fat_features = extractor.execute(sitk_img, sitk_fat_mask)   
            #    shorten features and combine features from two slices        
            tmp_fat_OD = OrderedDict((k, mask_fat_features[k]) for k in mask_fat_features if 'diagnostics' not in k)
            if slice_i==0:
                sub_fat_features = tmp_fat_OD
            else:
                for key, value in sub_fat_features.items():
                    if key in tmp_fat_OD:
                        sub_fat_features[key] = [ 0.5*(value + tmp_fat_OD[key]) ]  
            
            
            ##   get fatCore mask and extract features
            mask_fatCore_slice = processed_data[slice_i+4,:,:]
            sitk_fatCore_mask = sitk.GetImageFromArray(mask_fatCore_slice)
            sitk_fatCore_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_fatCore_mask = sitk.JoinSeries(sitk_fatCore_mask)
            sitk_fatCore_mask = sitk.Cast(sitk_fatCore_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT
            mask_fatCore_features = extractor.execute(sitk_img, sitk_fatCore_mask)   
            #    shorten features and combine features from two slices        
            tmp_fatCore_OD = OrderedDict((k, mask_fatCore_features[k]) for k in mask_fatCore_features if 'diagnostics' not in k)
            if slice_i==0:
                sub_fatCore_features = tmp_fatCore_OD
            else:
                for key, value in sub_fatCore_features.items():
                    if key in tmp_fatCore_OD:
                        sub_fatCore_features[key] = [ 0.5*(value + tmp_fatCore_OD[key]) ]  
                            
                        
                        
        # end of slice loop
    
        ##    wall features for all subjects
        if not bool(All_wall_features):
            All_wall_features = sub_wall_features
        else:
            for key, value in All_wall_features.items():
                if key in sub_wall_features:
                    value.extend(sub_wall_features[key])
    
        ##    wallCore-sectional features for all subjects
        if not bool(All_wallCore_features):
            All_wallCore_features = sub_wallCore_features
        else:
            for key, value in All_wallCore_features.items():
                if key in sub_wallCore_features:
                    value.extend(sub_wallCore_features[key])
                    
        ##    fat features for all subjects
        if not bool(All_fat_features):
            All_fat_features = sub_fat_features
        else:
            for key, value in All_fat_features.items():
                if key in sub_fat_features:
                    value.extend(sub_fat_features[key]) 
    
        ##    fatCore features for all subjects
        if not bool(All_fatCore_features):
            All_fatCore_features = sub_fatCore_features
        else:
            for key, value in All_fatCore_features.items():
                if key in sub_fatCore_features:
                    value.extend(sub_fatCore_features[key]) 
                    
    # end of subject loop                    
    All_wall_features_df = pd.DataFrame(All_wall_features, columns= All_wall_features.keys()) 
    All_wallCore_features_df = pd.DataFrame(All_wallCore_features, columns= All_wallCore_features.keys()) 
    All_fat_features_df = pd.DataFrame(All_fat_features, columns= All_fat_features.keys()) 
    All_fatCore_features_df = pd.DataFrame(All_fatCore_features, columns= All_fatCore_features.keys()) 
    
    
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    All_wall_features_df.to_pickle(out_path + param_root + Group_type+'_wall_features.pkl')  
    All_wallCore_features_df.to_pickle(out_path + param_root + Group_type+'_wallCore_features.pkl') 
    All_fat_features_df.to_pickle(out_path + param_root + Group_type+'_fat_features.pkl') 
    All_fatCore_features_df.to_pickle(out_path + param_root + Group_type+'_fatCore_features.pkl') 
    
    #checking for complex numbers
    sum(np.iscomplex(np.array(All_wall_features_df)))
    sum(np.iscomplex(np.array(All_wallCore_features_df)))
    sum(np.iscomplex(np.array(All_fat_features_df)))
    sum(np.iscomplex(np.array(All_fatCore_features_df)))
    
        