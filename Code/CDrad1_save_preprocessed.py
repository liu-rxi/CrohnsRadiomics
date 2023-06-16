# -*- coding: utf-8 -*-



import numpy as np
import nrrd
from matplotlib import pyplot as plt
import os
from pathlib import Path
import pandas as pd

# =============================================================================
# Preprocessing:
#  given normal and CD dir paths, writes a processed_data.npy in each subject's
#  directory and 
# =============================================================================

""""Extend out_df with row of summary statistics from df"""
def med_iqr_range(dirname, label, df, out_df):
    
    perf_median = df.apply(np.median, axis = 0)
    perf_ci_low = pd.DataFrame(df).apply(lambda x: np.percentile(x, 25), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    perf_ci_high = pd.DataFrame(df).apply(lambda x: np.percentile(x, 75), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    perf_min = pd.DataFrame(df).apply(np.min, axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    perf_max = pd.DataFrame(df).apply(np.max, axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])

    out_df.loc[len(out_df.index), :] = [
        dirname, label,
        '%0.0f' % (perf_median[0]), '%0.0f-%0.0f' % (perf_ci_low[0], perf_ci_high[0]), '%0.0f-%0.0f' % (perf_min[0], perf_max[0]),
        '%0.0f' % (perf_median[1]), '%0.0f-%0.0f' % (perf_ci_low[1], perf_ci_high[1]), '%0.0f-%0.0f' % (perf_min[1], perf_max[1]),
        '%0.0f' % (perf_median[2]), '%0.0f-%0.0f' % (perf_ci_low[2], perf_ci_high[2]), '%0.0f-%0.0f' % (perf_min[2], perf_max[2]),
        '%0.0f' % (perf_median[3]), '%0.0f-%0.0f' % (perf_ci_low[3], perf_ci_high[3]), '%0.0f-%0.0f' % (perf_min[3], perf_max[3])
        ]
 
    return(out_df)
    

out_path = "J:\\Summer-Students\\Output\\"
Norm_path = "J:\\Summer-Students\\new_normal"
Abnorm_path = "J:\\Summer-Students\\new_cd"
## Preprocess both norm_path + abnorm_path
run_dirs = [Norm_path, Abnorm_path]

out_df = pd.DataFrame(columns = [
    'dirname', 'label', 'wall median', 'wall IQR', 'wall range',
    'wallCore median','wallCore IQR', 'wallCore range',
    'fat median','fat IQR', 'fat range',
    'fatCore median','fatCore IQR', 'fatCore range'])

for Data_path in run_dirs:
            
    print("Preprocess: ", Data_path)

    #report by normal/abnormal status    
    count_df = pd.DataFrame(columns = ['wall', 'wallCore', 'fat', 'fatCore'])
    space_df = pd.DataFrame(columns = ['wall', 'wallCore', 'fat', 'fatCore'])
    i=0

    Dir_arr = os.listdir(Data_path)
    #Dir_arr = Dir_arr[0:1]   #  for coding one subject
    
    for sub_folder in Dir_arr:
        print(sub_folder)
        Sub_path = Data_path + '\\' + sub_folder
        sub_dir_arr = os.listdir(Sub_path)
        
    #   find .nrrd file image and mask
        matches_nrrd = []    
        for match in sub_dir_arr:
            if ".nrrd" in match:
                matches_nrrd.append(match) 
    #    print(matches_nrrd)
                
    #   separate image and mask file       
        for match in matches_nrrd:
            if ".seg." in match:
                mask_file = match
            else:
                image_file = match
    
    # read image and mask files for each subject
        Img_path = Sub_path + '\\' + image_file        
        img_data, img_header = nrrd.read(Img_path)
        img_data = np.transpose(img_data,(1, 0, 2))

        data_spacing = [img_header['space directions'][0,0], 
                        img_header['space directions'][1,1], 
                        img_header['space directions'][2,2]] 
    
    ##     masks
        Mask_path = Sub_path + '\\' + mask_file
        mask_data, mask_header = nrrd.read(Mask_path)   
    
    ##    processed img data
        processed_data = []
         
    ##    reconstruct cropped mask            
        mask_offset = np.fromstring(mask_header['Segmentation_ReferenceImageExtentOffset'],dtype=int, sep=' ')
        x = mask_offset[0]
        y = mask_offset[1]
        z = mask_offset[2]
        
        mask_slice_offset = np.fromstring(mask_header['Segment0_Extent'],dtype=int, sep=' ')
        mask_slice_arr = [mask_slice_offset[4], mask_slice_offset[5]]
        
        
    ## confirm segment identity, then get layer + value
        
        # wall
        if mask_header['Segment0_Name'] != 'large':
            print("Warning: Segment0_Name not 'large', but rather: " + mask_header['Segment0_Name'])
        wall_layer = int(mask_header["Segment0_Layer"])
        wall_label = int(mask_header["Segment0_LabelValue"])
        
        if mask_header['Segment1_Name'] != 'small':
            print("Warning: Segment1_Name not 'small', but rather: " + mask_header['Segment1_Name'])
        wallCore_layer = int(mask_header["Segment1_Layer"])
        wallCore_label = int(mask_header["Segment1_LabelValue"])
        
        if mask_header['Segment2_Name'] != 'fat':
            print("Warning: Segment2_Name not 'fat', but rather: " + mask_header['Segment2_Name'])
        fat_layer = int(mask_header["Segment2_Layer"])
        fat_label = int(mask_header["Segment2_LabelValue"])
        
        if mask_header['Segment3_Name'] != 'fatsmall':
            print("Warning: Segment3_Name not 'fatsmall', but rather: " + mask_header['Segment3_Name'])
        fatCore_layer = int(mask_header["Segment3_Layer"])
        fatCore_label = int(mask_header["Segment3_LabelValue"])
        
        
        
        for mask_slice_idx in mask_slice_arr:
            mask_wall = np.zeros([img_data.shape[0],img_data.shape[1]])
            mask_wallCore = np.zeros([img_data.shape[0],img_data.shape[1]]) 
            mask_fat = np.zeros([img_data.shape[0],img_data.shape[1]]) 
            mask_fatCore = np.zeros([img_data.shape[0],img_data.shape[1]]) 
    
            
        ##  The 1 out of 2 slices for image/wall/wallCore/fat
            img_slice = img_data[:,:,z+mask_slice_idx]
            processed_data.append(img_slice)
         #    wall = 1 in set 0
            crop_wall = mask_data[wall_layer, :, : , mask_slice_idx]==wall_label
            mask_wall[x:x+crop_wall.shape[0], y:y+crop_wall.shape[1]] = crop_wall
            mask_wall = np.transpose(mask_wall,(1, 0))
            processed_data.append(mask_wall)
            overlap_wall = img_slice - img_slice*mask_wall
            
        #    wallCore section = 1 in set 1
            crop_wallCore = mask_data[wallCore_layer, :, :, mask_slice_idx]==wallCore_label
            mask_wallCore[x:x+crop_wallCore.shape[0], y:y+crop_wallCore.shape[1]] = crop_wallCore
            mask_wallCore = np.transpose(mask_wallCore,(1, 0))
            processed_data.append(mask_wallCore)
            overlap_wallCore = img_slice - img_slice*mask_wallCore
        #    fat    
            crop_fat = mask_data[fat_layer, :, :, mask_slice_idx]==fat_label
            mask_fat[x:x+crop_fat.shape[0], y:y+crop_fat.shape[1]] = crop_fat
            mask_fat = np.transpose(mask_fat,(1, 0))
            processed_data.append(mask_fat)
            overlap_fat = img_slice - img_slice*mask_fat  
          
            #    fatCore  
            crop_fatCore = mask_data[fatCore_layer, :, :, mask_slice_idx]==fatCore_label
            mask_fatCore[x:x+crop_fatCore.shape[0], y:y+crop_fatCore.shape[1]] = crop_fatCore
            mask_fatCore = np.transpose(mask_fatCore,(1, 0))
            processed_data.append(mask_fatCore)
            overlap_fatCore = img_slice - img_slice*mask_fatCore  
    #        fig, axs = plt.subplots(1,4, sharey=True)
    #        axs[0].imshow(img_slice, cmap='gray')
    #        axs[1].imshow(overlap_wall, cmap='gray')
    #        axs[2].imshow(overlap_wallCore, cmap='gray')
    #        axs[3].imshow(overlap_fat, cmap='gray')
    #        plt.show()
            
        # save images for each mask, and overall
            Path(Sub_path+'\\masks\\').mkdir(parents=True, exist_ok=True)
            plt.imsave(Sub_path+'\\masks\\'+str(mask_slice_idx)+'_img.jpeg', img_slice, cmap='gray')
            plt.imsave(Sub_path+'\\masks\\'+str(mask_slice_idx)+'_wall.jpeg', overlap_wall, cmap='gray')
            plt.imsave(Sub_path+'\\masks\\'+str(mask_slice_idx)+'_wallCore.jpeg', overlap_wallCore, cmap='gray')
            plt.imsave(Sub_path+'\\masks\\'+str(mask_slice_idx)+'_fat.jpeg', overlap_fat, cmap='gray')
            plt.imsave(Sub_path+'\\masks\\'+str(mask_slice_idx)+'_fatCore.jpeg', overlap_fatCore, cmap='gray')
    
        
        # save mask volumes + space 
            count_df.loc[i, 'wall'] = sum(sum(mask_wall))
            count_df.loc[i, 'wallCore'] = sum(sum(mask_wallCore))
            count_df.loc[i, 'fat'] = sum(sum(mask_fat))
            count_df.loc[i, 'fatCore'] = sum(sum(mask_fatCore))
            
            vox_vol = data_spacing[0]*data_spacing[1]
            space_df.loc[i, 'wall'] = sum(sum(mask_wall))*vox_vol
            space_df.loc[i, 'wallCore'] = sum(sum(mask_wallCore))*vox_vol
            space_df.loc[i, 'fat'] = sum(sum(mask_fat))*vox_vol
            space_df.loc[i, 'fatCore'] = sum(sum(mask_fatCore))*vox_vol

            i=i+1
    ##  save all image and mask in a single processed data file 
    
        processed_data_np = np.array(processed_data)
        #    plt.imshow(processed_data[1], cmap='gray') 
        #    plt.show()
        #
        #    plt.imshow(processed_data[5], cmap='gray') 
        #    plt.show()
        #    
        #    plt.imshow(processed_data[3], cmap='gray') 
        #    plt.show()
            
        np.save(Sub_path+'\masks\processed_data.npy', processed_data_np)
        #    processed_data_load = np.load(Sub_path+'\processed_data.npy')
            

 ## after a group is done, summarize mask sizes 
    out_df = med_iqr_range(Data_path, "mask_voxels", count_df, out_df)    
    out_df = med_iqr_range(Data_path, "mask_space", space_df, out_df)       

    # perf_median = count_df.apply(np.median, axis = 0)
    # perf_ci_low = pd.DataFrame(count_df).apply(lambda x: np.percentile(x, 25), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    # perf_ci_high = pd.DataFrame(count_df).apply(lambda x: np.percentile(x, 75), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    # perf_min = pd.DataFrame(count_df).apply(np.min, axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    # perf_max = pd.DataFrame(count_df).apply(np.max, axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])

    # out_df.loc[len(out_df.index), :] = [
    #     'Mask voxels: mean',
    #     '%0.0f' % (perf_median[0]),
    #     '%0.0f' % (perf_median[1]),
    #     '%0.0f' % (perf_median[2]),
    #     '%0.0f' % (perf_median[3]),
    #     ]
    # out_df.loc[len(out_df.index), :] = [
    #     'Mask voxels: IQR',
    #     '%0.0f-%0.0f' % (perf_ci_low[0], perf_ci_high[0]),
    #     '%0.0f-%0.0f' % (perf_ci_low[1], perf_ci_high[1]),
    #     '%0.0f-%0.0f' % (perf_ci_low[2], perf_ci_high[2]),
    #     '%0.0f-%0.0f' % (perf_ci_low[3], perf_ci_high[3]),
    #     ]
    # out_df.loc[len(out_df.index), :] = [
    #     'Mask voxels: range',
    #     '%0.0f-%0.0f' % (perf_min[0], perf_max[0]),
    #     '%0.0f-%0.0f' % (perf_min[1], perf_max[1]),
    #     '%0.0f-%0.0f' % (perf_min[2], perf_max[2]),
    #     '%0.0f-%0.0f' % (perf_min[3], perf_max[3]),
    #     ]
    
    # perf_median = space_df.apply(np.median, axis = 0)
    # perf_ci_low = pd.DataFrame(space_df).apply(lambda x: np.percentile(x, 25), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])
    # perf_ci_high = pd.DataFrame(space_df).apply(lambda x: np.percentile(x, 75), axis=0) #), axis = 0, args=2.5, kwargs='q')#args=len(Perf_table.index)*[2.5])

    # out_df.loc[len(out_df.index), :] = [
    #     'Mask space, mm2',
    #     '%0.0f (%0.0f-%0.0f)' % (perf_median[0], perf_ci_low[0], perf_ci_high[0]),
    #     '%0.0f (%0.0f-%0.0f)' % (perf_median[1], perf_ci_low[1], perf_ci_high[1]),
    #     '%0.0f (%0.0f-%0.0f)' % (perf_median[2], perf_ci_low[2], perf_ci_high[2]),
    #     '%0.0f (%0.0f-%0.0f)' % (perf_median[3], perf_ci_low[3], perf_ci_high[3]),
    #     ]
    
           

out_df.transpose().to_excel(out_path + "Mask size distrib.xlsx")
