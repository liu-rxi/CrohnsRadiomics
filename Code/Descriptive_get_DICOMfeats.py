import pydicom
from pydicom.filereader import read_dicomdir
import numpy as np
import pandas as pd
import nrrd
from matplotlib import pyplot as plt
import os
from pathlib import Path
import glob
# =============================================================================
# Extract DICOM characteristics for all subdirs in given list of folders
#       Based on name matching with .nrrd (nonseg) file in the subject's dir
#       Write matrix of features to processed_data
#       Write summary table of features to output (median, IQR)
# =============================================================================


dat_path = 'J:\\Summer-Students\\Processed_data\\'
out_path = 'J:\\Summer-Students\\Output\\'


Norm_path = "J:\\Summer-Students\\new_normal"
Abnorm_path = "J:\\Summer-Students\\new_cd"

# Norm_path = "J:\\Summer-Students\\Normal revised"
# Abnorm_path = "J:\\Summer-Students\\CD+ revised"

## Get DICOM vars from both norm_path + abnorm_path
run_dirs = [Norm_path, Abnorm_path]

cols = ['EchoTime', 'MagneticFieldStrength', 'RepetitionTime', 'SliceThickness',
        'NumberOfAverages', 'AcquisitionMatrix', 'Manufacturer', 'ManufacturerModelName']
attrib_df = pd.DataFrame(columns = cols + ['vox_x', 'vox_y', 'vox_z'])
i = 0
ids = []

for Data_path in run_dirs:
            
    print("Get DICOM features: ", Data_path)
    
    Dir_arr = os.listdir(Data_path)
    ids = ids + Dir_arr
    #Dir_arr = Dir_arr[0:1]   #  for coding one subject
    for sub_folder in Dir_arr:
        print(sub_folder)
        Sub_path = Data_path + '\\' + sub_folder
        sub_dir_arr = os.listdir(Sub_path)
        
# =============================================================================
#  find .nrrd file image and mask, then get voxel spacing
# =============================================================================
        
        matches_nrrd = []    
        for match in sub_dir_arr:
            if ".nrrd" in match and not ".seg." in match:
                matches_nrrd.append(match) 
        
        if len(matches_nrrd) != 1:
            print("more than 1 .nrrd file in ", sub_folder)
            
        nrrd_fname = matches_nrrd[0]
        _, img_header = nrrd.read(Sub_path + "\\" +  nrrd_fname)

        data_spacing = [img_header['space directions'][0,0], 
                        img_header['space directions'][1,1], 
                        img_header['space directions'][2,2]]
        attrib_df.loc[i, 'vox_x'] = data_spacing[0]
        attrib_df.loc[i, 'vox_y'] = data_spacing[1]
        attrib_df.loc[i, 'vox_z'] = data_spacing[2]


# =============================================================================
# find DICOM directory corresponding to nrrd, and read header vars
# =============================================================================
        
        nrrd_str = nrrd_fname.split(".")[0].split("_")[0]
        nrrd_str1 = nrrd_str.replace(" ", "-", 1).replace(" ", "_")
        nrrd_str2 = nrrd_str.replace(" ", "_")
        
        # check images directory
        dcm_path = ""
        for fpath in glob.iglob(Sub_path + '/**/', recursive=True):
            if nrrd_str1 in fpath or nrrd_str2 in fpath:
                dcm_path = fpath
        
        if dcm_path == "":
            print("No matching DCM dir in ", sub_folder, " matching ", nrrd_str)
        
        #take first image from dcm dir and read
        ds = pydicom.dcmread(dcm_path + os.listdir(dcm_path)[0])
        
        # research scanners are Philips, but don't have the Manufacturer tag 
        for coln in cols:
            if coln == "Manufacturer" and '-318H-' not in sub_folder:
                attrib_df.loc[i, coln]= 'Philips'
            else:
                try:
                    attrib_df.loc[i, coln] = ds[coln].value
                except KeyError:
                    print("KeyError: ", coln, ", None assigned")
                    attrib_df.loc[i, coln] = np.nan
                except AttributeError:
                    print("AttributeError: ", coln, ", None assigned")
                    attrib_df.loc[i, coln] = np.nan
        # ds.EchoTime #ms
        # ds.MagneticFieldStrength
        # ds.RepetitionTime
        # ds.SliceThickness
        # ds.NumberOfAverages
        # ds.AcquisitionMatrix #frequency rows\frequency columns\phase rows\phase columns.
        # ds.Manufacturer
        # #     #frequency = x?
        # #     #phase = y?

        # ds[0x0008, 0x0070]

      
        # np.save(Sub_path+'\masks\processed_data.npy', processed_data_np)
        i = i+1
    #    processed_data_load = np.load(Sub_path+'\processed_data.npy')


#Manually pulled from PACS
attrib_df.loc[84, 'EchoTime'] = 80
attrib_df.loc[84, 'MagneticFieldStrength'] = 1.5
attrib_df.loc[84, 'RepetitionTime'] = 800
attrib_df.loc[84, 'SliceThickness'] = 5
attrib_df.loc[84, 'NumberOfAverages'] = 1
attrib_df.loc[84, 'AcquisitionMatrix'] = '[0, 216, 215, 0]'
attrib_df.loc[84, 'Manufacturer'] = 'Philips'


attrib_df.loc[:, "id"] = ids
attrib_df.loc[attrib_df.Manufacturer == "GE MEDICAL SYSTEMS", "Manufacturer"] = "GE"
attrib_df.loc[attrib_df.Manufacturer == "Philips Medical Systems", "Manufacturer"] = "Philips"
attrib_df.loc[attrib_df.Manufacturer == "Philips Healthcare", "Manufacturer"] = "Philips"
attrib_df.Manufacturer.value_counts()
attrib_df.MagneticFieldStrength.value_counts()


attrib_df.to_excel(dat_path + "135 MRI characteristics full.xlsx")

# 

def get_mediqr_str(a):
    p25 = np.percentile(a, 25)
    p75 = np.percentile(a, 75)
    med = np.percentile(a, 50)
    
    return("%0.2f (%0.2f-%0.2f)"%(med, p25, p75))

def get_range_str(a):
    med = np.percentile(a, 50) 
    return("%0.2f (%0.2f-%0.2f)"%(med, np.min(a), np.max(a)))

quant_cols = ['EchoTime', 'RepetitionTime', 'SliceThickness', 'NumberOfAverages', 'vox_x', 'vox_y', 'vox_z']

for coln in quant_cols:
    print(coln + ": " + get_mediqr_str(attrib_df.loc[:, coln]))
    
             
for coln in quant_cols:
    print(coln + ": " + get_range_str(attrib_df.loc[:, coln]))
    