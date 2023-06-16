# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:16:25 2022

@author: iis
"""

import scipy.stats as st
import pandas as pd
from tableone import TableOne
from pygrowup import Calculator, helpers


dat_path = "J:\\Summer-Students\\Misc raw data\\CD100 key + epic.xlsx"
dat_path2 = "J:\\Summer-Students\\Misc raw data\\35subj clin data raw.xlsx"
out_data_path = 'J:\\Summer-Students\\Processed_data\\'
out_path = 'J:\\Summer-Students\\Output\\Clinical\\'


df_100 = pd.read_excel(dat_path, sheet_name="segmented subjects 100")
df_100['sex'] = pd.Categorical(df_100.sex)

# turn ages from str y/m to float
ages = pd.DataFrame.from_records(df_100.age.str.split(','))
for i in range(ages.shape[0]):
    ages[0][i] = ages[0][i][:-1]
    ages[1][i] = ages[1][i][:-1]

ages[0] = pd.to_numeric(ages[0])
ages[1] = pd.to_numeric(ages[1])
age = ages[0] + ages[1]/12
# age_y = int(df_100.age.str.split(',')[0][0][:-1])
# age_m = int(df_100.age.str.split(',')[0][1][:-1])
# age = age_y + age_m/12
df_100.age = age

# calculate BMI and get pediatric BMI percentile
df_100.BMI = df_100.wt/(df_100.ht/100)**2
df_100['female'] = df_100['sex'] == "Female"

# =============================================================================
# Read other dataset and merge, then calculate BMI z/perc together
# =============================================================================
colns = ['id', 'CD', 'female', 'age', 'race', 'ethnicity', 'wt', 'ht', 'BMI', 'BMI_z', 'BMI_perc',
    'CRP', 'ESR', 'fecal_calprotectin', 'fecal_lactoferrin']
df_35 = pd.read_excel(dat_path2)
df_35.columns = colns

df_100.loc[:, ['BMI_z', 'BMI_perc']] = -100
df_100 = df_100.loc[:, colns]

# order them such that it goes 35_norm, 100_norm, 35_cd, 100_cd
df_135 = pd.concat([
    df_35[df_35.CD == 0],
    df_100[df_100.CD==0],
    df_35[df_35.CD == 1],
    df_100[df_100.CD==1]
    ], ignore_index = True)
    
#df_135 = df_135.reset_index(level=0)
df_135['CD'] = pd.Categorical(df_135.CD)


calculator = Calculator(include_cdc=True)
dummy_sex = df_135.female.astype("string")
dummy_sex.loc[[not elem for elem in df_135.female]] = 'M'
dummy_sex.loc[df_135.female] = 'F'
dummy_sex = dummy_sex.astype("string").tolist()
df_135['BMI_z'] = len(df_135.index)*[None]

# =============================================================================
# If age is >20, impute as 20
# =============================================================================
for i in df_135.index:
    age_mo = df_135.age[i]*12
    if(df_135.loc[i, "age"] > 20):
        age_mo = 20*12
        
    df_135.loc[i, 'BMI_z'] = calculator.bmifa(
        measurement=df_135.BMI[i], age_in_months=age_mo, sex=helpers.get_good_sex(dummy_sex[i]))

df_135.BMI_z= df_135.BMI_z.astype(float)
df_135['BMI_perc'] = 100*st.norm.cdf(df_135.BMI_z)
## calculate BMI

# =============================================================================
# Table one written to output dir
# =============================================================================
## display data in table one and save
table_cols = [
    'age', 'female', 'race', 'ethnicity', 'wt', 'ht', 'BMI', 'BMI_perc', 'CRP', 'ESR', 
    'fecal_calprotectin', 'fecal_lactoferrin'
    ]

table_categorical = ['female', 'race', 'ethnicity']
# table_nonnormal = ['CRP', 'ESR', 'BMI_perc', 'fecal_calprotectin', 'fecal_lactoferrin']

table1 = TableOne(df_135, columns = table_cols, categorical = table_categorical,
                  groupby = "CD", #nonnormal = table_nonnormal,
                  pval = True, overall=False)



#print(table1.tabulate(tablefmt = 'github'))
table1.to_excel(out_path + 'CD135_table1_sd.xlsx')



# =============================================================================
# Finish preprocessing and write to \\Processed_data\\ 
# =============================================================================

df_135.to_excel(out_data_path + "clinical_data_135_extra.xlsx", index=False)
