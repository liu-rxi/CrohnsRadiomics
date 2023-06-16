## Prediction of Crohn's disease from radiomic features of bowel and mesenteric fat
This is the github repo containing code from the manuscript, **Machine Learning Diagnosis of Small Bowel Crohn’s Disease using T2-weighted MRI Radiomic and Clinical Data: A Pilot Study**. Raw images, radiomic features, and clinical data are omitted from the repository due to IRB constraints. Please contact the corresponding author, Dr. Jonathan Dillman, to discuss data sharing.

This project uses radiomic features from 4 regions of interest (ROI) from mesenteric fat and terminal ileum to classify Crohn's disease. The four regions are the whole bowel, bowel core, whole fat, and fat core, the details of which are described in the manuscript.

Radiomic features were extracted using pyradiomics, an open-source python package. https://pyradiomics.readthedocs.io/en/latest/

Models were trained using sklearn.

## Instructions
Code for all final clinical and radiomic machine learning pipelines and bowel thickness logistic regression are in /Code/. Clinical and radiomic pipelines are siloed and numbered in order of preprocessing. 
Raw data would be in /Data/ and radiomic features would be in /Processed_data/, but are not available currently. 

## License
GNU GPL 3.0, see COPYING
