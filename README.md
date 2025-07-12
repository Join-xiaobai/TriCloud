# TriCloud
This is a project about TriCloud model and its related data set files.

1. Data set processing The folder contains three subfolders
   1. The original data set and its processing is about the processing of the most original dataset of Stanford dataset, by integrating these binary relationships two by two, we finally get the dataset of drug-target-disease ternary relationship.
      data_reprocess.py : Convert binary relationships into ternary relationships (drug-target-disease)
      disease_to_id.json: Disease name to ID mapping dictionary
      drug_to_id.json: drug name to ID mapping dictionary
      gene_to_id.json: target name to ID mapping dictionary
      drug_gene_disease.txt: final drug-target-disease ternary relationship (ID number)
   2. Drug-target-disease three-dimensional coordinates
      drug_gene_disease_coordinate.txt: Results after adding column names
   3. Initialize the normal vector of 3D coordinates and its normalization result
      This folder contains operations such as generating normal vector features for positive/negative samples, normalizing the coordinates and their normal vectors, and adding labeled columns.
2. TriCloud code (normal vector features)
   This folder is for model training using normal vectors as features (spatial geometric features)
   To start the model set the parameters in the main file and run it.
   The test results for this experiment are in the TriCloud/test folder
3. TriCloud code (semantic features)
   This folder is a version of TriCloud trained with semantic features
   TriCloud_feature: semantic feature generation architecture code
      1.feature_train.py: Feature framework pre-training
      2.feature_extract.py: Feature extraction using the trained framework
      3.feature_id_index.py: Mapping drug, target, and disease names to ID numbers based on a dictionary
   TriCloud_train: Training code for the model
   
