# smrtgrid_federated_learning


## AMI-FML framework
This repository contains code for the following paper.

M. Biswal, ASM Tayeen, and S. Misra,**“AMI-FML: A Privacy-Preserving Federated Machine Learning Framework for AMI”**. *arXiv preprint arXiv:2109.05666. 2021 Sep 13*


### Dataset

The dataset for the experiments comes from the Commission for Energy Regulation (CER) in Ireland. Please download the dataset (six zip files) from the original source. Put the zip files under the folder structure: **CER_Electricity/CER_Electricity_Revised_March_2012**. Unzip the files into the folder structure **data/load**. Run the *data_preprocess.py* file to process the raw data and create load data for each meter in a separate csv file. 

### Instructions
1. First make sure each meter data are available in individual csv files inside the **data/meter_load_half_hr** folder. 

2. Run *data_clustering.py* file first to generate the group sequence data. Make sure to set the parameters: 
- number of groups/aggregators.
- number of smart meters (size) in each group.
- whether meters will be chosen randomly or in sorted order of its ids to form groups.  

3. To train the meters of a group individually without federated settings, run the *individual_baseline_prediction.py* file by setting the following parameters: 

- `gid`: group id number.
- `test_len`: the number of test data points (e.g. 30 days worth of data is equal to 30 * 48 = 1440 data points)
- `window_size`: input sequence length (the number of previous time steps required to do prediction of the next time step)
- `num_hidden_nodes`: the number of LSTM blocks per layer

4. To train the meters of a group with federated learning, run the *federated_prediction.py* file by setting the following parameters:
- `gid`: group id number.
- number of epochs for global and local models.
- whether to include all meters in a group as the participants of federated learning.

5. The results will be stored in the **results** folder.

