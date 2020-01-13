# smrtgrid_federated_learning

1. First make sure each meter data are available in individual csv files inside the **data** folder. Unzip the **meter_load_half_hr.zip** file inside the **data** folder.

2. Run *data_clustering.py* file first to generate the group sequence data. Make sure set the parameters for how many groups are needed.

3. To perform individual group prediction, run the *baseline_prediction.py* file by setting the group number.

4. To perform federated learning, run the *federated_prediction* file.

