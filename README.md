
# Temporal Abstraction and Deep Learning for Time Series Classification

The main goal of this system is to examine the effect of abstraction on sequential neural networks for time series classification.

The system contains:
 * Implementations of different kinds of sequential neural networks.
 * Code for adjusting the representation of the raw data to the HugoBot system, that performs the temporal abstraction.
 * Different transformations of the abstracted data to tensor representation.
 * Results plots.
## Data 
The data archives that are used in this project: 
* The [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018), which contains the 128 univariate time series datasets. 
* The [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html), which contains the 13 multivariate time series datasets.

## Code   
The code is divided as follows: 
* main_cpu.py: contains the necessary code for running the data creation for the HugoBot system and creating the different representations of the abstract data.

* main_gpu.py: contains the necessary code for running the DNN models.

* run_models.py: contains the code that is responsible for calling the relevant functions of creating the models, called by the main_gpu.py.

* utils_folder: contains the necessary functions to read the datasets, visualize the plots and set the parameters.

* classifiers folder: contains 11 python files, one for each deep neural network.

* temporal_abstraction_f: contains the code responsible for the transformation that converts the code to the HugoBot system and for the neural networks.

* HugoBot system: performs temporal abstraction.

## The flow
1. Create an anaconda env and install the relevant packages as described below. **Note:** To convert the MTS format from MATLAB  to np: Run the main_cpu.py file with the following parameters - "transform_mts_to_ucr_format".

2. **Create the data for the hugobot system on the CPU cluster:** Run the main_cpu.py with the following parameters - "create_files_for_hugobot {archive_name} {per_entity}"  
**Note** - The first run must be with per_entity= False

3. **Create the data for the DNN:** Run the main_cpu.py file with the following parameters -  
`create_files {archive_name} {classifier} {after_TA} {TA_method} {combination} {transformation_number = 1} {per_entity}` (Below there is an explanation on the parameters).

4. **Run the DNN**  **on the gpu cluster:** Run the main_gpu.py file with the following parameters - `run_all {archive_name} {classifier} {after_TA} {TA_method} {combination} {transformation_number} {per_entity}`.

5. **Evaluate the results:** Run cd-diagram_graphs.py file or graphs.py file.  
Parameters for the graphs.py:  
The main function is 'create_all_graphs'. The function receives the following params:

	* graph_numbers – list of the graphs you want to create.

	* create_csv (Boolean) – combining all the results of the DNN architectures.  **Note** – this param needs to be set to True only in the first run.

	* type – {archive_name}

	**Parameters for the cd-diagram_graphs.py:**  
	The main function is create_diagram_main the function receives the following params:
	* type – {archive_name}
    * metric- the evaluation metric

## Manual for working on the cluster

User guide for running the experiments described in our paper on the university's cluster.

**Installation**

The code runs on GPUs, to deal with the randomness effect of the GPU and the Python's packages - we used the [TensorFlow-determinism package](https://github.com/NVIDIA/framework-determinism).

The following packages were also used:
 * [numpy](http://www.numpy.org/)
 * [pandas](https://pandas.pydata.org/)
 * [sklearn](http://scikit-learn.org/stable/)
 * [scipy](https://www.scipy.org/)
 *  [matplotlib](https://matplotlib.org/)
 * [tensorflow-gpu](https://www.tensorflow.org/)
 * [keras](https://keras.io/)
 * [h5py](http://docs.h5py.org/en/latest/build.html)
 * [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

**For installing these packages, please follow the next steps**:

1. Make sure you are connected through VPN or from within the campus.

2. Download an SSH terminal (e.g., [Mobaxterm](https://mobaxterm.mobatek.net/download.html)).

3. Open the SSH terminal and start an SSH session to your user on the cpu manager node ip. Use the organization username and password.

4. Once logged into the cluster's manager node, create your Conda environment: `conda create -n my_env python=3.7`.

5. Then, activate the environment that we created:  `conda activate my_env`

6. To install packages:  
`pip install <whatever package you need>` or `conda install <whatever package you need>`  
For the 'tensorflow-gpu' package, please do not use pip install to install. Instead, use: `conda install -c anaconda tensorflow-gpu`

7. `conda deactivate`

8. Copy the sbatch file (job launching file) by typing (do not forget the dot at the end!):  
`cp /storage/sbatch_gpu.example .`

9. Edit the file using nano/vim editor: `nano sbatch_gpu.example`.

10. You may change the job name by replacing my_job with your own string.

11. Change the following line in the file: `source activate my_env`, replace 'my_env' with your environment name that you have created on paragraph 4.

12. 'jupyter lab' is the program to run on the compute node – it will start a jupyter program that you may use. You may use another command instead of 'jupyter lab', such as `'python my_script.py my_arg'`

13. Press <ctrl>+x, then 'y' and '<Enter>' to save and leave the file.

14. Launch a new job: `sbatch sbatch_gpu.example`

15. You should instantly get the job id.

16. To see the status of your job(s) type `squeue --me`.

17. Under 'ST' (state) column if the state is 'PD' then the job is pending. If the state is 'R', the job is running, and you can look at the output file for initial results (Jupiter results will take up to a minute to show): `less job-<job id>.out`.

18. If you asked for Jupiter, then copy the 2nd link (which starts with 'https://132.72.'). Copy the whole link, including the token, and paste it into the address bar of your web browser. Make the browser advance (twice) in spite of its warnings.

## Parameters

* **archive_name** string, one of: UCR, MTS.\

* **classifier** string - one of: fcn, mlp, resnet, mcnn, tlenet, twiesn, encoder, cnn, inception, lstm_fcn, mlstm_fcn.

* **after_TA** boolean - whether to perform temporal abstraction on the data.

* **TA_method** string, the options listed in the Hugobot system.

* **Combination** boolean, combining two STI representations - gradient + state abstraction.

* **Transformation_number** int, the transformation to tensor, one of:
	1. Discrete Transformation.
	2. Symbol One-Hot Transformation.
	3. Endpoint One-Hot Transformation.

* **per_entity** boolean, whether to perform the temporal abstraction per entity, which means to learn the cutoff per entity, so that each entity has different cutoffs.
