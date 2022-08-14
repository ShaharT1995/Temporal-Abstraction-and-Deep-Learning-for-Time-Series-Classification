
# Temporal Abstraction and Deep Learning for Time Series Classification

The main goal of this system is to examine the effect of abstraction on sequential neural networks for time series classification.

The system contains code for:
* Implement different kinds of sequential neural networks
 * Adjusting the representation of the raw data to the HugoBot system, which is the system where the TA process happens.
 * Different transformations of the abstract data to tensor representation.
 * Results plots
## Data 
The data used in this project comes from two sources: 
* The [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018), which contains the 128 univariate time series datasets. 
* The [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html), which contains the 13 multivariate time series datasets.

## Code   
The code is divided as follows: 
* The main_cpu.py python file contains the necessary code for running the data creation for the HugoBot system and creating the different representations of the abstract data.

* The main_gpu.py python file contains the necessary code for running the DNN models

* The run_models.py python file called by the main_gpu.py contains the code that is responsible for calling the relevant functions of creating the models

* The **utils_folder** contains the necessary functions to read the datasets, visualize the plots and set the parameters.

* The **classifiers folder** contains 11 python files, one for each deep neural network.

* The **temporal_abstraction_f** Contains the code responsible for the transformation that converts the code to the HugoBot system and for the neural networks.

* **HugoBot system** - performs the process of temporal abstraction

## The flow
1. Create an anaconda env and install the relevant packages as described below. **Note:** To convert the MTS format from MATLAB  to np: Run the main_cpu.py file with the following parameters - "transform_mts_to_ucr_format".

2. **Create the data for the hugobot system on the CPU cluster:** Run the main_cpu.py  with the following parameters - "create_files_for_hugobot {archive_name} {per_entity}"  
**Note** - The first run must be with per_entity= False

3. **Create the data for the DNN:** Run the main_cpu.py file with the following parameters -  
"create_files {archive_name} {after_TA} {TA_metod} {combination} {transformation_number = 1} {per_entity}"

4. **Running the DNN**  **on the gpu cluster:** Run the main_gpu.py file with the following parameters - "run_all {archive_name} {after_TA} {TA_metod} {combination} {transformation_number} {per_entity}"

5. **Evaluating the results:** Run cd-diagram_graphs.py file or graphs.py file.  
Parameters for the graphs.py:  
The main function is 'create_all_graphs'. The function receives the following params:

	* graph_numbers – list of the graphs you want to create.

	* create_csv (Boolean) – combining all the results of the DNN architectures.  **Note** – this param needs to be set to True only in the first run.

	* type – {archive_name}

**Parameters for the cd-diagram_graphs.py:**  
The main function is create_diagram_main the function receives the following params:

* type – {archive_name}

* metric- the evaluation metric

## Manual for running the experiments

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

3. Open the SSH terminal and start an SSH session (port 22). The username is your campus, and the password is your campus password.

4. Once logged into the cluster's manager node, create your Conda environment. E.g.:  `conda create -n my_env python=3.7`

5. Then, activate the environment that we created:  `conda activate my_env`

6. To install packages:  
`pip install <whatever package you need>` or `conda install <whatever package you need>`  
For the 'tensorflow-gpu' package, please do not use pip install to install. Instead, use: `conda install -c anaconda tensorflow-gpu`

7. `conda deactivate`

8. Copy the sbatch file (job launching file) by typing (do not forget the dot at the end!):  
`cp /storage/sbatch_gpu.example` .

9. Edit the file using nano editor: `nano sbatch_gpu.example`

10. You may change the job name by replacing my_job with your own string.

11. Go to the last lines of the file. `'source activate my_env'`: if needed, replace 'my_env' with your environment name that you have created on paragraph 4.

12. 'jupyter lab' is the program to run on the compute node – it will start a jupyter program that you may use. You may use another command instead of 'jupyter lab', such as `'python my_script.py my_arg'`

13. Press <ctrl>+x, then 'y' and '<Enter>' to save and leave the file.

14. Launch a new job: `sbatch sbatch_gpu.example`

15. You should instantly, get the job id.

16. To see the status of your job(s) type `squeue --me`

17. Under 'ST' (state) column if the state is 'PD' then the job is pending. If the the state is 'R', then the job is running, and you can look at the output file for initial results (Jupiter results will take up to a minute to show): `less job-<job id>.out`

18. If you asked for Jupiter, then copy the 2nd link (which starts with 'https://132.72.'). Copy the whole link, including the token, and paste it into the address bar of your web browser. Make the browser advance (twice) in spite of its warnings.

## Parameters

**For running the program, the following parameters should be given in the configuration:**

* **archive_name** – UCR/MTS

* **after_TA –** Boolean - If raw data than after_ta = False

* **TA_metod –** the options listed in the hugobot system

* **Combination** – Boolean  - Combining two STI representations  – gradient + state abstraction

* **Transformation_number –**
* Discrete Transformation Transformation
* Symbol One-Hot Transformation
* Endpoint One-Hot Transformation

* **per_entity** – Boolean - Learning the cutoff per-entity, which means each entity has different cutoffs

