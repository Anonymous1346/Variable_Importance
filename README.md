# Variable_Importance
* The required packages are installed as a new conda environment including both R and Python dependencies with the following command:

```
conda env create -f requirements_conda.yml
```

* The missing R packages can be found in the "requirements_r.rda" file and can be downloaded using the following commands:

```
load("requirements_r.rda")

for (count in 1:length(installedpackages)) {
    install.packages(installedpackages[count])
}
```

* The ```permimp``` package can be downloaded with the following commands:

```
library(devtools)

install_github("Anonymous1346/permimp")
```

* For the 3 first experiments, ```compute_simulations``` is used along with ```plot_simulations_all```:
  * For the **first experiment**:
    * Uncomment both ```dnn_py``` and ```dnn_py_cond```
    * ```n_samples``` is set to 300 and ```n_featues``` is set to 100
    * Uncomment all the ```rho``` values
    * Set ```prob_sim_data``` to ```regression_perm```
    * The csv file can be downloaded at ```https://drive.google.com/file/d/1nzWc0b3FPNnghd-HfjPrMSFf5HZjkEaJ/view?usp=sharing```
  
  * For the **second experiment**:
    * Keep both ```dnn_py``` and ```dnn_py_cond``` uncommented
    * Set ```n_samples``` to ```n_samples = `if`(!DEBUG, seq(100, 1000, by = 100), 10L)``` (Uncomment the line directly below)
    * Set ```n_features``` to 50
    * In ```prob_sim_data```, comment ```regression_perm``` and uncomment all the rest
    * The csv file can be downloaded at ```https://drive.google.com/file/d/1e5djTGRn9SLIjxdMgPEKnjbBFH1D1Vy_/view?usp=sharing```
    
  * For the **third experiment**:
    * Uncomment all the methods
    * Set ```n_samples``` to 1000 and ```n_features``` to 50
    * In ```prob_sim_data```, comment ```regression_perm``` and uncomment all the rest
    * The csv file can be downloaded at ```https://drive.google.com/file/d/1iaKRp9i9H4MCz_n6Bxb713JIP1SuFVgH/view?usp=sharing```

  * Once the simulated data is computed, we move to the ```plot_simulations_all``` (Don't forget to change the name of the file to save with each expirement as it will be used later for the plots):
    * For the **first experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all_Mi.R")```
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power``` and ```run_time``` to TRUE
      * Finally, run the ```fix_fig_full_plot_Mi``` in the ```results``` folder
    
    * For the **second experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all_increasing_combine.R")```
      * Set ```run_plot_combine``` to TRUE
      * Don't forget to change the name of the input and output files
    
    * For the **third experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all.R")```
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power``` and ```run_time``` to TRUE
      * Finally, run the ```fix_fig_full_plot``` in the ```results``` folder
      
  * For the **forth experiment**, we move to the ```ukbb``` folder:
    * The data are the public data from UKBB that needs to sign an agreement before using it (Any personal data are already removed)
    * The csv file can be downloaded at ```https://drive.google.com/file/d/1ECfny4IOzoDn3yVBAvTO4wgDNtWmkVJ8/view?usp=sharing```
    * The ```ukbb_data_intelligence_no_hot_encoding``` csv file contains the needed data
    * In the ```process_intelligence``` script, set ```permfit_dnn`` and ```cpi_dnn``` to True to process the data and explore the importance of the variables
    * Finally, the plots are obtained using the ```Analyse_results``` script

* For the **predictive performance** of the core learners experiment:
  * In the ```plot_all_simulations```:
    * Uncomment ```Marg```, ```MDI```, ```BART```, ```Knockoff_lasso``` and ```Permfit-DNN```.
    * Set ```run_plot_pred``` only to TRUE
