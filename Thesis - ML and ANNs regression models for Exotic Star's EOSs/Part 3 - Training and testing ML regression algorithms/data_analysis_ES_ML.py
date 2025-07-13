# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py9
# Name: data_analysis_ES_ML.py

# Description: 
# Module offering classes and functions for assessing and analyzing the regression data
# of Exotic Stars using Machine Learning algorithms

# Abbrevations:
# ES -> Exotic Star
# NS -> Neutron Star
# QS -> Quark Star
# ML -> Machine Learning


# Importing necessary basic modules
import numpy as np 
import sympy as smp 
import matplotlib.pyplot as plt 
import random
from prettytable import PrettyTable
import os
import pandas as pd
import time 
import multiprocessing
from IPython import display as disp

# Importing modules for the preprocessing of the regresion data
# ML package for splitting the dataframe into train and test set and performing grid seacrh and cross validation
from sklearn.model_selection import train_test_split

# ML package for data scaling
from sklearn.preprocessing import StandardScaler

# Importing modules to perform grid search and cross validation during the training process
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# ML package for showing the learning progress of the model with a curve
from sklearn.model_selection import learning_curve

# Importing modules for metrics to evaluate the accuracy of the trained ML regression models
from sklearn.metrics import make_scorer,mean_squared_log_error,mean_squared_error

# Module to save and/or load regression models
import joblib


# Function that reads files containing data for ML regression and assesses the X data (i.e the Mass or Radius or both) 
# in terms of linear correlation between the (respective) columns
def check_linear_corr(filename,mag_check="both"):
    """
    Reading files containing data for machine learning (ML) regression and assessing the X data (i.e the Mass or Radius or both) 
    in terms of linear correlation between their (respective) columns

    1. filename: name of the file to be scanned
    2. mag_check: allowed values: ["both","mass","radius"]. The magnitude, the columns of which are to be checked for linear correlations
    """
    
    # Allowed values for the "mag_check" argument
    mag_check_allowedvalues = ["both","mass","radius"]
    if mag_check not in mag_check_allowedvalues:
        raise ValueError(f"Invalid value \"{mag_check}\" for the \"mag_check\" argument. Allowed values are: {mag_check_allowedvalues}")
    

    # Scanning the given file
    df = pd.read_csv(filename)

    # Getting the requested X data to be assessed, as well as the respective names of their columns
    if mag_check=="both":
        X_columns = [col for col in df.columns if col.startswith("M") or col.startswith("R")]
        X_df = df[X_columns]
    elif mag_check=="mass":
        X_columns = [col for col in df.columns if col.startswith("M")]
        X_df = df[X_columns]
    elif mag_check=="radius":
        X_columns = [col for col in df.columns if col.startswith("R")]
        X_df = df[X_columns]      

    X_df_rows,X_df_columns = np.shape(X_df) # shape of the X data

    # Initializing the Pretty Table to be filled with the linear correlation results
    show_linear_corr = PrettyTable()

    #  Forming the first (headers) column of the Pretty Table
    pretty_table_col = []
    show_linear_corr.add_column("i\j",X_columns)

    # Forming the rest columns of the Pretty Table
    for j in range(0,X_df_columns):
        pretty_table_col = []
        for i in range(0,X_df_columns):
            r_corr_coeff = X_df.iloc[:,j].corr(X_df.iloc[:,i]) # calculating the Pearson's correlation coefficient between two columns of the X dataframe
            pretty_table_col.append(f"{r_corr_coeff:.5f}")
        show_linear_corr.add_column(X_columns[j],pretty_table_col)

    # Printing the Pretty Table of the linear correlation results
    print(show_linear_corr)




# Class for training and assessing ML regression models using Mass and Radius values as features (explanatory data)
class regression_ML:
    """
    Training, assessing and saving machine learning regression models using Mass and Radius values as features (explanatory data)

    The reading of the .csv files is based on the automatic way the data are being saved in .csv files during the operation of the 'gen_reg_data'
    methods of the 'polyNSdata' and 'cflQSdata' classes in the 'ExoticStarsDataHandling.py' and 'ExoticStarsDataHandling2.py' modules
    """

    # Constructor of the class
    def __init__(self,filename=None,mag_reg="dPdE",test_ratio=0.25,samples_per_EOS=1):
        """
        Initializing the `regression_ML` class
        1. filename: name of the file containing data for regression purposes
        2. mag_reg: name of the category of target (response) variables for the regression models. Allowed inputs: ["dPdE","enrg","PtMmax","Gamma"]
        3. test_ratio: percentage (in decimal format) of the entire dataset to be used as a test dataset to evaluate the accuracy of the trained regression model
        4. samples_per_EOS: the number of rows that correspond to a single EOS. Each row represents a sample of this EOS. By default: 1 sample per EOS.
        """
        
        # Allowe values for the 'filesave' argument
        if filename==None:
            raise ValueError("An input for the \"filename\" argument must be given. Try again.")
        
        # Allowed values for the 'mag_reg' argument
        mag_reg_allowedvalues = ["dPdE","enrg","PtMmax","Gamma"]
        if mag_reg not in mag_reg_allowedvalues:
            raise ValueError(f"Invalid input \"{mag_reg}\" for the \"mag_reg\" argument. Valid inputs are: {mag_reg_allowedvalues}")
        
        # Allowed values for the 'test_ratio'
        if test_ratio<0 or test_ratio>1:
            raise ValueError(f"The value of the \"test_ratio\" argument must be a number in the [0,1] interval. Try again.")
        

        # Appending the inputs of the contrsuctor to self variables of the class
        self.filename = filename
        self.mag_reg = mag_reg
        self.test_ratio = test_ratio
        
        # Getting and appending the X (explanatory) and Y (response) data for regression to self variables of the class
        self.X_data,self.Y_data = self.split_df()

        # Getting the shape of X data
        X_rows,_ = np.shape(self.X_data)
        
        # Getting the split index bewteen train and test dataframes
        split_index = round((1-test_ratio)*(X_rows/samples_per_EOS))*samples_per_EOS

        # Splitting the X and Y data into train and test parts and appending them to self variables of tne class
        self.X_train = self.X_data.iloc[:split_index,:]
        self.X_test = self.X_data.iloc[split_index:,:]
        self.Y_train = self.Y_data.iloc[:split_index,:]
        self.Y_test = self.Y_data.iloc[split_index:,:]


    # Method that returns the X (explanatory) data and Y (response) data for regression
    def split_df(self):
        """
        Splitting the dataframe and returning the X (explanatory) data and Y (response) data for regression
        """
        
        # Getting the entire dataset from the scanned file
        df = pd.read_csv(self.filename)

        # Initializing the X and Y dataframes as empty lists
        X_data = []
        Y_data = []
        
        # Getting the X (explanatory) data
        X_columns = [col for col in df.columns if col.startswith("M") or col.startswith("R")]
        X_data = df[X_columns]

        # Getting the Y (response) data
        # Slope dPdE
        if self.mag_reg=="dPdE":
            Y_columns = [col for col in df.columns if col.startswith("dP_dE")]
            Y_data = df[Y_columns]
        # Energy on center    
        elif self.mag_reg=="enrg":
            Y_columns = [col for col in df.columns if col.startswith("E_c")]
            Y_data = df[Y_columns]
        # Center pressure at maximum mass    
        elif self.mag_reg=="PtMmax":
            Y_columns = [col for col in df.columns if col.startswith("Pc(M_max)") or col.startswith("Ec(M_max)")]
            Y_data = df[Y_columns]
        # Polytropic parameter Γ    
        elif self.mag_reg=="Gamma":
            Y_columns = [col for col in df.columns if col.startswith("Gamma")]
            Y_data = df[Y_columns]

        
        # Returning the X and Y data
        return X_data,Y_data
    

    # Method that shows an overview of the X and Y datasets, as well as their train and test parts
    def show_datasets(self):
        """
        Showing an overview of the X and Y datasets, as well as their train and test parts
        """

        print(">REGRESSION DATA OVERVIEW\n\n")

        print(">> X DATA (EXPLANATORY)\n")

        print(">>> Entire dataset:")
        disp.display(self.X_data)

        print(">>> Train dataset:")
        disp.display(self.X_train)

        print(">>> Test dataset:")
        disp.display(self.X_test)

        print("\n>> Y DATA (RESPONSE)\n")

        print(">>> Entire dataset:")
        disp.display(self.Y_data)

        print(">>> Train dataset:")
        disp.display(self.Y_train)

        print(">>> Test dataset:")
        disp.display(self.Y_test)

    
    # Method that performs grid search, trains with cross validation and evaluates a selected ML regression algorithm 
    # and saves the results in .pkl files
    def train_test(self,ml_regressor,kfold_splits,hpar_grid,cv_scorer_type,cores_par=2,filesave=None):
        """
        Training and assessing a regression model using a selected machine learning algorithm.
        1. ml_regressor: the machine learning regression algortihm to be selected for the model. Allowed inputs and their matches: ["rf"-> RandomForest,"dtree"-> Decision Tree,"svr"-> SVM sklearn, "gradboost"-> Gradient Boosting, "xgboost"-> Extreme Gradient Boosting, "adaboost"-> AdaBoost]
        2. kfold_splits: number of splits for the KFold cross-validation procedure
        3. hpar_grid: the grid of the values of the hyperparameters to be tuned during the Grid Search process. Must be given as a dictionary.
        4. cv_scorer_type: tool to evaluate the accuracy of the model during the cross-validation process. Allowed inputs and their matches: ["msle"-> mean squared log error, "mse"-> mean squared error]
        5. cores_par: number of cores to be used during the grid search and cross-validation procedure, allowing the algorithm to parallelize the workload and reduce the fitting time.
        6. filesave: name of the .pkl file, where the info of the best estimator and its metrics will be saved
        """ 

        # Allowed values for the 'ml_regressor' argument
        ml_regressor_allowedvalues = ["rf","dtree","svr","gradboost","xgboost","adaboost"]
        if ml_regressor not in ml_regressor_allowedvalues:
            raise ValueError(f"Invalid input \"{ml_regressor}\" for the \"ml_regressor\" argument. Valid inputs are: {ml_regressor_allowedvalues}")


        # Allowed values for the 'kfold_splits' argument
        if type(kfold_splits)!=type(2) or kfold_splits<=0:
            raise ValueError(f"The value of the \"kfold_splits\" argument must be a positive integer number. Try again.")

        # Allowed values for the 'hpar_grid' argument
        if type(hpar_grid)!=dict:
            raise ValueError(f"The input of the \"hpar_grid\" argument must be a dictionary. Try again.")

        # Allowed values for the 'scorer' argument
        cv_scorer_type_allowedvalues = ["msle","mse"]
        if cv_scorer_type not in cv_scorer_type_allowedvalues:
            raise ValueError(f"Invalid input \"{cv_scorer_type}\" for the \"scorer_type\" argument. Valid inputs are: {cv_scorer_type}")
        
        # Allowed values for the 'cores_par' argument
        cores_par = abs(cores_par) # getting the absolute value to ensure positive input value
        num_cores = multiprocessing.cpu_count() # number of available CPU cores
        if cores_par>num_cores-2: # leaving at least 2 cores of the CPU free for other tasks (this number can be changed arbitrarily)
            raise ValueError("Too many cores are selected, please give a lower number of CPU cores.") 
        
        # Allowed values for the 'filesave' argument
        if filesave==None:
            raise ValueError("An input for the \"filesave\" argument must be given. Try again.")


        # Preliminaries of the training process
        print("TRAINING AND ASSESSING A MACHINE LEARNING REGRESSION MODEL\n\n")

        print(">Preliminaries")
        print("===================================================================================================================")
        
        # DATA SCALING
        # General data info
        num_m_columns = len([col for col in self.X_train if col.startswith("M")]) # number of mass columns in the X data
        num_r_columns = len([col for col in self.X_train if col.startswith("R")]) # number of radius columns in the X data
        _,num_y_columns = np.shape(self.Y_data)

        num_train_rows,_ = np.shape(self.Y_train) # number of rows of the train datasets
        num_test_rows,_ = np.shape(self.Y_test) # number of rows of the test datasets


        # Initializing the scaler
        scaler = StandardScaler()

        # Scaling the X (explanatory) data
        X_train_scaled = scaler.fit_transform(np.array(self.X_train))
        X_test_scaled = scaler.transform(np.array(self.X_test))
        
        # Appending general data info and the fitted scaler info to a dictionary
        data_info = {"y_data_type":self.mag_reg, "y_columns":num_y_columns, 
                     "mass_columns":num_m_columns, "radius_columns":num_r_columns, 
                     "rows_train": num_train_rows, "rows_test": num_test_rows,
                     "test_size":self.test_ratio, "data_scaler":scaler}
        
        print(">> DATA INFO AND SCALING:")
        print("-------------------------------------------------------------------------------------------------------------------")
        print(f"Y (response) data type: \"{self.mag_reg}\"")
        print(f"Number of Y columns:  {num_y_columns}")
        print("X (explanatory) data type: \"Mass\" and \"Radius\"")
        print(f"Number of X columns:  {num_m_columns+num_r_columns}")
        print("The scaling of the X (explanatory) data has been completed")
        print("-------------------------------------------------------------------------------------------------------------------")

        
        # CROSS-VALIDATION SETTING
        print(">> CROSS-VALIDATION SETTINGS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Initializing the KFold cross-validation procedure
        kf = KFold(n_splits=kfold_splits, shuffle=True,random_state=50) # arbitrary selection of the random state
        print(f"The KFold cross-validator has been initialized with {kfold_splits} n_splits")

        # Initializing the scorer for the cross validation procedure
        if cv_scorer_type=="msle":
            cv_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
            cv_scorer_metric = "Mean_Squared_Log_Error"
        elif cv_scorer_type=="mse":
            cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            cv_scorer_metric = "Mean_Squared_Error"
        print(f"The cross-validation scorer has been initialized with the \"{cv_scorer_metric}\" as metric")
        print("-------------------------------------------------------------------------------------------------------------------")
        
        # REGRESSION MODEL SETTINGS
        print(">> ESTIMATOR INFO:")
        print("-------------------------------------------------------------------------------------------------------------------")
        
        # Loading and initializing regression algorithm
        
        # Decision Tree    
        if ml_regressor=="dtree":
            from sklearn.tree import DecisionTreeRegressor
            reg_model = DecisionTreeRegressor()
            reg_type = "DecisionTree"
        # Random Forest
        elif ml_regressor=="rf":
            from sklearn.ensemble import RandomForestRegressor
            reg_model = RandomForestRegressor(random_state=45) # arbitrary selection of the random state
            reg_type = "RandomForest"
        # SVR (SVM sklearn)    
        elif ml_regressor=="svr":
            from sklearn.svm import SVR
            from sklearn.multioutput import MultiOutputRegressor
            base_estimator = SVR()
            reg_model = MultiOutputRegressor(base_estimator)
            reg_type = "SVR (SVM sklearn)"
        # Gradient Boosting    
        elif ml_regressor=="gradboost":
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.multioutput import MultiOutputRegressor
            base_estimator = GradientBoostingRegressor(random_state=45) # arbitrary selection of the random state    
            reg_model = MultiOutputRegressor(base_estimator) 
            reg_type = "GradientBoosting"
        # Extreme Gradient Boosting
        elif ml_regressor=="xgboost":
            from sklearn.multioutput import MultiOutputRegressor
            from xgboost import XGBRegressor
            base_estimator = XGBRegressor(objective='reg:squarederror', verbosity=0)
            reg_model = MultiOutputRegressor(base_estimator)
            reg_type = "ExtremeGradientBoosting (XGBoost)"    
        # AdaBoost    
        elif ml_regressor=="adaboost":
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import AdaBoostRegressor
            from sklearn.multioutput import MultiOutputRegressor
            base_estimator = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3,random_state=45),random_state=45) # arbitrary selection of the random states
            reg_model = MultiOutputRegressor(base_estimator)
            reg_type = "AdaBoost"
        print(f"The \"{reg_type}\" regressor has been loaded and initialized")
        print("===================================================================================================================\n\n\n")


        # GRID SEARCH AND CROSS-VALIDATION PROCEDURES
        print(">Grid-Search and Cross-Validation")
        print("===================================================================================================================")
        print(">> HYPERPARAMETERS VALUES GRID:")
        print("-------------------------------------------------------------------------------------------------------------------")
        disp.display(hpar_grid)
        print("-------------------------------------------------------------------------------------------------------------------")

        print(">> FITTING PROCEDURE OVERVIEW:")
        print("-------------------------------------------------------------------------------------------------------------------")

        # Initializing the grid search
        cv_grid = GridSearchCV(estimator=reg_model,param_grid=hpar_grid,scoring=cv_scorer,cv=kf,n_jobs=cores_par,verbose=1)
        print("The grid search has been initialized")
        disp.display(cv_grid)
        print("Ongoing fitting process...")

        # Fitting the selected regression model 
        start_time = time.time() # starting the fitting time measurement

        cv_grid.fit(X_train_scaled,self.Y_train)
        disp.display(cv_grid)

        end_time = time.time() # terminating the fitting time measurement
        cpu_time_total = (end_time - start_time) # total execution time in seconds
        cpu_time_res = cpu_time_total%60 # remaining seconds if we express execution time in minutes
        cpu_time_total_mins = (cpu_time_total - cpu_time_res)/60 # minutes of the total execution time
        print("The fitting process has been completed")
        print("Elapsed fitting time: %.1f\'%.2f\""%(cpu_time_total_mins,cpu_time_res))
        print(f"Available CPU cores: {cores_par}")

        print("-------------------------------------------------------------------------------------------------------------------")
        print(">> RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Getting the best model and its parameters and score
        best_model = cv_grid.best_estimator_
        best_params = cv_grid.best_params_
        best_score = -cv_grid.best_score_

        # Printing the parameters and score of the best model
        print("Best model:  ",best_model)
        print("Best parameters:  ",best_params)
        print(f"Best cross-validation score ({cv_scorer_type}):  ",best_score)

        # Appending the best estimator's info to a dictionary
        best_estimator_info = {"model":best_model,"params":best_params,
        "cv_score":best_score,"fit_time":[cpu_time_total_mins,cpu_time_res],"cpu_cores":cores_par}
        print("===================================================================================================================\n\n\n")



        # OVERFITTING METRICS
        print(">Overfitting metrics (using the train dataset as test dataset)")
        print("===================================================================================================================")
        print(">> PREDICTIONS AND REAL VALUES:")
        print("-------------------------------------------------------------------------------------------------------------------")

        # Predictions of the best estimator on the X train data
        Y_predict_ovf = best_model.predict(X_train_scaled)
        
        # Printing the predicted values
        print(f"Predictions of \"{self.mag_reg}\"")
        disp.display(Y_predict_ovf)
        
        # Printing the real values
        print(f"Actual values of \"{self.mag_reg}\"")
        disp.display(self.Y_train)
        print("-------------------------------------------------------------------------------------------------------------------")
        
        print(">> MEAN SQUARED LOG ERROR (MSLE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared log error (MSLE)
        msle_ovftest_raw = mean_squared_log_error(self.Y_train,Y_predict_ovf,multioutput="raw_values")
        msle_ovftest_avg = mean_squared_log_error(self.Y_train,Y_predict_ovf,multioutput="uniform_average")

        print("Raw values")
        print(msle_ovftest_raw)
        print("Uniform average")
        print(msle_ovftest_avg)
        print("-------------------------------------------------------------------------------------------------------------------")

        print(">> MEAN SQUARED ERROR (MSE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared error (MSE)
        mse_ovftest_raw = mean_squared_error(self.Y_train,Y_predict_ovf,multioutput="raw_values")
        mse_ovftest_avg = mean_squared_error(self.Y_train,Y_predict_ovf,multioutput="uniform_average")

        print("Raw values")
        print(mse_ovftest_raw)
        print("Uniform average")
        print(mse_ovftest_avg)

        # Appending all the overfitting metrics to a dictionary
        ovf_metrics = {"msle_raw":msle_ovftest_raw, "msle_avg":msle_ovftest_avg, "mse_raw": mse_ovftest_raw, "mse_avg": mse_ovftest_avg}
        print("===================================================================================================================\n\n\n")


        # PREDICTION METRICS
        print(">Prediction metrics (using the actual test dataset)")
        print("===================================================================================================================")
        print(">> PREDICTIONS AND REAL VALUES:")
        print("-------------------------------------------------------------------------------------------------------------------")

        # Predictions of the best estimator on the X train data
        Y_predict = best_model.predict(X_test_scaled)
        
        # Printing the predicted values
        print(f"Predictions of \"{self.mag_reg}\"")
        disp.display(Y_predict)
        
        # Printing the real values
        print(f"Actual values of \"{self.mag_reg}\"")
        disp.display(self.Y_test)
        print("-------------------------------------------------------------------------------------------------------------------")
        
        print(">> MEAN SQUARED LOG ERROR (MSLE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared log error (MSLE)
        msle_test_raw = mean_squared_log_error(self.Y_test,Y_predict,multioutput="raw_values")
        msle_test_avg = mean_squared_log_error(self.Y_test,Y_predict,multioutput="uniform_average")

        print("Raw values")
        print(msle_test_raw)
        print("Uniform average")
        print(msle_test_avg)
        print("-------------------------------------------------------------------------------------------------------------------")

        print(">> MEAN SQUARED ERROR (MSE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared error (MSE)
        mse_test_raw = mean_squared_error(self.Y_test,Y_predict,multioutput="raw_values")
        mse_test_avg = mean_squared_error(self.Y_test,Y_predict,multioutput="uniform_average")

        print("Raw values")
        print(mse_test_raw)
        print("Uniform average")
        print(mse_test_avg)

        # Appending all the overfitting metrics to a dictionary
        test_metrics = {"msle_raw":msle_test_raw, "msle_avg":msle_test_avg, "mse_raw": mse_test_raw, "mse_avg": mse_test_avg}
        print("===================================================================================================================\n\n\n")

        # LEARNING CURVE
        # print(">Learning curve")
        # print("===================================================================================================================")

        # # Getting the train scores and validation scores of the best estimator to make its learning curve
        # train_sizes, lc_train_scores, lc_val_scores = learning_curve(
        #     best_model, X_train_scaled, self.Y_train, scoring='neg_mean_squared_log_error',cv=kfold_splits) # using the same KFold splits as in the GridSearchCV prossess

        # lc_train_scores_mean = -lc_train_scores.mean(axis=1)
        # lc_val_scores_mean = -lc_val_scores.mean(axis=1)

        # # Appending the data of the learning curve to a dictionary
        # learning_curve_data = {"train_sizes": train_sizes, "mean_train_scores": lc_train_scores_mean, "mean_val_scores": lc_val_scores_mean}
        
        # # Defining the figure and axis where the learning curve will be included
        # fig_lc,axis_lc = plt.subplots(1,1,figsize=(8,5))

        # axis_lc.plot(train_sizes, lc_train_scores_mean, label=f"Train: best MSLE->{lc_train_scores_mean[-1]:.2e}")
        # axis_lc.plot(train_sizes, lc_val_scores_mean, label=f"Validation: best MSLE->{lc_val_scores_mean[-1]:.2e}")
        # axis_lc.set_xlabel("Training size")
        # axis_lc.set_ylabel("MSLE")
        # axis_lc.set_yscale("log")
        # axis_lc.legend()
        # axis_lc.set_title(f"Learning Curve for best {reg_type} model")
        # plt.show()
        # print("===================================================================================================================\n\n\n")

        # SAVING THE BEST ESTIMATOR INFO AND ITS METRICS
        print(">Saving the grid search info:")
        print("===================================================================================================================")

        # Making an overview dictionary containing the grid search info
        grid_search_info = {"data_info": data_info, "best_estimator": best_estimator_info, 
                            "ovf_metrics": ovf_metrics, "test_metrics": test_metrics} # "learning_curve": learning_curve_data}
        
        # Saving the overview dictionary in a .pkl with the selected name
        joblib.dump(grid_search_info,f"{filesave}.pkl")

        print(f"The grid search info have been saved in the \"{filesave}.pkl\" file !!!")
        print("===================================================================================================================")



# Class to load saved ML regression models, get their metrics and/or make predictions from new data
class load_ML:
    """
    Loading saved machine learning regression models, geting their info and/or make predictions with new data

    All the processes of this class is based on the automatic way the informations are saved in .pkl files during the operation of
    the 'train_test' method of the 'regression_ML' class and the automatic way the data are being saved in .csv files during the operation 
    of the 'gen_reg_data' methods of the 'polyNSdata' and 'cflQSdata' classes in the 'ExoticStarsDataHandling.py' and 'ExoticStarsDataHandling2.py' modules
    """
    
    # Constructor of the class
    def __init__(self,model_file=None):
        """
        Scanning and opening the file containing the grid's info. Appending the info to self variables of the class
        1. model_file: name of the file to be scanned
        """

        # Allowed values for the 'model_file' argument
        if model_file==None:
            raise ValueError("An input for the \"model_file\" argument must be given. Try again.")
        
        # Opening the file and appending its info to a self variable
        self.model_file = model_file
        self.grid_load = joblib.load(model_file)


    # Method that returns the info of the data that was used to train and test the model    
    def get_used_data_info(self):
        """
        Returning the info of the data that was used to train and test the model
        """

        return self.grid_load["data_info"]
    
    # Method that returns the best estimator and its info
    def get_best_estimator(self):
        """
        Returning the best estimator and its info
        """

        return self.grid_load["best_estimator"]
    
    # Method that returns the metrics of the overfitting test
    def get_ovf_metrics(self):
        """
        Returning the he metrics of the overfitting test
        """

        return self.grid_load["ovf_metrics"]
    
    # Method that returns the metrics of the predictions test
    def get_test_metrics(self):
        """
        Returning the he metrics of the predictions test
        """

        return self.grid_load["test_metrics"]
    
    # Method that returns the learning curve info
    # def get_learning_curve_info(self):
    #     """
    #     Returning the learning curve info 
    #     """

    #     return self.grid_load["learning_curve"]
    
    # Method that makes predictions on foreign data
    def predict_new(self,new_data_file=None):
        """
        Making predictions on new dataset
        1. new_data_file: name of the .csv file containing foreign data to be used
        """
        
        # Allowed values for the 'new_data_file' argument
        if new_data_file==None:
            raise ValueError("An input for the \"new_data_file\" argument must be given. Try again.")
    

        # CHECKING FOR COMPATIBILITY OF THE NEW DATA WITH THE SELECTED MODEL
        print("PREDICTION ON NEW DATA\n\n")
        
        print(">Checking for compatibility")
        print("===================================================================================================================")
         # Getting the new data from the given file
        new_df = pd.read_csv(new_data_file)

        # Getting the number of Mass and Radius columns that were used to train the selected model
        num_m_columns_used = self.get_used_data_info()["mass_columns"]
        num_r_columns_used = self.get_used_data_info()["radius_columns"]

        # Getting the number of Mass and Radius columns of the new data
        m_columns_new = [col for col in new_df.columns if col.startswith("M")]
        num_m_columns_new = len(m_columns_new)
        r_columns_new = [col for col in new_df.columns if col.startswith("R")]
        num_r_columns_new = len(r_columns_new)

        # Raise value errors if the Mass and/or Radius columns in new data are different from the used data
        if num_m_columns_new!=num_m_columns_used:
            raise ValueError("The new data have different number of Mass columns from the train/test data of the selected model. Select another model or another new data")
        elif num_r_columns_new!=num_r_columns_used:
            raise ValueError("The new data have different number of Radius columns from the train/test data of the selected model. Select another model or another new data")

        # Getting the number and type of Y (response) data columns of the selected model
        num_y_columns_used = self.get_used_data_info()["y_columns"]
        y_type_used = self.get_used_data_info()["y_data_type"]

        # Getting the number of Y (response) data columns of the new data
        # Slope dPdE as response data
        if y_type_used=="dPdE":
           y_columns_new = [col for col in new_df.columns if col.startswith("dP_dE")]
           num_y_columns_new = len(y_columns_new)

        # Energy on center as response data
        elif y_type_used=="enrg":
           y_columns_new = [col for col in new_df.columns if col.startswith("E_c")]
           num_y_columns_new = len(y_columns_new)
        
        # Center pressure at maximum mass
        elif y_type_used=="PcMmax":
            y_columns_new = [col for col in new_df.columns if col.startswith("Pc(M_max)")]
            num_y_columns_new = len(y_columns_new)

        # Polytropic parameter Γ    
        elif self.mag_reg=="Gamma":
            y_columns_new = [col for col in new_df.columns if col.startswith("Gamma")]
            num_y_columns_new = len(y_columns_new)

        # Raise value errors if the Y data columns in new data are different from the used data
        if num_y_columns_new!=num_y_columns_used:
            raise ValueError("The new data have different number of Y (response) columns from the train/test data of the selected model. Select another model or another new data")         
        
        print(f"The new data on the \"{new_data_file}\" file and the selected model on the {self.model_file} are compatible !!!")
        print("Moving to predictions...")
        print("===================================================================================================================\n\n")
        

        # MAKING PREDICTIONS
        print(">Predictions")
        print("===================================================================================================================")

        # Getting the X (explanatory) data from the new data
        x_columns_new = [col for col in new_df.columns if col.startswith("M") or col.startswith("R")]
        X_new = new_df[x_columns_new]

        # Scaling the new data using the scaler of the used data of the selected model
        scaler_used = self.get_used_data_info()["data_scaler"]
        X_new_scaled = scaler_used.transform(X_new)
        
        # Getting and printing the actual values of the Y data of the new data
        Y_new = new_df[y_columns_new]
        print(f"Actual values of \"{y_type_used}\"")
        disp.display(Y_new)

        # Getting the selected model
        model_select = self.get_best_estimator()["model"]
        
        # Getting and printing the predictions
        Y_predict_new = model_select.predict(X_new_scaled)
        print(f"Predicted values of \"{y_type_used}\"")
        disp.display(Y_predict_new)
        print("===================================================================================================================\n\n")

        # CALCULATING METRICS
        print(">Metrics")
        print("===================================================================================================================")
        print(">> MEAN SQUARED LOG ERROR (MSLE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared log error (MSLE)
        msle_test_raw = mean_squared_log_error(Y_new,Y_predict_new,multioutput="raw_values")
        msle_test_avg = mean_squared_log_error(Y_new,Y_predict_new,multioutput="uniform_average")

        print("Raw values")
        print(msle_test_raw)
        print("Uniform average")
        print(msle_test_avg)
        print("-------------------------------------------------------------------------------------------------------------------")

        print(">> MEAN SQUARED ERROR (MSE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared error (MSE)
        mse_test_raw = mean_squared_error(Y_new,Y_predict_new,multioutput="raw_values")
        mse_test_avg = mean_squared_error(Y_new,Y_predict_new,multioutput="uniform_average")

        print("Raw values")
        print(mse_test_raw)
        print("Uniform average")
        print(mse_test_avg)

        # Appending all the overfitting metrics to a dictionary
        test_metrics = {"msle_raw":msle_test_raw, "msle_avg":msle_test_avg, "mse_raw": mse_test_raw, "mse_avg": mse_test_avg}
        print("===================================================================================================================\n\n\n")



# Class to load, list, summarize and compare metrics and performance of ML regression models
class assess_ML:
    """
    Loading, listing, summarizing and comparing metrics and performance of ML regression models

    All the processes of this class is based on the automatic way the informations are saved in .pkl files during the operation of
    the 'train_test' method of the 'regression_ML' class and the automatic way the data are being saved in .csv files during the operation 
    of the 'gen_reg_data' methods of the 'polyNSdata' and 'cflQSdata' classes in the 'ExoticStarsDataHandling.py' and 'ExoticStarsDataHandling2.py' modules
    """

    # Constructor of the class
    def __init__(self,star_type,mag_reg,x_data_types,EOS_type=None):
        """
        Initializing the 'assess_ML' class
        1. star_type: type of the star, the data of the EOSs of which were used in the regression processes. Allowed values: ["NS"->Neutron Star,"QS"->Quark Star]
        2. mag_reg: name of the category of target (response) variables for the regression models. Allowed inputs: ["dpde","enrg","gamma","PcMmax"]
        3. x_data_types: list of types of X (explanatory) data used in the regression in coded format. For example: '16X_rwsh' means 16 columns of rowwised shuffled X data
        4. EOS_type: type of the EOS. Default: None. Allowed inputs: [None,"poly","lin"]
        """
        
        # Allowed inputs for the 'star_type' argument
        star_type_allowedvalues = ["NS","QS"]
        if star_type not in star_type_allowedvalues:
            raise ValueError(f"Invalid input \"{star_type}\" for the \"star_type\" argument. Valid inputs are: {star_type_allowedvalues}")

        # Allowed inputs for the 'mag_reg' argument
        mag_reg_allowedvalues = ["dpde","enrg","gamma","PcMmax"]
        if mag_reg not in mag_reg_allowedvalues:
            raise ValueError(f"Invalid input \"{mag_reg}\" for the \"mag_reg\" argument. Valid inputs are: {mag_reg_allowedvalues}")

        # Allowed inputs for the 'EOS_type' argument
        EOS_type_allowedvalues = [None,"poly","lin"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid input \"{EOS_type}\" for the \"EOS_type\" argument. Valid inputs are: {EOS_type_allowedvalues}")
        if EOS_type==None:
            EOS_type=""    


        # Appending the 'star_type', 'mag_reg' and 'X_data_types' inputs to self variables of the 'assess_ML' class
        self.star_type = star_type
        self.mag_reg = mag_reg
        self.x_data_types = x_data_types
        self.EOS_type = EOS_type
     
        # Initialing the list of names of the ML algorithms as self variable
        self.ml_names = ["DecisionTree","RandomForest","GradientBoosting","XGBoost"]

        # Initialing the list of coded names of the ML regression algorithms as self variable
        self.ml_coded_names = ["dtree","rf","gradboost","xgboost"] # same order as in the self.ml_names list

        # Getting and appending the MSLE and MSE results to a self variable
        self.msle_array = self.listing_metric_results("msle")
        self.mse_array = self.listing_metric_results("mse")


    # Method to load all the available .pkl files containing grid search info over machine learning models, and listing the metrics
    def listing_metric_results(self,metric):
        """
        Loading and listing the metrics of all machine learning models, the grid search info of which are available in .pkl filess
        1. metric: the type of metric the values of which are going to be listed. Allowed values: ["msle","mse"]
        """
        
        # Allowed inputs for the 'metric' argument
        metric_allowedvalues = ["msle","mse"]
        if metric not in metric_allowedvalues:
            raise ValueError(f"Invalid input \"{metric}\" for the \"metric_type\" argument. Valid inputs are: {metric_allowedvalues}")
        

        # Initializing an array where the values of the metric will be listed and stored
        n = len(self.ml_names) # number of ML regression algorithms
        m = len(self.x_data_types) # number of X data types
        metric_array = np.ones((n,2*m)) # 2*m columns to include the overfitting results


        # Outer-Iterative process to scan all the available ML algorithms in .pkl files
        for i in range(0,n):
            # Inner-Iterative process to scan all the available X data types in .pkl files
            for j in range(0,m):
                filename = f"{self.EOS_type}{self.star_type}_{self.ml_coded_names[i]}_grid_{self.mag_reg}_{self.x_data_types[j]}.pkl"

                if os.path.exists(filename):
                    metrics_test = load_ML(filename).get_test_metrics() # getting the test metrics from the .pkl file
                    metrics_ovf = load_ML(filename).get_ovf_metrics() # getting the overfitting metrics from the .pkl file

                    if metric=="msle":
                        metric_array[i,2*j] = metrics_test["msle_avg"]
                        metric_array[i,2*j+1] = metrics_ovf["msle_avg"]
                    elif metric=="mse":
                        metric_array[i,2*j] = metrics_test["mse_avg"]
                        metric_array[i,2*j+1] = metrics_ovf["mse_avg"]
                else:
                    metric_array[i,2*j] = np.NaN
                    metric_array[i,2*j+1] = np.NaN


        return metric_array

    # Method that summarizes and saves the selected metric results in a .csv file
    def save_metric_results(self,metric):
        """
        Summarizing and saving the selected metric results in a .csv
        1. metric: the type of metric the values of which are going to be listed. Allowed values: ["msle","mse"]
        """

        # Allowed inputs for the 'metric' argument
        metric_allowedvalues = ["msle","mse"]
        if metric not in metric_allowedvalues:
            raise ValueError(f"Invalid input \"{metric_type}\" for the \"metric_type\" argument. Valid inputs are: {metric_allowedvalues}")

        # Filled rows and columns with data in the .csv file
        n = len(self.ml_names)
        m = len(self.x_data_types)
        
        # Name of the .csv file 
        filename = f"{self.EOS_type}{self.star_type}_{self.mag_reg}_{metric}_res.csv"

        # Getting the array with the results
        if metric=="msle":
            metric_array = self.msle_array
        elif metric=="mse":
            metric_array = self.mse_array    

        # Creating a new .csv file and its headers row
        headers_info = "models\X_type,"
        for j in range(0,m):
            if j==m-1:
                headers_info = headers_info + f"{self.x_data_types[j]},"
                headers_info = headers_info + f"{self.x_data_types[j]} (ovf.)\n"
            else:
                headers_info = headers_info + f"{self.x_data_types[j]},"
                headers_info = headers_info + f"{self.x_data_types[j]} (ovf.),"
       
        with open(filename,"w") as file:
            file.write(headers_info)

        # Filling the rest rows in the file
        for i in range(0,n):
            row_info = f"{self.ml_names[i]},"
            for j in range(0,m):
                if j==m-1:
                    row_info = row_info + f"{metric_array[i,2*j]:.4e},"
                    row_info = row_info + f"{metric_array[i,2*j+1]:.4e}\n"
                else:
                    row_info = row_info + f"{metric_array[i,2*j]:.4e},"
                    row_info = row_info + f"{metric_array[i,2*j+1]:.4e},"   
            
            with open(filename,"a+") as file:
                file.write(row_info)


        # Printing saving message
        print(f"The {metric.upper()} results for \"{self.mag_reg}\" prediction have been summarized and saved on the {filename} file !!!")
    
    # Method that summarizes and saves the learning curve info in a .pkl file
    # def save_lc_info(self,x_data_type_idx):
    #     """
    #     Summarizing and saving the learning curve info in a .pkl file
    #     1. x_data_type_idx: index of elements in the 'self.x_data_types' list
    #     """

    #     n = len(self.ml_names) # number of ML regression algorithms

    #     savefile = f"{self.EOS_type}{self.star_type}_lc_{self.mag_reg}_{self.x_data_types[x_data_type_idx]}.pkl" # .pkl file where the learning curve info will be saved
    #     learning_curves_info = {} # dictionary of learning curve info

    #     # Outer-Iterative process to scan all the available ML algorithms in .pkl files
    #     for i in range(0,n):
    #         filename = f"{EOS_type}{self.star_type}_{self.ml_coded_names[i]}_grid_{self.mag_reg}_{self.x_data_types[x_data_type_idx]}.pkl"

    #         learning_curve_model_info = load_ML(filename).get_learning_curve_info()
    #         learning_curves_info[f"{self.ml_coded_names[i]}"] = learning_curve_model_info

    #     # Saving the dictionary of learning curve info as .pkl file
    #     joblib.dump(learning_curves_info,savefile)
    #     print(f"The learning curves info for \"{self.x_data_types[x_data_type_idx]}\" X data and \"{self.mag_reg}\" Y data have been summarized and saved on the {savefile} file !!!")     



    # Method that prints the selected metric results in a PrettyTable format
    def print_metric_results(self,metric):
        """
        Printing the selected metric results in a PrettyTable format
        1. metric: the type of metric the values of which are going to be listed. Allowed values: ["msle","mse"]
        """

        # Allowed inputs for the 'metric' argument
        metric_allowedvalues = ["msle","mse"]
        if metric not in metric_allowedvalues:
            raise ValueError(f"Invalid input \"{metric}\" for the \"metric_type\" argument. Valid inputs are: {metric_allowedvalues}")

        # Getting the name of the file with the metric results
        if metric=="msle":
            filename = f"{self.EOS_type}{self.star_type}_{self.mag_reg}_{metric}_res.csv"
        elif metric=="mse":
            filename = f"{self.EOS_type}{self.star_type}_{self.mag_reg}_{metric}_res.csv"


        # Check if the file exists in current folder and load its contents
        if os.path.exists(filename):
            metric_data = pd.read_csv(filename) # loading the file with the metric's results data

            metric_data_columns = metric_data.columns # getting the headers of the columns of the dataframe
            describing_column = metric_data_columns[0] # getting the header of the first describing column
            x_data_types = metric_data_columns[1::2] # getting the headers of the X data types
            models_names = metric_data.iloc[:,0].to_numpy() # converting the dataframe part with the models names to a numpy array
            

            metric_array = metric_data.iloc[:,1:].to_numpy() # converting the dataframe part with the metric results to a numpy array
        else:
            raise ValueError(f"The file \"{filename}\" does not exist in current folder. You may use the \"save_metric_results()\" of the class and try again.")        

        
        # Initializing the PrettyTable
        metric_results = PrettyTable()

        # Forming the first (headers) column of the PrettyTable
        metric_results.add_column(describing_column,models_names)

        # Forming the rest columns of the PrettyTable
        n = len(models_names)
        m = len(x_data_types)
        for j in range(0,m):
            column_test_ovf = []
            for i in range(0,n):
                column_test_ovf.append(f"{metric_array[i,2*j]:.4e} ({metric_array[i,2*j+1]:.4e})")
            metric_results.add_column(f"{x_data_types[j]} (ovf.)",column_test_ovf)

        # Printing the PrettyTable
        print(metric_results)


    

    # Method the makes grouped bar plots of the selected metric values
    def make_bar_plots(self,metric,axis_gbar,bar_colors=None):
        """
        Making grouped bar plots of the selected metric values
        1. metric: the type of metric the values of which are going to be listed. Allowed values: ["msle","mse"]
        2. axis_gbar: the axis where the groups of bar plots will be included
        3. bar_colors: list of the colors of the bars. Bars that correspond to the same type of X data will have the same color. Thus the list must have size equal to the number of different X data types.
        """             
          
        # Allowed inputs for the 'metric' argument
        metric_allowedvalues = ["msle","mse"]
        if metric not in metric_allowedvalues:
            raise ValueError(f"Invalid input \"{metric}\" for the \"metric_type\" argument. Valid inputs are: {metric_allowedvalues}")

        
        # Getting the name of the file with the metric results
        if metric=="msle":
            filename = f"{self.EOS_type}{self.star_type}_{self.mag_reg}_{metric}_res.csv"
        elif metric=="mse":
            filename = f"{self.EOS_type}{self.star_type}_{self.mag_reg}_{metric}_res.csv"


       # Check if the file exists in current folder and load its contents
        if os.path.exists(filename):
            metric_data = pd.read_csv(filename) # loading the file with the metric's results data

            metric_data_columns = metric_data.columns # getting the headers of the columns of the dataframe
            describing_column = metric_data_columns[0] # getting the header of the first describing column
            x_data_types = metric_data_columns[1::2] # getting the headers of the X data types
            models_names = metric_data.iloc[:,0].to_numpy() # converting the dataframe part with the models names to a numpy array
            

            metric_array = metric_data.iloc[:,1:].to_numpy() # converting the dataframe part with the metric results to a numpy array
        else:
            raise ValueError(f"The file \"{filename}\" does not exist in current folder. You may use the \"save_metric_results()\" method of the class and try again.")
        
        
        n,_ = np.shape(metric_array) # number of different groups of bars (one group per ML algorithm)
        m = len(x_data_types) # number of bars per group

        # Setting the bar width
        bar_width = 1/(2*n)

        # Setting the fixed-starting position of each bar-group
        start_pos = np.arange(n)

        pos_offsets = np.linspace(-m/2,m/2,m)

        # Outer-iterative process to include all type of X data
        for j in range(0,m):
            bar_pos = start_pos + pos_offsets[j]*bar_width
            if bar_colors==None:
                axis_gbar.bar(bar_pos,metric_array[:,2*j],bar_width,label=self.x_data_types[j])
            else:    
                axis_gbar.bar(bar_pos,metric_array[:,2*j],bar_width,label=self.x_data_types[j],color=bar_colors[j])
            
            for i in range(0,n):
                ovf_metric = metric_array[i,2*j+1]
                axis_gbar.hlines(ovf_metric, bar_pos[i]-bar_width/2, bar_pos[i]+bar_width/2,colors="black",linewidth=3)


        # Setting the title in latex format
        if self.mag_reg=="dpde":
            y_type_latex = r"$\frac{dP}{dE}$"
        elif self.mag_reg=="enrg":
            y_type_latex = r"$E_c$"
        elif self.mag_reg=="gamma":
            y_type_latex = r"$Γ$"
        if self.mag_reg=="PcMmax":
            y_type_latex = r"$P_c(M_{max})$"                    

        # Adding tick_labels, legend and title for clarity
        axis_gbar.set_xticks(start_pos)
        axis_gbar.set_xticklabels(self.ml_names,fontsize=12)
        axis_gbar.set_ylabel(f"{metric.upper()}",fontsize=12)
        axis_gbar.set_title("Model " + metric.upper() + " Performance and Overfitting (Train/Test Comparison) on predicting " + y_type_latex + " values",fontsize=13)
        axis_gbar.legend(bbox_to_anchor=(1,1.01))

    # Method that plots the learning curves for a specific type of X (explanatory data)
    # def plot_learning_curves(self,x_data_type_idx,axis_lc,colors_lc):
    #     """
    #     Plotting the learning curves for a specific type of X (explanatory data)
    #     1. x_data_type_idx: index of elements in the 'self.x_data_types' list
    #     2. axis_lc: the axis where the learning curves will be included
    #     """

    #     # Making the name of the .pkl file where the learning curves info are saved
    #     filename = f"{self.EOS_type}{self.star_type}_lc_{self.mag_reg}_{self.x_data_types[x_data_type_idx]}.pkl"

    #     # Check if the file exists in current folder and load its contents
    #     if os.path.exists(filename):
    #         learning_curves_info = joblib.load(filename)
    #         models = list(learning_curves_info.keys()) # getting the names of the ML models used for regression
    #     else:
    #         raise ValueError(f"The file \"{filename}\" does not exist in current folder. You may use the \"save_lc_info()\" method of the class and try again.")    
            

    #     for i in range(0,len(models)):
    #         # Getting the train sizes, the mean train and validation scores for the current ML model
    #         train_sizes = learning_curves_info[models[i]]["train_sizes"]
    #         mean_train_scores = learning_curves_info[models[i]]["mean_train_scores"]
    #         mean_val_scores = learning_curves_info[models[i]]["mean_val_scores"]

    #         # Plotting the train learning curve of current model
    #         axis_lc.plot(train_sizes,mean_train_scores,color=colors_lc[i],label=f"{models[i]}_train")

    #         # Plotting the validation learning curve of current model
    #         axis_lc.plot(train_sizes,mean_val_scores,"--",color=colors_lc[i],label=f"{models[i]}_valid.")

            
    #     # Setting the title in latex format
    #     if self.mag_reg=="dpde":
    #         y_type_latex = r"$\frac{dP}{dE}$"
    #     elif self.mag_reg=="enrg":
    #         y_type_latex = r"$E_c$"
    #     elif self.mag_reg=="gamma":
    #         y_type_latex = r"$Γ$"
    #     if self.mag_reg=="PcMmax":
    #         y_type_latex = r"$P_c(M_{max})$" 

    #     # Adding tlabels, scale, legend and title for clarity
    #     axis_lc.set_xlabel("Train size",fontsize=12)
    #     axis_lc.set_ylabel("MSLE",fontsize=12)
    #     axis_lc.set_yscale("log")
    #     axis_lc.set_title("Predicting " + y_type_latex + f" values from {self.x_data_types[x_data_type_idx]} data",fontsize=13)
    #     axis_lc.legend(bbox_to_anchor=(1,1.01),fontsize=10)
