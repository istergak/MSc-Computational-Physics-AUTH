# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py10
# Name: data_analysis_ES_ANNs.py

# Description: 
# Module offering classes and functions for assessing and analyzing the regression data
# of Exotic Stars by building and using Deep Neural Networks (DNNs)

# Abbrevations:
# ES -> Exotic Star
# NS -> Neutron Star
# QS -> Quark Star
# DL -> Deep Learning
# DNN -> Deep Neural Network


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

# Package for data scaling
from sklearn.preprocessing import StandardScaler

# Importing modules to perform grid search and cross validation during the training process
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# Importing modules for metrics to evaluate the accuracy of the trained ML regression models
from sklearn.metrics import mean_squared_log_error,mean_squared_error

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


# Class for building, training and assessing an ANN for regression using Mass and Radius values as features (explanatory data)
class regression_ANN:
    """
    Building, training, assessing and saving an Artificial Neural Network for regression using Mass and Radius values as features (explanatory data)

    The reading of the .csv files is based on the automatic way the data are being saved in .csv files during the operation of the 'gen_reg_data'
    methods of the 'polyNSdata' and 'cflQSdata' classes in the 'ExoticStarsDataHandling.py' and 'ExoticStarsDataHandling2.py' modules
    """

    # Constructor of the class
    def __init__(self,filename=None,mag_reg="dPdE",test_ratio=0.25,val_ratio=0.2,samples_per_EOS=1):
        """
        Initializing the `regression_ANN` class
        1. filename: name of the file containing data for regression purposes
        2. mag_reg: name of the category of target (response) variables for the regression models. Allowed inputs: ["dPdE","enrg","PtMmax","Gamma"]
        3. test_ratio: decimal ratio of the entire dataset to be used as a test dataset to evaluate the accuracy of the trained ANN regression model
        4. val_ratio: decimal ratio of the train dataset to be used as a validation dataset to evaluate the accuracy during the training process ANN regression model
        5. samples_per_EOS: the number of rows that correspond to a single EOS. Each row represents a sample of this EOS. By default: 1 sample per EOS.
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

        # Getting the shape of X train data
        X_rows_train,_ = np.shape(self.X_train)

        # Getting the split index bewteen final train and validation dataframes
        split_index_val = round((1-val_ratio)*(X_rows_train/samples_per_EOS))*samples_per_EOS

        # Splitting the X and Y train data into final train and validation parts and appending them to self variables of tne class
        self.X_final_train = self.X_train.iloc[:split_index_val,:]
        self.X_val = self.X_train.iloc[split_index_val:,:]
        self.Y_final_train = self.Y_train.iloc[:split_index_val,:]
        self.Y_val = self.Y_train.iloc[split_index_val:,:]



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

        print(">>> Final Train dataset:")
        disp.display(self.X_final_train)

        print(">>> Validation dataset:")
        disp.display(self.X_val)

        print("\n>> Y DATA (RESPONSE)\n")

        print(">>> Entire dataset:")
        disp.display(self.Y_data)

        print(">>> Train dataset:")
        disp.display(self.Y_train)

        print(">>> Test dataset:")
        disp.display(self.Y_test)

        print(">>> Final Train dataset:")
        disp.display(self.Y_final_train)

        print(">>> Validation dataset:")
        disp.display(self.Y_val)


    # Method to build and compile an ANN for regression
    def compile_model(self,layers_neurons=None,layers_activations=None,layers_dropouts=None,adam_learn_rate=10e-4):
        """
        Building and compiling an Artificial Neural Network (ANN) for regression. The model will have at least one middle (secret) layer, apart from the input and output layers.
        1. layers_neurons: list with the numbers of neurons for each of the middle layers of the network. The length of the list is equal to the amount of middle layers of the network.
        If `None` is given as input (default), only one middle layer will be included in the network, having twice the number of neurons of the input layer, the `relu` activation function and no dropout of neurons.
        2. layers_activations: list with the activation functions of each middle layer of the network. If `None` is given as input (default), the 'relu' function will be used as activation function for each middle layer.
        3. layers_dropouts: list with the dropouts of neurons after the optimization of each middle layer. The dropouts of neurons helps to avoid overfitting and to generalize the model.
        if `None` is given as input (default), then no dropouts will be used in the training process of the network.
        4. adam_learn_rate: the learning rate of the `Adam` optimizer during the training process of the network. By default a learing rate of `10e-4` is given.
        """
        
        # Importing the layers formation and optimizing tools for the netwrok from the tensorflow module
        from tensorflow import keras
        from tensorflow.keras import backend as K
        import gc

        
        from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam

        # Getting the number of columns of X (explanatory) and Y (response) data
        x_rows,x_cols = np.shape(self.X_data)
        y_row,y_cols = np.shape(self.Y_data)

        # Reseting naming counters of the model and its layers and clearing residual memory from previous models 
        K.clear_session()  
        gc.collect()  

        # Forming the layers of the neural network

        # Initializing the list with the structure of the neural network
        model_structure = []

        # INPUT LAYER
        model_structure.append(Input(shape=(x_cols,)))

        # MIDDLE (HIDDEN) LAYERS
        if layers_neurons==None: # default case (one middle layer with twice the number of neurons of the input layer)
            # Dense layer
            model_structure.append(Dense(2*x_cols,activation='relu')) 
            # Regularization process to reduce overfitting
            model_structure.append(BatchNormalization()) 
        else:
            for i in range(0,len(layers_neurons)):
                # Dense layer
                if layers_activations==None: # default case (every layer will have the `relu` as activation function)
                    model_structure.append(Dense(layers_neurons[i],activation='relu'))
                else:
                    model_structure.append(Dense(layers_neurons[i],activation=layers_activations[i]))

                # Regularization process to reduce overfitting
                model_structure.append(BatchNormalization())

                # Dropout of neurons of the current layer to further reduce overfitting
                if layers_dropouts!=None:
                    model_structure.append(Dropout(rate=layers_dropouts[i]))

        # OUTPUT LAYER
        model_structure.append(Dense(y_cols))

        
        # Building the model
        model = keras.Sequential(model_structure)

        # Forming the Adam optimizer for the model
        model_optimizer = Adam(learning_rate=adam_learn_rate)

        # Compiling the model
        model.compile(optimizer=model_optimizer,loss='mean_squared_logarithmic_error')

        # Returning the compiled model
        return model

    
    # Method to train the neural network using the loaded regression data
    def train_model(self,y_scaling="no",layers_neurons=None,layers_activations=None,layers_dropouts=None,adam_learn_rate=1e-4,train_shuffle=True,train_epochs=100,batch_size=100,filesave=None):
        """
        Training the neural network using the loaded regression data
        1. y_scaling: wether the algorithm will use the scaled response data or not. By default: "no". Allowed inputs: ["no","yes"].
        2. layers_neurons: list with the numbers of neurons for each of the middle layers of the network. The length of the list is equal to the amount of middle layers of the network.
        If `None` is given as input (default), only one middle layer will be included in the network, having twice the number of neurons of the input layer, the `relu` activation function and no dropout of neurons.
        3. layers_activations: list with the activation functions of each middle layer of the network. If `None` is given as input (default), the 'relu' function will be used as activation function for each middle layer.
        4. layers_dropouts: list with the dropouts of neurons after the optimization of each middle layer. The dropouts of neurons helps to avoid overfitting and to generalize the model.
        if `None` is given as input (default), then no dropouts will be used in the training process of the network.
        5. adam_learn_rate: the learning rate of the `Adam` optimizer during the training process of the network. By default a learing rate of `1e-4` is given.
        6. train_shuffle: Boolean, whether the algorithm shuffles the train dataset before each epoch or not. The validation dataset is not shuffled.
        7. train_epochs [Default = 100.]: epochs of the training process.
        8. batch_size [Default = 100]: batch size of the training process.
        9. filesave: name of the .pkl file, where the info of the trained model, its log history and its metrics will be saved
        """
        
        # Allowed values for the 'y_scaling' argument
        y_scaling_allowedvalues = ["no","yes"]
        if y_scaling not in y_scaling_allowedvalues:
            raise ValueError(f"Invalid input \"{y_scaling}\" for the \"y_scaling\" argument. Valid inputs are: {y_scaling_allowedvalues}")
        
        # Allowed values for the 'filesave' argument
        if filesave==None:
            raise ValueError("An input for the \"filesave\" argument must be given. Try again.")
        

        # Preliminaries of the training process
        print("TRAINING AND ASSESSING AN ARTIFICIAL NEURAL NETWORK REGRESSION MODEL\n\n")

        print(">Preliminaries")
        print("===================================================================================================================")
        
        # DATA SCALING
        # General data info
        num_m_columns = len([col for col in self.X_train if col.startswith("M")]) # number of mass columns in the X data
        num_r_columns = len([col for col in self.X_train if col.startswith("R")]) # number of radius columns in the X data
        _,num_y_columns = np.shape(self.Y_data)

        num_train_rows,_ = np.shape(self.Y_train) # number of rows of the train datasets
        num_test_rows,_ = np.shape(self.Y_test) # number of rows of the test datasets


        # Initializing the scalers for X and Y data
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # Scaling the X (explanatory) data
        X_final_train_scaled = scaler_x.fit_transform(np.array(self.X_final_train))
        X_val_scaled = scaler_x.transform(np.array(self.X_val))
        X_test_scaled = scaler_x.transform(np.array(self.X_test))

        # Scaling the Y (response) data
        Y_final_train_scaled = scaler_y.fit_transform(np.array(self.Y_final_train))
        Y_val_scaled = scaler_y.transform(np.array(self.Y_val))
        Y_test_scaled = scaler_y.transform(np.array(self.Y_test))
        
        # Appending general data info and the fitted scaler info to a dictionary
        data_info = {"y_data_type":self.mag_reg, "y_columns":num_y_columns, 
                     "mass_columns":num_m_columns, "radius_columns":num_r_columns, 
                     "rows_train": num_train_rows, "rows_test": num_test_rows,
                     "test_size":self.test_ratio, "data_scaler": {"X": scaler_x, "Y": scaler_y}}
        
        print(">> DATA INFO AND SCALING:")
        print("-------------------------------------------------------------------------------------------------------------------")
        print(f"Y (response) data type: \"{self.mag_reg}\"")
        print(f"Number of Y columns:  {num_y_columns}")
        print("X (explanatory) data type: \"Mass\" and \"Radius\"")
        print(f"Number of X columns:  {num_m_columns+num_r_columns}")
        print("The scaling of the X (explanatory) data has been completed")
        print("The scaling of the Y (response) data has been completed")
        print("===================================================================================================================\n\n\n")

        # COMPILING AND FITTING PROCEDURES
        print(">Compiling and fitting the model")
        print("===================================================================================================================")

        # Compiling an ANN using the `self.compile_model()` method of the class
        model = self.compile_model(layers_neurons,layers_activations,layers_dropouts,adam_learn_rate)

        # Presenting a summary of the compiled model
        print(">> COMPILATION SUMMARY:")
        print("-------------------------------------------------------------------------------------------------------------------")
        disp.display(model.summary())
        print("-------------------------------------------------------------------------------------------------------------------")
        print(">> TRAINING:")
        print("-------------------------------------------------------------------------------------------------------------------")
        print("Ongoing fitting process...")

        start_time = time.time() # starting the fitting time measurement
        
        if y_scaling == "no":
            training_log = model.fit(X_final_train_scaled, self.Y_final_train, epochs = train_epochs, shuffle=train_shuffle, batch_size=batch_size,validation_data=(X_val_scaled,self.Y_val))
        elif y_scaling == "yes":
            training_log = model.fit(X_final_train_scaled, Y_final_train_scaled, epochs = train_epochs, shuffle=train_shuffle, batch_size=batch_size,validation_data=(X_val_scaled,Y_val_scaled))    

        end_time = time.time() # terminating the fitting time measurement
        cpu_time_total = (end_time - start_time) # total execution time in seconds
        cpu_time_res = cpu_time_total%60 # remaining seconds if we express execution time in minutes
        cpu_time_total_mins = (cpu_time_total - cpu_time_res)/60 # minutes of the total execution time
        print("The fitting process has been completed")
        print("Elapsed fitting time: %.1f\'%.2f\""%(cpu_time_total_mins,cpu_time_res))
        print("===================================================================================================================\n\n\n")


        # OVERFITTING METRICS
        print(">Overfitting metrics (using the train dataset as test dataset)")
        print("===================================================================================================================")
        print(">> PREDICTIONS AND REAL VALUES:")
        print("-------------------------------------------------------------------------------------------------------------------")

        # Predictions of the best estimator on the X train data
        if y_scaling == "no":
            Y_predict_ovf = abs(model.predict(X_final_train_scaled))
        elif y_scaling == "yes":
            Y_predict_ovf_scaled = model.predict(X_final_train_scaled)    
            Y_predict_ovf = abs(scaler_y.inverse_transform(Y_predict_ovf_scaled))
        
        # Printing the predicted values
        print(f"Predictions of \"{self.mag_reg}\"")
        disp.display(Y_predict_ovf)
        
        # Printing the real values
        print(f"Actual values of \"{self.mag_reg}\"")
        disp.display(self.Y_final_train)
        print("-------------------------------------------------------------------------------------------------------------------")
        
        print(">> MEAN SQUARED LOG ERROR (MSLE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared log error (MSLE)
        msle_ovftest_raw = mean_squared_log_error(self.Y_final_train,Y_predict_ovf,multioutput="raw_values")
        msle_ovftest_avg = mean_squared_log_error(self.Y_final_train,Y_predict_ovf,multioutput="uniform_average")

        print("Raw values")
        print(msle_ovftest_raw)
        print("Uniform average")
        print(msle_ovftest_avg)
        print("-------------------------------------------------------------------------------------------------------------------")

        print(">> MEAN SQUARED ERROR (MSE) RESULTS:")
        print("-------------------------------------------------------------------------------------------------------------------")
        # Measuring the mean squared error (MSE)
        mse_ovftest_raw = mean_squared_error(self.Y_final_train,Y_predict_ovf,multioutput="raw_values")
        mse_ovftest_avg = mean_squared_error(self.Y_final_train,Y_predict_ovf,multioutput="uniform_average")

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
        if y_scaling=="no":
            Y_predict = abs(model.predict(X_test_scaled))
        elif y_scaling=="yes":
            Y_predict_scaled = model.predict(X_test_scaled)  
            Y_predict = abs(scaler_y.inverse_transform(Y_predict_scaled))
        
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
        print(">Learning curve")
        print("===================================================================================================================")
        
        # Defining the figure and axis where the learning curve will be included
        fig_lc,axis_lc = plt.subplots(1,1,figsize=(8,5))
        
        # Making a dictionary for the learning curve info
        learning_curve_info = {'loss_log': training_log.history['loss'],
                               'val_loss_log': training_log.history['val_loss']}

        axis_lc.plot(training_log.history['loss'], label='Training Loss', linestyle='--', color='blue')
        axis_lc.plot(training_log.history['val_loss'], label='Validation Loss', linestyle='-', color='orange')
        axis_lc.set_xlabel("Epochs")
        axis_lc.set_ylabel("Loss (MSLE)")
        axis_lc.set_xscale("log")
        axis_lc.set_yscale("log")
        axis_lc.legend()
        axis_lc.set_title("Learning curve: Training and Validation Loss Over Epochs")
        plt.show()
        print("===================================================================================================================\n\n\n")

        # SAVING THE BEST ESTIMATOR INFO AND ITS METRICS
        print(">Saving the ANN model info:")
        print("===================================================================================================================")

        # Making an overview dictionary containing the grid search info
        model_info = {"data_info": data_info, "estimator": model,
                      "fit_time": [cpu_time_total_mins,cpu_time_res],
                      "ovf_metrics": ovf_metrics, "test_metrics": test_metrics,
                      "learning_curve": learning_curve_info}
        
        # Adding the suffix 'ysc' in the name if y_scaling was included in the training
        if y_scaling=="yes":
            filesave = filesave + "_ysc"

        # Saving the overview dictionary in a .pkl with the selected name
        joblib.dump(model_info,f"{filesave}.pkl")

        print(f"The ANNs model info have been saved in the \"{filesave}.pkl\" file !!!")
        print("===================================================================================================================")


# Class to load saved ANN regression models and get their metrics
class load_ANN:
    """
    Loading saved artificial neural network regression models and geting their info

    All the processes of this class is based on the automatic way the informations are saved in .pkl files during the operation of
    the 'train_model' method of the 'regression_ANN' class and the automatic way the data are being saved in .csv files during the operation 
    of the 'gen_reg_data' methods of the 'polyNSdata' and 'cflQSdata' classes in the 'ExoticStarsDataHandling.py' and 'ExoticStarsDataHandling2.py' modules
    """
    
    # Constructor of the class
    def __init__(self,model_file=None):
        """
        Scanning and opening the file containing the neural network's info. Appending the info to self variables of the class
        1. model_file: name of the file to be scanned
        """

        # Allowed values for the 'model_file' argument
        if model_file==None:
            raise ValueError("An input for the \"model_file\" argument must be given. Try again.")
        
        # Opening the file and appending its info to a self variable
        self.model_file = model_file
        self.model_load = joblib.load(model_file)


    # Method that returns the info of the data that was used to train and test the model    
    def get_used_data_info(self):
        """
        Returning the info of the data that was used to train and test the model
        """

        return self.model_load["data_info"]
    
    # Method that returns the estimator
    def get_estimator(self):
        """
        Returning the estimator
        """

        return self.model_load["estimator"]
    
    # Method that returns the fitting time of the neural network model
    def get_fit_time(self):
        """
        Returning the fitting time of the neural network model
        """

        return self.model_load["fit_time"]
    
    
    # Method that returns the learning curve info
    def get_learning_curve_info(self):
        """
        Returning the learning curve info of the model
        """

        return self.model_load["learning_curve"]
