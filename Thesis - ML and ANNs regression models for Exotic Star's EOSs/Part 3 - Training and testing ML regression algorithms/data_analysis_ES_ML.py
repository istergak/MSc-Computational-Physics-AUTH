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
# Module offering classes and functions for assessing and analyzing the regression data of Exotic Stars

# Abbrevations:
# ES -> Exotic Star
# NS -> Neutron Star
# QS -> Quark Star
# ML -> Machine Learning


# Importing necessary modules
import numpy as np 
import sympy as smp 
import matplotlib.pyplot as plt 
import random
from prettytable import PrettyTable
import os
import pandas as pd


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