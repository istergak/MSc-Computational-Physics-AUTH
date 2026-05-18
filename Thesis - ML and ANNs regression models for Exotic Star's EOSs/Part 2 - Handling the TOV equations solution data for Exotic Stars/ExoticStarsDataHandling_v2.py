# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py8a
# Name: ExoticStarsDataHandling_v2.py

# Description: 
# Module offering classes and functions for plotting MR curves, EOS curves and sampling data
# from the solution data of the TOV equations for Exotic Stars (Neutron Stars, Quark Stars)

# Version 2 additions:
# -Improved file reading and data separation

# Abbrevations:
# ES -> Exotic Star
# NS -> Neutron Star
# QS -> Quark Star


# Importing necessary modules
import numpy as np 
import sympy as smp 
import matplotlib.pyplot as plt 
import random
from prettytable import PrettyTable
import os
import pandas as pd
from scipy.interpolate import pchip

# Defining useful constants
ρ_sat = 2.7*10**14 # nuclear saturation density [g/cm^3]
n_sat = 0.16 # baryon density at saturation [fm^-3]
c = 3e8 # vaccuum speed of light in m/s
MeV_to_J = 1.60218e-13 # 1MeV value in Joules
m_s = 95 # MeV, mass of Strange quark
m_n = 939.565 # MeV, mass of Neutron
hbarc = 197.327 # MeV*fm, hbar*c constant

# Defining the relation between pressure and baryon density at the linear region (baryon density is expressed in fm^-3 units)
def n_b_transition(P,rho_frac_tr,P_tr,E_tr):
    return n_sat*rho_frac_tr*np.sqrt((2*P-P_tr+E_tr)/(E_tr+P_tr))

# Function that reads .csv files containing solution data of the TOV equations 
def file_read(filename,EOS_type="hadronic"):
    """
    Reading a .csv file containing the solution data of TOV equations for a main EOS or a polytropic Neutron Star EOS or a cfl matter Quark Star EOS.
    Obtaining and seperating the data in terms of violation of the causality (c_s/c<=1). Returning the data as a nested list.
    1. filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
    module script. To scan on another folder, the exact path of the file must be provided.
    2. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS or a Quark Star EOS.
    Allowed values: ["hadronic","hadronic_compose","quark"], Default->"hadronic".
    """

    # Allowed values for the 'EOS_type' argument
    EOS_type_allowedvalues = ["hadronic","hadronic_compose","quark"]
    if EOS_type not in EOS_type_allowedvalues:
        raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type_allowedvalues}")
        
    if EOS_type=="hadronic":
        # Reading the file 
        ES_data = pd.read_csv(filename)

        # Getting the columns of the data
        Pc_col = [col for col in ES_data.columns if col.count("P_c")]
        Ec_col = [col for col in ES_data.columns if col.count("E_c")]
        dE_dP_col = [col for col in ES_data.columns if col.count("dE_dP") or col.count("dE/dP")]
        M_col = [col for col in ES_data.columns if col.count("M")]
        R_col = [col for col in ES_data.columns if col.startswith("R") or col.startswith(" R")]
        yR_col = [col for col in ES_data.columns if col.count("yR")]
        k2_col = [col for col in ES_data.columns if col.count("k2")]
        Lambda_col = [col for col in ES_data.columns if col.count("L")]
        dens_col = [col for col in ES_data.columns if col.count("rho/rho_sat") or col.count("n_b")]
        if dens_col==[]:
            ES_data["rho/rho_sat"] = np.nan
            dens_col = "rho/rho_sat"


        # Obtaining the data of the respective columns
        Pc_data_raw = pd.to_numeric(ES_data[Pc_col].squeeze(), errors="coerce")
        Ec_data_raw = pd.to_numeric(ES_data[Ec_col].squeeze(), errors="coerce")
        dE_dP_data_raw = pd.to_numeric(ES_data[dE_dP_col].squeeze(), errors="coerce")
        M_data_raw = pd.to_numeric(ES_data[M_col].squeeze(), errors="coerce")
        R_data_raw = pd.to_numeric(ES_data[R_col].squeeze(), errors="coerce")
        yR_data_raw = pd.to_numeric(ES_data[yR_col].squeeze(), errors="coerce")
        k2_data_raw = pd.to_numeric(ES_data[k2_col].squeeze(), errors="coerce")
        Lambda_data_raw = pd.to_numeric(ES_data[Lambda_col].squeeze(), errors="coerce")
        dens_data_raw = pd.to_numeric(ES_data[dens_col].squeeze(), errors="coerce")
        
        # Obtaining the data that do not violate causality
        Pc_caus = Pc_data_raw[dE_dP_data_raw>=1]
        Ec_caus = Ec_data_raw[dE_dP_data_raw>=1]
        dE_dP_caus = dE_dP_data_raw[dE_dP_data_raw>=1]
        M_caus = M_data_raw[dE_dP_data_raw>=1]
        R_caus = R_data_raw[dE_dP_data_raw>=1]
        yR_caus = yR_data_raw[dE_dP_data_raw>=1]
        k2_caus = k2_data_raw[dE_dP_data_raw>=1]
        Lambda_caus = Lambda_data_raw[dE_dP_data_raw>=1]
        dens_caus = dens_data_raw[dE_dP_data_raw>=1]
                
        # Obtaining the data that do violate causality
        Pc_no_caus = Pc_data_raw[dE_dP_data_raw<1]
        Ec_no_caus = Ec_data_raw[dE_dP_data_raw<1]
        dE_dP_no_caus = dE_dP_data_raw[dE_dP_data_raw<1]
        M_no_caus = M_data_raw[dE_dP_data_raw<1]
        R_no_caus = R_data_raw[dE_dP_data_raw<1]
        yR_no_caus = yR_data_raw[dE_dP_data_raw<1]
        k2_no_caus = k2_data_raw[dE_dP_data_raw<1]
        Lambda_no_caus = Lambda_data_raw[dE_dP_data_raw<1]
        dens_no_caus = dens_data_raw[dE_dP_data_raw<1]

    elif EOS_type=="hadronic_compose":
        # Reading the file 
        ES_data = pd.read_csv(filename)

        # Getting the columns of the data
        Pc_col = [col for col in ES_data.columns if col.count("P_c")]
        Ec_col = [col for col in ES_data.columns if col.count("E")]
        c_s_sq_col = [col for col in ES_data.columns if col.count("c_s_sq")]
        M_col = [col for col in ES_data.columns if col.count("M")]
        R_col = [col for col in ES_data.columns if col.startswith("R") or col.startswith(" R")]
        yR_col = [col for col in ES_data.columns if col.count("yR")]
        k2_col = [col for col in ES_data.columns if col.count("k2")]
        Lambda_col = [col for col in ES_data.columns if col.count("L")]
        dens_col = [col for col in ES_data.columns if col.count("n_b")]

        # Obtaining the data of the respective columns
        Pc_data_raw = pd.to_numeric(ES_data[Pc_col].squeeze(), errors="coerce")
        Ec_data_raw = pd.to_numeric(ES_data[Ec_col].squeeze(), errors="coerce")
        c_s_sq_data_raw = pd.to_numeric(ES_data[c_s_sq_col].squeeze(), errors="coerce")
        dE_dP_data_raw = 1./c_s_sq_data_raw
        M_data_raw = pd.to_numeric(ES_data[M_col].squeeze(), errors="coerce")
        R_data_raw = pd.to_numeric(ES_data[R_col].squeeze(), errors="coerce")
        yR_data_raw = pd.to_numeric(ES_data[yR_col].squeeze(), errors="coerce")
        k2_data_raw = pd.to_numeric(ES_data[k2_col].squeeze(), errors="coerce")
        Lambda_data_raw = pd.to_numeric(ES_data[Lambda_col].squeeze(), errors="coerce")
        dens_data_raw = pd.to_numeric(ES_data[dens_col].squeeze(), errors="coerce")
        
        # Obtaining the data that do not violate causality
        Pc_caus = Pc_data_raw[dE_dP_data_raw>=1]
        Ec_caus = Ec_data_raw[dE_dP_data_raw>=1]
        dE_dP_caus = dE_dP_data_raw[dE_dP_data_raw>=1]
        M_caus = M_data_raw[dE_dP_data_raw>=1]
        R_caus = R_data_raw[dE_dP_data_raw>=1]
        yR_caus = yR_data_raw[dE_dP_data_raw>=1]
        k2_caus = k2_data_raw[dE_dP_data_raw>=1]
        Lambda_caus = Lambda_data_raw[dE_dP_data_raw>=1]
        dens_caus = dens_data_raw[dE_dP_data_raw>=1]
                
        # Obtaining the data that do violate causality
        Pc_no_caus = Pc_data_raw[dE_dP_data_raw<1]
        Ec_no_caus = Ec_data_raw[dE_dP_data_raw<1]
        dE_dP_no_caus = dE_dP_data_raw[dE_dP_data_raw<1]
        M_no_caus = M_data_raw[dE_dP_data_raw<1]
        R_no_caus = R_data_raw[dE_dP_data_raw<1]
        yR_no_caus = yR_data_raw[dE_dP_data_raw<1]
        k2_no_caus = k2_data_raw[dE_dP_data_raw<1]
        Lambda_no_caus = Lambda_data_raw[dE_dP_data_raw<1]
        dens_no_caus = dens_data_raw[dE_dP_data_raw<1]    
   

    elif EOS_type=="quark":
        # Reading the file 
        ES_data = pd.read_csv(filename)


        # Getting the columns of the data
        Pc_col = [col for col in ES_data.columns if col.count("P_c")]
        Ec_col = [col for col in ES_data.columns if col.count("E_c")]
        dE_dP_col = [col for col in ES_data.columns if col.count("dE_dP") or col.count("dE/dP")]
        M_col = [col for col in ES_data.columns if col.count("M")]
        R_col = [col for col in ES_data.columns if col.startswith("R") or col.startswith(" R")]
        yR_col = [col for col in ES_data.columns if col.count("yR_ext")]
        k2_col = [col for col in ES_data.columns if col.count("k2")]
        Lambda_col = [col for col in ES_data.columns if col.count("L")]
        dens_col = [col for col in ES_data.columns if col.count("rho/rho_sat") or col.count("n_b")]

        # Obtaining the data of the respective columns
        Pc_data_raw = pd.to_numeric(ES_data[Pc_col].squeeze(), errors="coerce")
        Ec_data_raw = pd.to_numeric(ES_data[Ec_col].squeeze(), errors="coerce")
        dE_dP_data_raw = pd.to_numeric(ES_data[dE_dP_col].squeeze(), errors="coerce")
        M_data_raw = pd.to_numeric(ES_data[M_col].squeeze(), errors="coerce")
        R_data_raw = pd.to_numeric(ES_data[R_col].squeeze(), errors="coerce")
        yR_data_raw = pd.to_numeric(ES_data[yR_col].squeeze(), errors="coerce")
        k2_data_raw = pd.to_numeric(ES_data[k2_col].squeeze(), errors="coerce")
        Lambda_data_raw = pd.to_numeric(ES_data[Lambda_col].squeeze(), errors="coerce")
        dens_data_raw = ES_data[dens_col]
                
        # Obtaining the data that do not violate causality
        Pc_caus = Pc_data_raw[dE_dP_data_raw>=1]
        Ec_caus = Ec_data_raw[dE_dP_data_raw>=1]
        dE_dP_caus = dE_dP_data_raw[dE_dP_data_raw>=1]
        M_caus = M_data_raw[dE_dP_data_raw>=1]
        R_caus = R_data_raw[dE_dP_data_raw>=1]
        yR_caus = yR_data_raw[dE_dP_data_raw>=1]
        k2_caus = k2_data_raw[dE_dP_data_raw>=1]
        Lambda_caus = Lambda_data_raw[dE_dP_data_raw>=1]
        dens_caus = dens_data_raw[dE_dP_data_raw>=1]
                
        # Obtaining the data that do violate causality
        Pc_no_caus = Pc_data_raw[dE_dP_data_raw<1]
        Ec_no_caus = Ec_data_raw[dE_dP_data_raw<1]
        dE_dP_no_caus = dE_dP_data_raw[dE_dP_data_raw<1]
        M_no_caus = M_data_raw[dE_dP_data_raw<1]
        R_no_caus = R_data_raw[dE_dP_data_raw<1]
        yR_no_caus = yR_data_raw[dE_dP_data_raw<1]
        k2_no_caus = k2_data_raw[dE_dP_data_raw<1]
        Lambda_no_caus = Lambda_data_raw[dE_dP_data_raw<1]
        dens_no_caus = dens_data_raw[dE_dP_data_raw<1]
 
        
    # Nested list containg the seperated data of the scanned file
    file_data = []
    file_data.append([Pc_caus,Pc_no_caus,Pc_data_raw])
    file_data.append([Ec_caus,Ec_no_caus,Ec_data_raw])
    file_data.append([dE_dP_caus,dE_dP_no_caus,dE_dP_data_raw])
    file_data.append([M_caus,M_no_caus,M_data_raw])
    file_data.append([R_caus,R_no_caus,R_data_raw])
    file_data.append([yR_caus,yR_no_caus,yR_data_raw])
    file_data.append([k2_caus,k2_no_caus,k2_data_raw])
    file_data.append([Lambda_caus,Lambda_no_caus,Lambda_data_raw])
    file_data.append([dens_caus,dens_no_caus,dens_data_raw])

    return file_data

# Defining a function that checks if a list contains a value multiple times (i.e. more than one time)
def check_same_value(data_list):
    """
    Checking if a list contains a value multiple times (i.e. more than one time)
    1. data_list: the list to be checked
    """
    check_idx =  False # initializing the check index with the boolean 'False' value
    n = len(data_list) # taking the length of the given list
    for i in range(0,n):
        if data_list.count(data_list[i])>1:
            check_idx = True # changing the check index value to the boolean 'True' value
            break
    return check_idx

# Defining a class for plotting and sampling data from main NS EOSs
class mainNSdata:
    """
    Handling data from the solution of TOV equations for main Neutron Stars' EOSs:
    1. Plotting M-R curves (2D and 3D) and EOSs curves
    2. Sampling data for regression purposes
    """

    # Constructor of the class
    def __init__(self):
        """
        Initializing the class
        """

        # Appending the list with the names of the 21 main EOSs at a self variable of the class
        self.main_EOSs_names = ["APR-1","BGP","BL-1","BL-2","DH","HHJ-1","HHJ-2","HLPS-2","HLPS-3","MDI-1","MDI-2","MDI-3","MDI-4","NLD","PS","SCVBB","Ska","SkI4","W","WFF-1","WFF-2"]


    # Method that plots the M-R 2D or 3D curve of a main NS EOS
    def plot_MR_curve(self,filename,main_EOS_name,axis_MR,clr_caus='#1f77b4',projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a main Neutron Star's EOS
        1. filename: file where the data of the "main" EOS are stored
        2. main_EOS_name: the name of the main Neutron Star EOS 
        3. axis_MR: the axis that will include the M-R 2D or 3D curve
        4. clr_caus: the color of the points of the M-R 2D or 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """
        
        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
        

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")    

        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type="hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        # Setting the line width of the curve
        line_width = 1.5   
        
        # Plotting the EOS in 2d or 3d space
        if projection=="2d": # 2D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],M_data[0],".",ms=2,color=clr_caus,label=f"{main_EOS_name}")
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],M_data[1],"--",lw=0.75*line_width,color="darkgrey")
        elif projection=="3d": # 3D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_data[0],M_data[0],lw=line_width,color=clr_caus)
            # Plotting the projection on the M-R plane of the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_proj*np.ones_like(Pc_data[0]),M_data[0],"--",lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_data[1],M_data[1],lw=0.75*line_width,color="darkgrey")
            # Plotting the projection on the M-R plane of the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_proj*np.ones_like(Pc_data[1]),M_data[1],"--",lw=0.75*line_width,color="darkgrey")

    # Method that plots the Ec-Pc 2D curve of a main NS EOS
    def plot_EOS_curve(self,filename,main_EOS_name,axis_EOS,clr_caus='#1f77b4'):
        """
        Reading the EOS data from a given file and plot the respective Ec-Pc 2D curve of a main Neutron Star's EOS
        1. filename: file where the data of the "main" EOS are stored
        2. main_EOS_name: the name of the main Neutron Star EOS 
        module script. To scan on another folder, the exact path of the file must be provided.
        3. axis_EOS: the axis that will include the 2D curve of the EOS
        4. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """

        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
      
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type="hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data

        # Setting the line width of the curve depending on the type of EOS
        line_width = 1.5
        
        # Plotting the EOS 
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_EOS.plot(Pc_data[0],Ec_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
        # Plotting the Ec-Pc data that do violate causality
        axis_EOS.plot(Pc_data[1],Ec_data[1],"--",lw=0.75*line_width,color="darkgrey")

    # Method that plots the yR-M 2D curve of a main NS EOS
    def plot_yRM_curve(self,filename,main_EOS_name,axis_yRM,clr_caus='#1f77b4'):
        """
        Reading the EOS data from a given file and plot the respective yR-M 2D curve of a main Neutron Star's EOS
        1. filename: file where the data of the "main" EOS are stored
        2. main_EOS_name: the name of the main Neutron Star EOS 
        module script. To scan on another folder, the exact path of the file must be provided.
        3. axis_EOS: the axis that will include the 2D curve of the EOS
        4. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """

        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
      
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type="hadronic")
        M_data = sol_data[3] # getting the NS Mass data
        yR_data = sol_data[5] # getting the NS k2 data

        # Setting the line width of the curve depending on the type of EOS
        line_width = 1.5
        
        # Plotting the EOS 
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_yRM.plot(M_data[0],yR_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
        # Plotting the Ec-Pc data that do violate causality
        axis_yRM.plot(M_data[1],yR_data[1],"--",lw=0.75*line_width,color="darkgrey")
    
    
    # Method that plots the k2-M 2D curve of a main NS EOS
    def plot_k2M_curve(self,filename,main_EOS_name,axis_k2M,clr_caus='#1f77b4'):
        """
        Reading the EOS data from a given file and plot the respective k2-M 2D curve of a main Neutron Star's EOS
        1. filename: file where the data of the "main" EOS are stored
        2. main_EOS_name: the name of the main Neutron Star EOS 
        module script. To scan on another folder, the exact path of the file must be provided.
        3. axis_EOS: the axis that will include the 2D curve of the EOS
        4. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """

        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
      
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type="hadronic")
        M_data = sol_data[3] # getting the NS Mass data
        k2_data = sol_data[6] # getting the NS k2 data

        # Setting the line width of the curve depending on the type of EOS
        line_width = 1.5
        
        # Plotting the EOS 
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_k2M.plot(M_data[0],k2_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
        # Plotting the Ec-Pc data that do violate causality
        axis_k2M.plot(M_data[1],k2_data[1],"--",lw=0.75*line_width,color="darkgrey")

    # Method that plots the Λ-M 2D curve of a main NS EOS
    def plot_ΛM_curve(self,filename,main_EOS_name,axis_ΛM,clr_caus='#1f77b4'):
        """
        Reading the EOS data from a given file and plot the respective Λ-M 2D curve of a main Neutron Star's EOS
        1. filename: file where the data of the "main" EOS are stored
        2. main_EOS_name: the name of the main Neutron Star EOS 
        module script. To scan on another folder, the exact path of the file must be provided.
        3. axis_EOS: the axis that will include the 2D curve of the EOS
        4. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """

        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
      
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type="hadronic")
        M_data = sol_data[3] # getting the NS Mass data
        Lambda_data = sol_data[7] # getting the NS k2 data

        # Setting the line width of the curve depending on the type of EOS
        line_width = 1.5
        
        # Plotting the EOS 
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_ΛM.plot(M_data[0],Lambda_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
        # Plotting the Ec-Pc data that do violate causality
        axis_ΛM.plot(M_data[1],Lambda_data[1],"--",lw=0.75*line_width,color="darkgrey")          


    # Method the plots the M-R curves of selected main Neutron Star EOSs
    def plot_select_MR(self,main_EOS_files_list,main_EOS_list,colors_list,axis_MR,projection="2d",Pc_proj=0):
        """
        Plotting the M-R curves of selected main Neutron Star EOSs
        1. main_EOS_files_list: list with the names of files with  data of the main Neutron Star EOSs, the respective M-R curves of which are going to be plotted
        2. main_EOS_list: list with the names of the main Neutron Star EOS, the respective M-R curves of which are going to be plotted
        3. colors_list: list with the colors for the M-R curves
        4. axis_MR: the axis that will include the M-R 2D or 3D curves
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'. 
        """ 
        
        # Iterative process scanning for plotting the respective M-R curves of the given main EOSs
        for i in range(0,len(main_EOS_files_list)):
            self.plot_MR_curve(main_EOS_files_list[i],main_EOS_list[i],axis_MR,colors_list[i],projection=projection,Pc_proj=Pc_proj)

    # Method the plots the Ec-Pc curves of selected main Neutron Star EOSs
    def plot_select_EOS(self,main_EOS_files_list,main_EOS_list,colors_list,axis_EOS):
        """
        Plotting the Ec-Pc curves of selected main Neutron Star EOSs
        1. main_EOS_files_list: list with the names of files with data of the main Neutron Star EOSs, the respective Ec-Pc curves of which are going to be plotted
        2. main_EOS_list: list with the names of the main Neutron Star EOS, the respective Ec-Pc curves of which are going to be plotted
        3. colors_list: list with the colors for the M-R curves
        4. axis_EOS: the axis that will include the M-R 2D or 3D curves
        """ 
        
        # Iterative process for plotting the respective Ec-Pc curves of the given main EOSs 
        for i in range(0,len(main_EOS_files_list)):
            self.plot_EOS_curve(main_EOS_files_list[i],main_EOS_list[i],axis_EOS,colors_list[i])

    # Method the plots the yR-M curves of selected main Neutron Star EOSs
    def plot_select_yRM(self,main_EOS_files_list,main_EOS_list,colors_list,axis_EOS):
        """
        Plotting the Ec-Pc curves of selected main Neutron Star EOSs
        1. main_EOS_files_list: list with the names of files with data of the main Neutron Star EOSs, the respective yR-M curves of which are going to be plotted
        2. main_EOS_list: list with the names of the main Neutron Star EOS, the respective Ec-Pc curves of which are going to be plotted
        3. colors_list: list with the colors for the M-R curves
        4. axis_EOS: the axis that will include the M-R 2D or 3D curves
        """ 
        
        # Iterative process for plotting the respective Ec-Pc curves of the given main EOSs 
        for i in range(0,len(main_EOS_files_list)):
            self.plot_yRM_curve(main_EOS_files_list[i],main_EOS_list[i],axis_EOS,colors_list[i])

    # Method the plots the k2-M curves of selected main Neutron Star EOSs
    def plot_select_k2M(self,main_EOS_files_list,main_EOS_list,colors_list,axis_EOS):
        """
        Plotting the Ec-Pc curves of selected main Neutron Star EOSs
        1. main_EOS_files_list: list with the names of files with data of the main Neutron Star EOSs, the respective k2-M curves of which are going to be plotted
        2. main_EOS_list: list with the names of the main Neutron Star EOS, the respective Ec-Pc curves of which are going to be plotted
        3. colors_list: list with the colors for the M-R curves
        4. axis_EOS: the axis that will include the M-R 2D or 3D curves
        """ 
        
        # Iterative process for plotting the respective Ec-Pc curves of the given main EOSs 
        for i in range(0,len(main_EOS_files_list)):
            self.plot_k2M_curve(main_EOS_files_list[i],main_EOS_list[i],axis_EOS,colors_list[i])

    # Method the plots the Λ-M curves of selected main Neutron Star EOSs
    def plot_select_ΛM(self,main_EOS_files_list,main_EOS_list,colors_list,axis_EOS):
        """
        Plotting the Ec-Pc curves of selected main Neutron Star EOSs
        1. main_EOS_files_list: list with the names of files with data of the main Neutron Star EOSs, the respective Λ-M curves of which are going to be plotted
        2. main_EOS_list: list with the names of the main Neutron Star EOS, the respective Ec-Pc curves of which are going to be plotted
        3. colors_list: list with the colors for the M-R curves
        4. axis_EOS: the axis that will include the M-R 2D or 3D curves
        """ 
        
        # Iterative process for plotting the respective Ec-Pc curves of the given main EOSs 
        for i in range(0,len(main_EOS_files_list)):
            self.plot_ΛM_curve(main_EOS_files_list[i],main_EOS_list[i],axis_EOS,colors_list[i])                        

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of main NS EOS
    def sample_MR(self,filename,Pc_threshold=0,M_threshold=0,points_MR=16,violate_caus="no",noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a main Neutron Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the sampling of Mass and Radius values. By default its value is set to 0.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        4. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        5. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        6. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        8. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        9. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'Μ_threshold' argument
        if type(M_threshold)!=type(2) and type(M_threshold)!=type(2.5):
            raise ValueError("The value of the \"M_threshold\" argument must be a number. Try again.")
        elif M_threshold<0:
            raise ValueError("The value of the \"M_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'points_MR' argument
        if type(points_MR)!=type(2):
            raise ValueError("The value of the \"points_MR\" argument must be an integer. Try again.")
        elif points_MR<=0:
            raise ValueError("The value of the \"points_MR\" argument must be positive. Try again.")
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")

        # Allowed values for the 'noiseM_mv' argument
        if type(noiseM_mv)!=type(2) and type(noiseM_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseM_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseM_std' argument
        if type(noiseM_std)!=type(2) and type(noiseM_std)!=type(2.5):
            raise ValueError("The value of the \"noiseM_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_mv' argument
        if type(noiseR_mv)!=type(2) and type(noiseR_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseR_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_std' argument
        if type(noiseR_std)!=type(2) and type(noiseR_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")            

        mass_segments = points_MR-1 # number of Mass range segments
        obs = points_MR # number of observations in the samples

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,EOS_type="hadronic")
            
            # Getting data from the scaned file
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                M_data = list(sol_data[3][0]) # getting the NS Mass data
                R_data = list(sol_data[4][0]) # getting the NS Radius data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                M_data = list(sol_data[3][1]) # getting the NS Mass data
                R_data = list(sol_data[4][1]) # getting the NS Radius data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                M_data = list(sol_data[3][2]) # getting the NS Mass data
                R_data = list(sol_data[4][2]) # getting the NS Radius data 
            
            # Sampling Mass and Radius values only if the EOS overcomes the threshold pressure
            # and there are Mass values more than M_threshold
            if Pc_data[-1]>=Pc_threshold and max(M_data)>=M_threshold:

                # Filtering the M_caus data to contain Mass values over M_threshold Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_data) if mass_value>=M_threshold]
                M_data_filt = [M_data[j] for j in idx_filt]

                # Getting the respective Radius values from the R_caus data list
                R_data_filt = [R_data[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_bounds = np.linspace(min(M_data_filt),max(M_data_filt),mass_segments+1)
                
                # Sampling Mass and Radius values at each segment
                for M_bound in M_bounds:
                    M_diff_abs = abs(M_data_filt-M_bound*np.ones_like(M_data_filt))
                    idx_min_diff = np.argmin(M_diff_abs)
                    mass = M_data_filt[idx_min_diff]
                    radius = R_data_filt[idx_min_diff]

                    # Appening to the storage lists
                    mass_sample.append(mass)
                    radius_sample.append(radius)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]                
    

    # Method that samples Slope (dE_dP), Energy density on center data (that do not violate causality) and center pressure at maximum mass, from TOV solution data files of a main NS EOS
    def sample_EOS(self,filename,Pc_points=[10,25,50,75,100,200,300,400,500,600,700,800],violate_caus="no",noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a main Neutron Star EOS and sampling Slope (dE_dP), Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,300,400,500,600,700,800] MeV*fm^-3.
        3. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        4. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        5. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        6. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        7. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")
        
        # Allowed values for the 'noiseSl_mv' argument
        if type(noiseSl_mv)!=type(2) and type(noiseSl_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseSl_std' argument
        if type(noiseSl_std)!=type(2) and type(noiseSl_std)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_mv' argument
        if type(noiseEc_mv)!=type(2) and type(noiseEc_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseEc_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_std' argument
        if type(noiseEc_std)!=type(2) and type(noiseEc_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")
        
        # Getting the number of pressure points
        n = len(Pc_points)

        # Initializing storage lists for the Slope (dE_dP), Energy density on center and center pressure at maximum mass values samples
        dEdP_sample = []
        enrg_dens_sample = []
        dEdP_sample_with_noise = [np.NaN]
        enrg_dens_sample_with_noise =[np.NaN]
        Pc_max_mass = np.NaN
        Ec_max_mass = np.NaN

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"hadronic")

             # Getting data from the scaned file
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][0]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][0]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][0]) # getting the NS mass data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][1]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][1]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][1]) # getting the NS mass data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][2]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][2]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][2]) # getting the NS mass data

            # Sampling Slope (dE_dP) and Energy density on center values if the final pressure of the scaned file is greater
            # than the maximum pressure point in Pc_points list
            if Pc_data[-1]>max(Pc_points):
                for Pc in Pc_points:
                    idx_press_val = Pc_data.index(Pc)
                    dEdP_sample.append(dEdP_data[idx_press_val])
                    enrg_dens_sample.append(Ec_data[idx_press_val])

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)    
                
            # Getting the center pressure and energy density at maximum mass
            idx_max_mass = np.argmax(M_data)
            Pc_max_mass = Pc_data[idx_max_mass]
            Ec_max_mass = Ec_data[idx_max_mass]

        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass,Ec_max_mass]



# Defining a class for plotting and sampling data from polytropic NS EOSs
class polyNSdata:
    """ 
    Handling data from the solution of TOV equations for polytropic Neutron Stars' EOSs:
    1. Plotting M-R curves (2D and 3D) and EOSs curves
    2. Sampling data for regression purposes
    """

    # Constructor of the class
    def __init__(self,Γ_choices_codec=["A","B","C","D"],Γ_choices=[1,2,3,4],num_segments=4,dir_poly="",dir_main=""):
        """
        Initializing the `polyNSdata` class:
        1. Appending the (given) list of available coded choices of the Γ parameter to the self variable 'Γ_choices_codec'
        2. Appending the (given) list of available choices of the Γ parameter to the self variable 'Γ_choices'
        3. Appending the (given) number of mass density ρ segments to the self variable 'num_segments'
        4. Appending the (given) directory of the polytropic EoSs files to self variable "dir_poly"
        5. Appending the (given) directory of the main EoSs files to self variable "dir_main"
        6. Getting and appending the total Γ combinations using the 'Γ_total_combos' method of the 'polyNSdata' class
        """
        self.Γ_choices_codec = Γ_choices_codec
        self.Γ_choices = Γ_choices
        self.num_segments = num_segments
        self.Γ_total_combos_sorted = self.Γ_total_combos()
        self.dir_poly = dir_poly
        self.dir_main = dir_main



    # Method that returns all the combinations of Γ depending on the available Γ choices and the number of mass density segments
    def Γ_total_combos(self):
        """ 
        Method that returns all the combinations of Γ depending on the available Γ choices and the number of mass density segments.
        Notice that the algorithm generates the polytropic models (i.e. the Γ combinations) as in the 'tov_solver_polyNS_par.py' or 
        the 'tov_solver_polyNS_par2.py' scripts.
        """

        # Count the given number of choices for Γ
        n = len(self.Γ_choices_codec)
    
        # Find all the possible combinations of Γ
        total_combos = n**self.num_segments # number of total combinations
        i = 1 # counter
        Γ_selections = [] # storage list for the combinations

        # Random selection of a Γ value per each of the given mass density segments
        Γ_choose = random.choices(self.Γ_choices_codec,k=self.num_segments)

        # Creating the combination in a "XXXX" format
        Γ_selection = ""
        for j in range(0,len(Γ_choose)):
            Γ_selection = Γ_selection + Γ_choose[j]
        Γ_selections.append(Γ_selection)
        i=i+1
        while i<=total_combos:
            # Random selection of a Γ value per each of the given mass density segments
            Γ_choose = random.choices(self.Γ_choices_codec,k=self.num_segments)

            # Creating the combination in a "XXXX" format
            Γ_selection = ""
            for j in range(0,len(Γ_choose)):
                Γ_selection = Γ_selection + Γ_choose[j]

            # Checking if the current combination has already been created   
            if Γ_selections.count(Γ_selection)!=0:
                continue
            else:
                Γ_selections.append(Γ_selection)
                i=i+1

        # Sorting all the combinations in ascending alphabetical order
        Γ_selections_sort = sorted(Γ_selections)

        return Γ_selections_sort
    
    # Method that decodes the coded value of Γ and returns its numerical value
    def Γ_decode(self,Γ_codec):
        """
        Method that decodes the coded value of Γ and returns its numerical value
        1. Γ_codec = coded value of Γ 
        """
        m = len(self.Γ_choices_codec)
        Γ_result = "Codec letter not found"
        for j in range(0,m):
            askii_codec = chr(65+j)
            if Γ_codec==askii_codec:
                Γ_result = self.Γ_choices[j]
                break

        return Γ_result
    
    # Method that calculates and fills the values of mass density at the linear region
    def fill_linear_rho(self,tidal="yes"):
        """
        Method that calculates and fills the values of mass density at the linear region
        1. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["yes", "no"]
        """

        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"

        mainEOSnames = ["HLPS-2","HLPS-3"]

        EOSs_fixed_counter = 0

        for mainEOSname in mainEOSnames:
            for Γ_combo in self.Γ_total_combos_sorted:
                filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                df_EOS = pd.read_csv(filename)
                #print(df_EOS)

                Pc_col = [col for col in df_EOS.columns if col.count("P_c")]
                Ec_col = [col for col in df_EOS.columns if col.count("E_c")]
                dE_dP_col = [col for col in df_EOS.columns if col.count("dE_dP") or col.count("dE/dP")]
                dens_col = [col for col in df_EOS.columns if col.count("rho/rho_sat") or col.count("n_b")]

                Pc_data_raw = pd.to_numeric(df_EOS[Pc_col].squeeze(), errors="coerce")
                Ec_data_raw = pd.to_numeric(df_EOS[Ec_col].squeeze(), errors="coerce")
                dE_dP_data_raw = pd.to_numeric(df_EOS[dE_dP_col].squeeze(), errors="coerce")
                dens_data_raw = pd.to_numeric(df_EOS[dens_col].squeeze(), errors="coerce")

                # Getting the data of the linear region
                Pc_data_lin = Pc_data_raw[dE_dP_data_raw==1]
                Ec_data_lin = Ec_data_raw[dE_dP_data_raw==1]
                dE_dP_data_lin = dE_dP_data_raw[dE_dP_data_raw==1]
                dens_data_lin = dens_data_raw[dE_dP_data_raw==1]
                idx_data_lin = Pc_data_lin.index

                

                #print(dens_data_lin)

                # Finding the transition central pressure, energy density and mass density
                if dens_data_lin.empty:
                    continue
                else:
                    idx_tr = idx_data_lin[0]-1
                    P_tr = Pc_data_raw.iloc[idx_tr]
                    E_tr = Ec_data_raw.iloc[idx_tr]
                    rho_tr = dens_data_raw.iloc[idx_tr]

                print(f"P_tr:{P_tr}, E_tr:{E_tr}, ρ_tr:{rho_tr}, EOS: {mainEOSname}_{Γ_combo}") 

                EOSs_fixed_counter +=1
                   

                # Calculating the mass density on linear region
                dens_data_lin_calc = n_b_transition(P=Pc_data_lin.iloc[:],rho_frac_tr=rho_tr,P_tr=P_tr,E_tr=E_tr)/n_sat

                #print(dens_data_lin_calc)

                # Replacing the "-" values of mass density in original DataFrame with the calculated values of mass density
                df_EOS.loc[idx_data_lin,dens_col] = dens_data_lin_calc

                #print(df_EOS)

                # Updating the file with the new filled DataFrame
                df_EOS.to_csv(filename,index=False)

                

        # Print message of completion
        print(f"All {EOSs_fixed_counter} out of {2*len(self.Γ_total_combos_sorted)} EOS files have been updated, icluding values of mass density at linear region !!!")          


    
    # Method that plots a M-R 2D or 3D curve of a main or a polytropic NS EOS
    def plot_MR_curve(self,filename,axis_MR,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0,Mmax_threshold=0,projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D or 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        8. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        9. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'Mmax_treshold' argument
        if type(Mmax_threshold)!=type(2) and type(Mmax_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Mmax_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure 
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8    
        
        # Plotting only if EOS overcomes the threshold pressure and its maximum mass the maximum mass threshold
        if Pc_data[2].iloc[-1]>=Pc_threshold and max(M_data[2])>=Mmax_threshold:
            EOS_overcome = 1

            if projection=="2d": # 2D-plotting
                # Plotting the M-R data that do not violate causality
                if curve_label==None:
                    axis_MR.plot(R_data[0],M_data[0],lw=line_width,color=clr_caus)
                else:
                    axis_MR.plot(R_data[0],M_data[0],lw=line_width,color=clr_caus,label=curve_label)    
                # Plotting the M-R data that do violate causality
                axis_MR.plot(R_data[1],M_data[1],lw=line_width,color="darkgrey")
            elif projection=="3d": # 3D-plotting
                # Plotting the M-R data that do not violate causality
                axis_MR.plot(R_data[0],Pc_data[0],M_data[0],lw=line_width,color=clr_caus)
                # Plotting the projection on the M-R plane of the M-R data that do not violate causality
                axis_MR.plot(R_data[0],Pc_proj*np.ones_like(Pc_data[0]),M_data[0],"--",lw=line_width,color=clr_caus)
                # Plotting the M-R data that do violate causality
                axis_MR.plot(R_data[1],Pc_data[1],M_data[1],lw=line_width,color="darkgrey")
                # Plotting the projection on the M-R plane of the M-R data that do violate causality
                axis_MR.plot(R_data[1],Pc_proj*np.ones_like(Pc_data[1]),M_data[1],"--",lw=line_width,color="darkgrey")

        return EOS_overcome

    # Method that plots the Ec-Pc 2D curve of a main or a polytropic NS EOS
    def plot_EOS_curve(self,filename,axis_EOS,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0,Mmax_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Ec-Pc 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'Mmax_treshold' argument
        if type(Mmax_threshold)!=type(2) and type(Mmax_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Mmax_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure 

        #print(EOS_type) 
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data
        M_data = sol_data[3] # getting the NS Mass data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure and its maximum mass the maximum mass threshold
        if Pc_data[2].iloc[-1]>=Pc_threshold and max(M_data[2])>=Mmax_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            if curve_label==None:
                axis_EOS.plot(Pc_data[0],Ec_data[0],lw=line_width,color=clr_caus)
            else:
                axis_EOS.plot(Pc_data[0],Ec_data[0],lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the Ec-Pc data that do violate causality
            axis_EOS.plot(Pc_data[1],Ec_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome

    # Method that plots the Slope (dE_dP) vs Pressure 2D curve of a main or a polytropic NS EOS
    def plot_dEdP_curve(self,filename,axis_slope,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Slope (dE_dP) vs Pressure 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_slope: the axis that will include the Slope (dE_dP) vs Pressure 2D curve
        3. clr_caus: the color of the points of the Slope (dE_dP) vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","main_tdl","polytropic","polytropic_tdl"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the c_s-Pc data that do not violate causality
            if curve_label==None:
                axis_slope.plot(Pc_data[0],dEdP_data[0],lw=line_width,color=clr_caus)
            else:
                axis_slope.plot(Pc_data[0],dEdP_data[0],lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the c_s-Pc data that do violate causality
            axis_slope.plot(Pc_data[1],dEdP_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome                 

    # Method that plots the Speed of sound vs Pressure 2D curve of a main or a polytropic NS EOS
    def plot_cs_curve(self,filename,axis_cs,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Speed of sound vs Pressure 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_cs: the axis that will include the Speed of sound vs Pressure 2D curve
        3. clr_caus: the color of the points of the Speed of sound vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,'hadronic')
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Calculating the speed of sound (c_s) values from the slope data
        cs_caus = np.sqrt(1/np.array(dEdP_data[0])) # values that do not violate causality
        cs_no_caus = np.sqrt(1/np.array(dEdP_data[1])) # values that do violate causality

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the c_s-Pc data that do not violate causality
            if curve_label==None:
                axis_cs.plot(Pc_data[0],cs_caus,lw=line_width,color=clr_caus)
            else:
                axis_cs.plot(Pc_data[0],cs_caus,lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the c_s-Pc data that do violate causality
            axis_cs.plot(Pc_data[1],cs_no_caus,lw=line_width,color="darkgrey")
            

        return EOS_overcome

    # Method that plots the yR-M 2D curve of a main or a polytropic NS EOS
    def plot_yRM_curve(self,filename,axis_yRM,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective yR-M curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_yRM: the axis that will include the yR-M 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        yR_data = sol_data[5] # getting the NS yR data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            if curve_label==None:
                axis_yRM.plot(M_data[0],yR_data[0],lw=line_width,color=clr_caus)
            else:
                axis_yRM.plot(M_data[0],yR_data[0],lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the Ec-Pc data that do violate causality
            axis_yRM.plot(M_data[1],yR_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome

    # Method that plots the k2-M 2D curve of a main or a polytropic NS EOS
    def plot_k2M_curve(self,filename,axis_k2M,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective k2-M curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_k2M: the axis that will include the k2-M 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        k2_data = sol_data[6] # getting the NS k2 data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            if curve_label==None:
                axis_k2M.plot(M_data[0],k2_data[0],lw=line_width,color=clr_caus)
            else:
                axis_k2M.plot(M_data[0],k2_data[0],lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the Ec-Pc data that do violate causality
            axis_k2M.plot(M_data[1],k2_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome

    # Method that plots the Λ-M 2D curve of a main or a polytropic NS EOS
    def plot_ΛM_curve(self,filename,axis_ΛM,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Λ-M curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_ΛΜ: the axis that will include the Λ-M 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main","polytropic"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        Λ_data = sol_data[7] # getting the NS Λ data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            if curve_label==None:
                axis_ΛM.plot(M_data[0],Λ_data[0],lw=line_width,color=clr_caus)
            else:
                axis_ΛM.plot(M_data[0],Λ_data[0],lw=line_width,color=clr_caus,label=curve_label)    
            # Plotting the Ec-Pc data that do violate causality
            axis_ΛM.plot(M_data[1],Λ_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome             


    # Method that plots the M-R 2D or 3D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_poly_MR(self,axis_MR,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0,Mmax_threshold=0,projection="2d",Pc_proj=0):
        """ 
        Plotting the M-R curves of polytropic Neutron Stars for a certain main EOS.
        1. axis_MR: the axis that will include the M-R 2D or 3D curves
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the M-R curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' is selected the aformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["yes", "no"]
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        8. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        9. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        
        
        on par with the automated recording process on .csv files during the operation of the 'tov_solver_polyNS_par.py' or 
        the 'tov_solver_polyNS_par2.py' scripts. Notice that the algorithm of the 'polyNSdata' class, generates
        the polytropic models (i.e. the Γ combinations) as in those two scripts.
        """
        
        # Allowed values for the 'mainEOSname' argument
        mainEOSname_allowedvalues = ["HLPS-2","HLPS-3"]
        if mainEOSname not in mainEOSname_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOSname}\" for the \"mainEOSname\" argument. Allowed values are: {mainEOSname_allowedvalues}")
        
        # Allowed values for the 'first_segment_value' argument
        first_segment_value_allowedvalues = ["all"]
        for Γ_choice in self.Γ_choices_codec:
            first_segment_value_allowedvalues.append(Γ_choice)
        if first_segment_value not in first_segment_value_allowedvalues:
            raise ValueError(f"Invalid value \"{first_segment_value}\" for the \"first_segment_value\" argument. Allowed values are: {first_segment_value_allowedvalues}")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            #clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with linear behavior
        elif mainEOSname=="HLPS-3":
            #clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's M-R data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        MR_curve_result = self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus=clr_poly2,EOS_type="polytropic",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + MR_curve_result
                
                # Scanning for data with EOS non-linear behavior at the last mass density segment
                elif linear_behavior=="no":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        MR_curve_result = self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus=clr_poly,EOS_type="polytropic",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)
                        EOSs_over_threshold = EOSs_over_threshold + MR_curve_result             

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print(150*"-")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3] and maximum mass over {Mmax_threshold:.2f} M_sun:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print(150*"-")

        # # Scanning and plotting TOV solution's M-R data for the main EOSs
        # filename = f"{self.dir_main}{mainEOSname}{tidal_suffix}_sol.csv"
        # # Checking wether the file exists or not, and plottting the data if it exists
        # if os.path.exists(filename):
        #     self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus=clr_main,EOS_type="main",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)  

        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=18)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=18)
            axis_MR.set_xbound([4,16]) # displayed radius R values within [4,16] km
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=16)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=16)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([4,20]) # displayed radius R values within [4,20] km
            axis_MR.view_init(25,-125)    

        return []          


    # Method that plots the M-R 2D or 3D curves of the main EOSs HLPS2 and HLPS3
    def plot_main_MR(self,axis_MR,mainEOS="HLPS-2",tidal="yes",Pc_threshold=0,Mmax_threshold=0,projection="2d",Pc_proj=0):
        """ 
        Plotting the M-R curves of the main EOSs HLPS-2 and HLPS-3.
        1. axis_MR: the axis that will include the M-R 2D or 3D curves
        2. mainEOS: the name of the main EOS, by default 'HLPS-2'.
        3. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["yes", "no"]
        4. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        5. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        6. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        7. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.

        The files to be scanned have the general format:
        1. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 

        """

        # Allowed values for the 'mainEOSname' argument
        mainEOS_allowedvalues = ["HLPS-2","HLPS-3","both"]
        if mainEOS not in mainEOS_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOS}\" for the \"mainEOS\" argument. Allowed values are: {mainEOS_allowedvalues}")

        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")

        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"

        # Defining the colors of the plots
        if mainEOS=="HLPS-2":
            clr_main = ["darkgreen"] # color for the M-R curve of the main HLPS-2 EOS
            EOS_models = [mainEOS]
        elif mainEOS=="HLPS-3":
            clr_main = ["gold"] # color for the M-R curve of the main HLPS-3 EOS
            EOS_models = [mainEOS]
        elif mainEOS=="both":
            clr_main = ["darkgreen","gold"] # color for the M-R curve of the main HLPS-3 EOS
            EOS_models = ["HLPS-2","HLPS-3"]


        for i in range(len(clr_main)):
            # Scanning and plotting TOV solution's M-R data for the main EOSs
            filename = f"{self.dir_main}{EOS_models[i]}{tidal_suffix}_sol.csv"
            # Checking wether the file exists or not, and plottting the data if it exists
            if os.path.exists(filename):
                self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus=clr_main[i],EOS_type="main",curve_label=f"{EOS_models[i]}",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)  

        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=18)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=18)
            axis_MR.set_xbound([4,16]) # displayed radius R values within [4,16] km
            axis_MR.legend(fontsize=12)
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=16)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=16)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([4,20]) # displayed radius R values within [4,20] km
            axis_MR.view_init(25,-125)    

        return []               

    
    # Method that plots the EOS 2D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_poly_EOS(self,axis_EOS,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0,Mmax_threshold=0):
        """ 
        Plotting the EOS curves of polytropic Neutron Stars for a certain main EOS.
        1. axis_EOS: the axis that will include the 2D curves of the EOSs
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the EOS curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        3. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 
        
        on par with the automated recording process on .csv files during the operation of 'the tov_solver_polyNS_par.py' or 
        the 'the tov_solver_polyNS_par2.py' script. Notice that the algorithm of the 'polyNSdata' class, generates
        the polytropic models (i.e. the Γ combinations) as in those two scripts.
        """
        
        # Allowed values for the 'mainEOSname' argument
        mainEOSname_allowedvalues = ["HLPS-2","HLPS-3"]
        if mainEOSname not in mainEOSname_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOSname}\" for the \"mainEOSname\" argument. Allowed values are: {mainEOSname_allowedvalues}")
        
        # Allowed values for the 'first_segment_value' argument
        first_segment_value_allowedvalues = ["all"]
        for Γ_choice in self.Γ_choices_codec:
            first_segment_value_allowedvalues.append(Γ_choice)
        if first_segment_value not in first_segment_value_allowedvalues:
            raise ValueError(f"Invalid value \"{first_segment_value}\" for the \"first_segment_value\" argument. Allowed values are: {first_segment_value_allowedvalues}")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            #clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with non-linear behavior
        elif mainEOSname=="HLPS-3":
            #clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with non-linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's Ec-Pc data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                    #print(filename)
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus=clr_poly2,EOS_type="polytropic",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + EOS_curve_result
                
                # Scanning for data with EOS non-linear behavior at the last mass density segment (if selected)
                elif linear_behavior=="no":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus=clr_poly,EOS_type="polytropic",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold)
                        EOSs_over_threshold = EOSs_over_threshold + EOS_curve_result        

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print(150*"-")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3] and maximum mass over {Mmax_threshold:.2f} M_sun:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print(150*"-")

        # Scanning and plotting TOV solution's Ec-Pc data for the main EOS
        # filename = f"{self.dir_main}{mainEOSname}{tidal_suffix}_sol.csv"
        # # Checking wether the file exists or not, and plottting the data if it exists
        # if os.path.exists(filename):
        #     self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus=clr_main,EOS_type="main",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold)
            

        # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_EOS.set_xlabel(r"$P$ $[MeV\cdot fm^{-3}]$",fontsize=16)
        axis_EOS.set_ylabel(r"$\epsilon$ $[MeV\cdot fm^{-3}]$",fontsize=16)
        axis_EOS.set_xscale("log")
        axis_EOS.set_yscale("log")

        return []
    
    # Method that plots the EOS 2D curves of the main EOSs HLPS-2 and HLPS-3
    def plot_main_EOS(self,axis_EOS,mainEOS="HLPS-2",tidal="yes",Pc_threshold=0,Mmax_threshold=0):
        """ 
        Plotting the EOS curves of the main EOSs HLPS-2 and HLPS-3.
        1. axis_MR: the axis that will include the M-R 2D or 3D curves
        2. mainEOS: the name of the main EOS, by default 'HLPS-2'.
        3. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["yes", "no"]
        4. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        5. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.

        The files to be scanned have the general format:
        1. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 

        """

        # Allowed values for the 'mainEOSname' argument
        mainEOS_allowedvalues = ["HLPS-2","HLPS-3","both"]
        if mainEOS not in mainEOS_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOS}\" for the \"mainEOS\" argument. Allowed values are: {mainEOS_allowedvalues}")

        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")

        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"

        # Defining the colors of the plots
        if mainEOS=="HLPS-2":
            clr_main = ["darkgreen"] # color for the M-R curve of the main HLPS-2 EOS
            EOS_models = [mainEOS]
        elif mainEOS=="HLPS-3":
            clr_main = ["gold"] # color for the M-R curve of the main HLPS-3 EOS
            EOS_models = [mainEOS]
        elif mainEOS=="both":
            clr_main = ["darkgreen","gold"] # color for the M-R curve of the main HLPS-3 EOS
            EOS_models = ["HLPS-2","HLPS-3"]


        for i in range(len(clr_main)):
            # Scanning and plotting TOV solution's M-R data for the main EOSs
            filename = f"{self.dir_main}{EOS_models[i]}{tidal_suffix}_sol.csv"
            # Checking wether the file exists or not, and plottting the data if it exists
            if os.path.exists(filename):
                self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus=clr_main[i],EOS_type="main",curve_label=f"{EOS_models[i]}",Pc_threshold=Pc_threshold,Mmax_threshold=Mmax_threshold)  

       # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_EOS.set_xlabel(r"$P$ $[MeV\cdot fm^{-3}]$",fontsize=16)
        axis_EOS.set_ylabel(r"$\epsilon$ $[MeV\cdot fm^{-3}]$",fontsize=16)
        axis_EOS.set_xscale("log")
        axis_EOS.set_yscale("log")
        axis_EOS.legend(fontsize=12)   

        return []

    
    # Method that plots the yR-M 2D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_yRM_single(self,axis_yRM,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0):
        """ 
        Plotting the yR-M curves of polytropic Neutron Stars for a certain main EOS and the corresponding main EOS curve.
        1. axis_yRM: the axis that will include the yR-M 2D curves
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the yR-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        3. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 
        
        on par with the automated recording process on .csv files during the operation of 'the tov_solver_polyNS_par.py' or 
        the 'the tov_solver_polyNS_par2.py' script. Notice that the algorithm of the 'polyNSdata' class, generates
        the polytropic models (i.e. the Γ combinations) as in those two scripts.
        """
        
        # Allowed values for the 'mainEOSname' argument
        mainEOSname_allowedvalues = ["HLPS-2","HLPS-3"]
        if mainEOSname not in mainEOSname_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOSname}\" for the \"mainEOSname\" argument. Allowed values are: {mainEOSname_allowedvalues}")
        
        # Allowed values for the 'first_segment_value' argument
        first_segment_value_allowedvalues = ["all"]
        for Γ_choice in self.Γ_choices_codec:
            first_segment_value_allowedvalues.append(Γ_choice)
        if first_segment_value not in first_segment_value_allowedvalues:
            raise ValueError(f"Invalid value \"{first_segment_value}\" for the \"first_segment_value\" argument. Allowed values are: {first_segment_value_allowedvalues}")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")
        
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with non-linear behavior
        elif mainEOSname=="HLPS-3":
            clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with non-linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's Ec-Pc data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_yRM_curve(filename,axis_yRM=axis_yRM,clr_caus=clr_poly2,EOS_type="polytropic",Pc_threshold=Pc_threshold)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + EOS_curve_result        

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print("-----------------------------------------------------------------------------------------------")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3]:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print("-----------------------------------------------------------------------------------------------")

        # Scanning and plotting TOV solution's Ec-Pc data for the main EOS
        filename = f"{self.dir_main}{mainEOSname}{tidal_suffix}_sol.csv"
        # Checking wether the file exists or not, and plottting the data if it exists
        if os.path.exists(filename):
            self.plot_yRM_curve(filename,axis_yRM=axis_yRM,clr_caus=clr_main,EOS_type="main",Pc_threshold=Pc_threshold)
            

        # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_yRM.set_ylabel(r"$y_R$",fontsize=16)
        axis_yRM.set_xlabel(r"$M$ $(M_\odot)$",fontsize=16)
        # axis_yRM.set_xscale("log")
        # axis_yRM.set_yscale("log")

        return []
    
    # Method that plots the yR-M 2D curves of polytropic EOSs for both main EOSs (HLPS-2 and HLPS-3)
    def plot_yRM_both(self,axis_yRM,first_segment_value="all",linear_behavior="no",tidal="no",Pc_threshold=0):
        """ 
        Plotting the yR-M curves of polytropic Neutron Stars for both HLPS-2 and HLPS-3 main EOSs and the corresponding main EOSs.
        1. axis_yRM: the axis that will include the yR-M 2D curves of the EOSs
        2. first_segment_value: allowing the user to draw only the yR-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        3. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        4. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        """
        self.plot_yRM_single(axis_yRM,"HLPS-2",first_segment_value,linear_behavior,tidal,Pc_threshold)
        self.plot_yRM_single(axis_yRM,"HLPS-3",first_segment_value,linear_behavior,tidal,Pc_threshold)

        return []
    
    # Method that plots the k2-M 2D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_k2M_single(self,axis_k2M,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0):
        """ 
        Plotting the k2-M curves of polytropic Neutron Stars for a certain main EOS and the corresponding main EOS curve.
        1. axis_k2M: the axis that will include the k2-M 2D curves
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the k2-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        3. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 
        
        on par with the automated recording process on .csv files during the operation of 'the tov_solver_polyNS_par.py' or 
        the 'the tov_solver_polyNS_par2.py' script. Notice that the algorithm of the 'polyNSdata' class, generates
        the polytropic models (i.e. the Γ combinations) as in those two scripts.
        """
        
        # Allowed values for the 'mainEOSname' argument
        mainEOSname_allowedvalues = ["HLPS-2","HLPS-3"]
        if mainEOSname not in mainEOSname_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOSname}\" for the \"mainEOSname\" argument. Allowed values are: {mainEOSname_allowedvalues}")
        
        # Allowed values for the 'first_segment_value' argument
        first_segment_value_allowedvalues = ["all"]
        for Γ_choice in self.Γ_choices_codec:
            first_segment_value_allowedvalues.append(Γ_choice)
        if first_segment_value not in first_segment_value_allowedvalues:
            raise ValueError(f"Invalid value \"{first_segment_value}\" for the \"first_segment_value\" argument. Allowed values are: {first_segment_value_allowedvalues}")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")
        
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix = "_tdl"
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with non-linear behavior
        elif mainEOSname=="HLPS-3":
            clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with non-linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's Ec-Pc data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_k2M_curve(filename,axis_k2M=axis_k2M,clr_caus=clr_poly2,EOS_type=f"polytropic",Pc_threshold=Pc_threshold)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + EOS_curve_result        

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print("-----------------------------------------------------------------------------------------------")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3]:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print("-----------------------------------------------------------------------------------------------")

        # Scanning and plotting TOV solution's Ec-Pc data for the main EOS
        filename = f"{self.dir_main}{mainEOSname}{tidal_suffix}_sol.csv"
        # Checking wether the file exists or not, and plottting the data if it exists
        if os.path.exists(filename):
            self.plot_k2M_curve(filename,axis_k2M=axis_k2M,clr_caus=clr_main,EOS_type="main",Pc_threshold=Pc_threshold)
            

        # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_k2M.set_ylabel(r"$k2$",fontsize=16)
        axis_k2M.set_xlabel(r"$M$ $(M_\odot)$",fontsize=16)
        # axis_yRM.set_xscale("log")
        # axis_yRM.set_yscale("log")

        return []
    
    # Method that plots the k2-M 2D curves of polytropic EOSs for both main EOSs (HLPS-2 and HLPS-3)
    def plot_k2M_both(self,axis_k2M,first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0):
        """ 
        Plotting the k2-M curves of polytropic Neutron Stars for both HLPS-2 and HLPS-3 main EOSs and the corresponding main EOSs.
        1. axis_k2M: the axis that will include the k2-M 2D curves of the EOSs
        2. first_segment_value: allowing the user to draw only the k2-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        3. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        4. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        """
        self.plot_k2M_single(axis_k2M,"HLPS-2",first_segment_value,linear_behavior,tidal,Pc_threshold)
        self.plot_k2M_single(axis_k2M,"HLPS-3",first_segment_value,linear_behavior,tidal,Pc_threshold)

        return []
    
    # Method that plots the Λ-M 2D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_ΛM_single(self,axis_ΛM,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0):
        """ 
        Plotting the Λ-M curves of polytropic Neutron Stars for a certain main EOS and the corresponding main EOS curve.
        1. axis_ΛM: the axis that will include the Λ-M 2D curves
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the Λ-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        3. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 
        
        on par with the automated recording process on .csv files during the operation of 'the tov_solver_polyNS_par.py' or 
        the 'the tov_solver_polyNS_par2.py' script. Notice that the algorithm of the 'polyNSdata' class, generates
        the polytropic models (i.e. the Γ combinations) as in those two scripts.
        """
        
        # Allowed values for the 'mainEOSname' argument
        mainEOSname_allowedvalues = ["HLPS-2","HLPS-3"]
        if mainEOSname not in mainEOSname_allowedvalues:
            raise ValueError(f"Invalid value \"{mainEOSname}\" for the \"mainEOSname\" argument. Allowed values are: {mainEOSname_allowedvalues}")
        
        # Allowed values for the 'first_segment_value' argument
        first_segment_value_allowedvalues = ["all"]
        for Γ_choice in self.Γ_choices_codec:
            first_segment_value_allowedvalues.append(Γ_choice)
        if first_segment_value not in first_segment_value_allowedvalues:
            raise ValueError(f"Invalid value \"{first_segment_value}\" for the \"first_segment_value\" argument. Allowed values are: {first_segment_value_allowedvalues}")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")
        
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Tidal suffix
        tidal_suffix = " "
        if tidal=="yes":
            tidal_suffix = "_tdl"
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with non-linear behavior
        elif mainEOSname=="HLPS-3":
            clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with non-linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's Ec-Pc data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{self.dir_poly}{mainEOSname}_{Γ_combo}L{tidal_suffix}_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_ΛM_curve(filename,axis_ΛM=axis_ΛM,clr_caus=clr_poly2,EOS_type="polytropic",Pc_threshold=Pc_threshold)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + EOS_curve_result        

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print("-----------------------------------------------------------------------------------------------")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3]:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print("-----------------------------------------------------------------------------------------------")

        # Scanning and plotting TOV solution's Ec-Pc data for the main EOS
        filename = f"{self.dir_main}{mainEOSname}{tidal_suffix}_sol.csv"
        # Checking wether the file exists or not, and plottting the data if it exists
        if os.path.exists(filename):
            self.plot_ΛM_curve(filename,axis_ΛM=axis_ΛM,clr_caus=clr_main,EOS_type="main",Pc_threshold=Pc_threshold)
            

        # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_ΛM.set_ylabel(r"$\Lambda$",fontsize=16)
        axis_ΛM.set_xlabel(r"$M$ $(M_\odot)$",fontsize=16)
        # axis_yRM.set_xscale("log")
        axis_ΛM.set_yscale("log")

        return []
    
    # Method that plots the Λ-M 2D curves of polytropic EOSs for both main EOSs (HLPS-2 and HLPS-3)
    def plot_ΛM_both(self,axis_ΛM,first_segment_value="all",linear_behavior="yes",tidal="yes",Pc_threshold=0):
        """ 
        Plotting the Λ-M curves of polytropic Neutron Stars for both HLPS-2 and HLPS-3 main EOSs and the corresponding main EOSs.
        1. axis_ΛM: the axis that will include the Λ-M 2D curves of the EOSs
        2. first_segment_value: allowing the user to draw only the Λ-M curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        3. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        4. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        """
        self.plot_ΛM_single(axis_ΛM,"HLPS-2",first_segment_value,linear_behavior,tidal,Pc_threshold)
        self.plot_ΛM_single(axis_ΛM,"HLPS-3",first_segment_value,linear_behavior,tidal,Pc_threshold)

        return []

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of a polytropic NS EOS
    def sample_MR(self,filename,Pc_threshold=0,M_threshold=0,points_MR=16,violate_caus="no",noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a polytropic Neutron Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the sampling of Mass and Radius values. By default its value is set to 0.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        4. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        5. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        6. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        8. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        9. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'Μ_treshold' argument
        if type(M_threshold)!=type(2) and type(M_threshold)!=type(2.5):
            raise ValueError("The value of the \"M_threshold\" argument must be a number. Try again.")
        elif M_threshold<0:
            raise ValueError("The value of the \"M_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'points_MR' argument
        if type(points_MR)!=type(2):
            raise ValueError("The value of the \"points_MR\" argument must be an integer. Try again.")
        elif points_MR<=0:
            raise ValueError("The value of the \"points_MR\" argument must be positive. Try again.")
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")

        # Allowed values for the 'noiseM_mv' argument
        if type(noiseM_mv)!=type(2) and type(noiseM_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseM_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseM_std' argument
        if type(noiseM_std)!=type(2) and type(noiseM_std)!=type(2.5):
            raise ValueError("The value of the \"noiseM_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_mv' argument
        if type(noiseR_mv)!=type(2) and type(noiseR_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseR_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_std' argument
        if type(noiseR_std)!=type(2) and type(noiseR_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")            

        mass_segments = points_MR-1 # number of Mass range segments
        obs = points_MR # number of observations in the samples

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"hadronic")

            # Getting data from the scaned file
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                M_data = list(sol_data[3][0]) # getting the NS Mass data
                R_data = list(sol_data[4][0]) # getting the NS Radius data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                M_data = list(sol_data[3][1]) # getting the NS Mass data
                R_data = list(sol_data[4][1]) # getting the NS Radius data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                M_data = list(sol_data[3][2]) # getting the NS Mass data
                R_data = list(sol_data[4][2]) # getting the NS Radius data   

            # Sampling Mass and Radius values only if the EOS overcomes the threshold pressure
            # and there are Mass values more than M_threshold
            if Pc_data[-1]>=Pc_threshold and max(M_data)>=M_threshold:

                # Filtering the M_raw data to contain Mass values over M_threshold Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_data) if mass_value>=M_threshold]
                M_data_filt = [M_data[j] for j in idx_filt]

                # Getting the respective Radius values from the R_raw data list
                R_data_filt = [R_data[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_bounds = np.linspace(min(M_data_filt),max(M_data_filt),mass_segments+1)
                
                # Sampling Mass and Radius values at each segment
                for M_bound in M_bounds:
                    M_diff_abs = abs(M_data_filt-M_bound*np.ones_like(M_data_filt))
                    idx_min_diff = np.argmin(M_diff_abs)
                    mass = M_data_filt[idx_min_diff]
                    radius = R_data_filt[idx_min_diff]

                    # Appening to the storage lists
                    mass_sample.append(mass)
                    radius_sample.append(radius)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]
    

    # Method that samples Slope (dE_dP), Energy density on center data (that do not violate causality) and center pressure at maximum mass, from TOV solution data files of a polytropic NS EOS
    def sample_EOS(self,filename,Pc_points=[10,25,50,75,100,200,400,800],violate_caus="no",noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a polytropic Neutron Star EOS and sampling Slope (dE_dP), Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,400,800] MeV*fm^-3.
        3. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        4. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        5. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        6. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        7. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")

        # Allowed values for the 'noiseSl_mv' argument
        if type(noiseSl_mv)!=type(2) and type(noiseSl_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseSl_std' argument
        if type(noiseSl_std)!=type(2) and type(noiseSl_std)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_mv' argument
        if type(noiseEc_mv)!=type(2) and type(noiseEc_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseEc_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_std' argument
        if type(noiseEc_std)!=type(2) and type(noiseEc_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")
        
        # Getting the number of pressure points
        n = len(Pc_points)

        # Initializing storage lists for the Slope (dE_dP), Energy density on center and center pressure at maximum mass values samples
        dEdP_sample = []
        enrg_dens_sample = []
        dEdP_sample_with_noise = [np.NaN]
        enrg_dens_sample_with_noise =[np.NaN]
        Pc_max_mass = np.NaN
        Ec_max_mass = np.NaN

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"hadronic")

            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][0]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][0]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][0]) # getting the NS mass data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][1]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][1]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][1]) # getting the NS mass data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][2]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][2]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][2]) # getting the NS mass data          

            # Sampling Slope (dE_dP) and Energy density on center values only if the EOS overcomes 
            # the value of the maximum pressure point in Pc_points list
            if Pc_data[-1]>max(Pc_points):
                for Pc in Pc_points:
                    idx_press_val = Pc_data.index(Pc)
                    dEdP_sample.append(dEdP_data[idx_press_val])
                    enrg_dens_sample.append(Ec_data[idx_press_val])

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)    
                
            # Getting the center pressure and energy density at maximum mass
            idx_max_mass = np.argmax(M_data)
            Pc_max_mass = Pc_data[idx_max_mass]
            Ec_max_mass = Ec_data[idx_max_mass]

            #print(max(M_caus),Ec_max_mass)

            
        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass,Ec_max_mass]

    # Method that samples the predicted maximum mass and the respective energy density at the center of the neutron star for a single EOS
    def sample_max_M(self,filename,violate_caus="both"):
        """

        Scanning a file containing the TOV equations' solution data for a polytropic Neutron Star EOS and collecting the predicted maximum mass
        and the respective energy density at the center of the star
        1. filename: name of the file to be scanned
        2. violate_caus: wether the sampled ,ass values do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        """

        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")
        
        # Allowed values for the 'tidal' argument
        # tidal_allowedvalues = ["no","yes"]
        # if tidal not in tidal_allowedvalues:
        #     raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Initializing variables for the maximum mass and the respective energy density at the center of the star
        max_mass = np.NaN
        Ec_dens = np.NaN
        
        # Scanning for the file
        if os.path.exists(filename):
            # if tidal=="no":
            sol_data = file_read(filename,"hadronic")
            # elif tidal=="yes":
            #     sol_data = file_read(filename,"polytropic_tdl") 

            if violate_caus=="no": # case 1: data that do not violate causality
                Ec_data = list(sol_data[1][0]) # getting the NS Energy Density on center data
                M_data = list(sol_data[3][0]) # getting the NS mass data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Ec_data = list(sol_data[1][1]) # getting the NS Energy Density on center data
                M_data = list(sol_data[3][1]) # getting the NS mass data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Ec_data = list(sol_data[1][2]) # getting the NS Energy Density on center data
                M_data = list(sol_data[3][2]) # getting the NS mass data

            # Getting the maximum mass and the  energy density at the center at star's center
            max_mass = max(M_data)
            idx_max_mass = np.argmax(M_data) 
            Ec_dens = Ec_data[idx_max_mass]

        return [max_mass,Ec_dens]

    # Method that generates and records on .csv files maximum mass and energy density at star's center data
    def gen_max_mass_data(self,save_filename,linear_behavior="no",violate_caus="both",tidal="yes"):
        """
        Sampling and recordinf on .csv files maximum mass and energy density at star's center data
        1. save_filename: the name of the final .csv file, in which the data are being recorded
        2. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["no","yes","both"]
        3. violate_caus: wether the sampled mass values do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        4. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]

        """ 

        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes","both"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")

        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")     
    
        # Creating the file in which the regression data will be recorder and forming its headers
        headers_eos = f"EoS name," # headers for the name of the EOS
        headers_max_mass = "M_max," # headers for the maximum mass values
        headers_Ec_max_mass = "Ec(M_max)\n" # headers for the center energy density at maximum mass

        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_eos +  headers_max_mass + headers_Ec_max_mass
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)
        
        # Linear behavior signs
        if linear_behavior=="no":
            linear_behavior_signs = [""]
        elif linear_behavior=="yes":
            linear_behavior_signs = ["L"]    
        elif linear_behavior=="both":
            linear_behavior_signs = ["","L"]    

        # Main EOSs to be included
        main_EOSs = ["HLPS-2","HLPS-3"]

        # Neutron Stars EOS types to be scanned
        eos_models = []
        for main_EOS in main_EOSs:
            for linear_sign in linear_behavior_signs:
                for Γ_combo in self.Γ_total_combos_sorted:
                    if tidal=="no":
                        eos_models.append(f"{main_EOS}_{Γ_combo}{linear_sign}")
                    elif tidal=="yes":
                        eos_models.append(f"{main_EOS}_{Γ_combo}{linear_sign}_tdl")
        

        # Scanning the files for EOS types
        for eos_model in eos_models:
            # Getting the name of the file to be scanned
            filename = f"{self.dir_poly}{eos_model}_sol.csv"

            # Getting the maximum mass and energy density at star's center data
            max_mass, Ec_dens = self.sample_max_M(filename,violate_caus=violate_caus,tidal=tidal)

            # Appending the collected data as f-string
            row_eos_model = f"{eos_model},"
            row_max_mass = f"{max_mass},"
            row_Ec_dens = f"{Ec_dens}\n" 

            # Getting the total info of the row and recording it to the final .csv file
            row_info = row_eos_model + row_max_mass + row_Ec_dens 

            with open(f"{save_filename}.csv","a+") as file:
                        file.write(row_info)  

        # Printing completion message
        print(f">The recording process of maximum mass data on the \"{save_filename}.csv\" file has been completed !!!")                      
    
    # Method that generates and records on .csv files data of polytropic Neutron Stars for regression purposes
    def gen_reg_data(self,save_filename,samples_per_EOS=1,M_threshold=0,points_MR=16,Pc_points=[10,25,50,75,100,200,300,400,500,600,700,800],linear_behavior="no",violate_caus="no",noises_mv=[0,0,0,0],noises_std=[0,0,0,0]):
        """
        Sampling data of polytropic Neutron Stars for regression purposes and recording them on .csv files
        1. save_filename: the name of the final .csv file, in which the regression data are being recorded
        2. samples_per_EOS: number of samples to be generated per polytropic EOS. Each sample is recorded as a row in the final .csv file and includes the
        selected values of Speed of Sound c_s^2 = dP_dE, Energy Density at center, Mass and Radius. By default, 1 sample is generated per polytropic EOS.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the EOS must have resulted in causality valid 
        Mass values greater than M_threshold
        4. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        5. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,300,400,500,600,700,800] MeV*fm^-3.
        6. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["no","yes","both"]
        7. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        8. noises_mv: list containing the mean values for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the mean values are set to 0.
        9. noises_std: list containing the standard deviations for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the standard deviations are set to 0.
        """ 

        # Allowed lentgh for the noises_mv list
        if len(noises_mv)!=4:
            raise ValueError(f"The length of the \"noises_mv\" list must be 4, but a list with length {len(noises_mv)} has been given.")

        # Allowed lentgh for the noises_std list
        if len(noises_std)!=4:
            raise ValueError(f"The length of the \"noises_std\" list must be 4, but a list with length {len(noises_std)} has been given.")
        
        # Allowed values for the 'linear_behavior' argument
        linear_behavior_allowedvalues = ["no","yes","both"]
        if linear_behavior not in linear_behavior_allowedvalues:
            raise ValueError(f"Invalid value \"{linear_behavior}\" for the \"linear_behavior\" argument. Allowed values are: {linear_behavior_allowedvalues}")

        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")
        
        # Getting the mean values and the standard deviations of the observational noises
        obs_noiseM_mv = noises_mv[0] # mean value for the noise of the Mass values
        obs_noiseR_mv = noises_mv[1] # mean value for the noise of the Radius values
        obs_noiseSl_mv = noises_mv[2] # mean value for the noise of the Slope (dP_dE) values
        obs_noiseEc_mv = noises_mv[3] # mean value for the noise of the Energy Density at center values

        obs_noiseM_std = noises_std[0] # standard deviation for the noise of the Mass values
        obs_noiseR_std = noises_std[1] # standard deviation for the noise of the Radius values
        obs_noiseSl_std = noises_std[2] # standard deviation for the noise of the Slope (dP_dE) values 
        obs_noiseEc_std = noises_std[3] # standard deviation for the noise of the Energy Density at center values

        # Getting the number of M-R points
        num_mr_points = points_MR

        # Getting the number of pressure points
        num_pc_points = len(Pc_points)

        # Creating the file in which the regression data will be recorder and forming its headers
        headers_eos = f"EoS name," # headers for the name of the EOS
        headers_dPdE = f"" # headers for the Slope (dP_dE) values
        headers_enrg = f"" # headers for the Energy Density at center values
        headers_Pc_max_mass = "Pc(M_max)," # headers for the center pressure at maximum mass
        headers_Ec_max_mass = "Ec(M_max)," # headers for the center enerhy density at maximum mass
        headers_Γ = f"" # headers for the values of the polytropic parameter Γ in the pressure segments
        headers_mass = f"" # headers for the Mass values
        headers_radius = f"" # headers for the Radius values
        

        # Forming the headers for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
        for i in range(0,num_pc_points):
            headers_dPdE = headers_dPdE + f"dP_dE({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis
            headers_enrg = headers_enrg + f"E_c({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis 

        # Forming the headers of the values of the polytropic parameter Γ in the pressure segments
        for i in range(0,self.num_segments):
            headers_Γ = headers_Γ + f"Gamma_{i+1},"          

        # Forming the headers for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
        for i in range(0,num_mr_points):
            headers_mass = headers_mass + f"M_{i+1},"
            # the last column of the headers needs \n and not a comma in the end
            if i==num_mr_points-1:
                headers_radius = headers_radius + f"R_{i+1}\n"
            else:
                headers_radius = headers_radius + f"R_{i+1},"   

               
        
        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_eos + headers_dPdE + headers_enrg + headers_Pc_max_mass + headers_Ec_max_mass + headers_Γ + headers_mass + headers_radius
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)
        
        # Creating a copy .csv file where the values in the columns of the X data are shuffled rowwise to avoid correlation between them 
        with open(f"{save_filename}_rwshuffled.csv","w") as file:
            file.write(headers_info)

        # Linear behavior signs
        if linear_behavior=="no":
            linear_behavior_signs = [""]
        elif linear_behavior=="yes":
            linear_behavior_signs = ["L"]    
        elif linear_behavior=="both":
            linear_behavior_signs = ["","L"]    

        # Main EOSs to be included
        main_EOSs = ["HLPS-2","HLPS-3"]

        # Neutron Stars EOS types to be scanned
        eos_models = []
        for main_EOS in main_EOSs:
            for linear_sign in linear_behavior_signs:
                for Γ_combo in self.Γ_total_combos_sorted:
                    eos_models.append(f"{main_EOS}_{Γ_combo}{linear_sign}")
        
        #print(eos_models)
        
        # Shuffling randomly the Γ_combos
        random.seed(13) # fixing the seed for random shuffling
        random.shuffle(eos_models)
        #print(eos_models)

        # Scanning the files for EOS types
        for eos_model in eos_models:
            # Getting the name of the file to be scanned
            filename = f"{self.dir_poly}{eos_model}_sol.csv"

            # Getting the basic sample of the Slope (dE_dP) and Energy Density at center values
            dEdP_basic_sample,enrg_basic_sample,Pc_max_mass,Ec_max_mass= self.sample_EOS(filename,Pc_points,noiseSl_mv=0,violate_caus=violate_caus,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0)

            # Getting the basic sample of the Mass and Radius values
            mass_basic_sample,radius_basic_sample = self.sample_MR(filename,max(Pc_points)+50,M_threshold,points_MR,violate_caus=violate_caus,noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0)
                    
            # print(slope_basic_sample)
            # print(enrg_basic_sample)
            # print(mass_basic_sample)
            # print(radius_basic_sample)

            # If any of the basic samples is NaN the algorithm skips the recording for this polytropic EOS and moves to the next polytrtopic EOS
            if dEdP_basic_sample[0]==np.NaN or enrg_basic_sample[0]==np.NaN or mass_basic_sample[0]==np.NaN or radius_basic_sample[0]==np.NaN:
                break
            else:
                j=1 # intiliazing a counter for the samples to be made per polytropic EOS
                while j<=samples_per_EOS:
                    # Adding noise to the values of the main samples and recording the resuted sample as a row in the final .csv file
                    dPdE_sample_with_noise = 1/np.array(dEdP_basic_sample) + np.random.normal(loc=obs_noiseSl_mv,scale=obs_noiseSl_std,size=num_pc_points)
                    enrg_sample_with_noise = enrg_basic_sample + np.random.normal(loc=obs_noiseEc_mv,scale=obs_noiseEc_std,size=num_pc_points)
                    mass_sample_with_noise = np.abs(mass_basic_sample + np.random.normal(loc=obs_noiseM_mv,scale=obs_noiseM_std,size=num_mr_points)) # getting the absolute value to ensure positive values for the Mass
                    radius_sample_with_noise = radius_basic_sample + np.random.normal(loc=obs_noiseR_mv,scale=obs_noiseR_std,size=num_mr_points)
                            
                    # Initializing the row info for the basic .csv file
                    row_eos_info = f"{eos_model}," # row info for the name of EOS model
                    row_dPdE_info = f"" # row info for the Slope (dP_dE) values
                    row_enrg_info = f"" # row info for the Energy Density at center values
                    row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                    row_Ec_max_mass_info = f"{Ec_max_mass}," # row info for the center energy density at maximum mass
                    row_Γ_info = f"" # row info for the values of the polytropic parameter Γ in the pressure segments 
                    row_mass_info = f"" # row info for the Mass values
                    row_radius_info = f"" # row info for the Radius values
                            

                    # Initializing the row info for the shuffled .csv file
                    shuffled_row_eos_info = f"{eos_model}," # row info for the name of EOS model
                    shuffled_row_dPdE_info = f"" # row info for the Slope (dP_dE) values
                    shuffled_row_enrg_info = f"" # row info for the Energy Density at center values
                    shuffled_row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                    shuffled_row_Ec_max_mass_info = f"{Ec_max_mass}," # row info for the center energy density at maximum mass
                    shuffled_row_Γ_info = f"" # row info for the values of the polytropic parameter Γ in the pressure segments 
                    shuffled_row_mass_info = f"" # row info for the Mass values
                    shuffled_row_radius_info = f"" # row info for the Radius values
                            

                    # Getting the row info for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
                    for k in range(0,num_pc_points):
                        row_dPdE_info = row_dPdE_info + f"{dPdE_sample_with_noise[k]},"
                        row_enrg_info = row_enrg_info + f"{enrg_sample_with_noise[k]},"
                        shuffled_row_dPdE_info = shuffled_row_dPdE_info + f"{dPdE_sample_with_noise[k]},"
                        shuffled_row_enrg_info = shuffled_row_enrg_info + f"{enrg_sample_with_noise[k]},"
                    
                    Γ_combo = eos_model.removeprefix("HLPS-2_").removeprefix("HLPS-3_").removesuffix("L")
                    # Getting the row info for the Γ parameter data
                    for k in range(0,self.num_segments):
                        row_Γ_info = row_Γ_info + f"{self.Γ_decode(Γ_combo[k])},"
                        shuffled_row_Γ_info = shuffled_row_Γ_info + f"{self.Γ_decode(Γ_combo[k])},"    
                            
                    # Getting the row info for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
                    idx_mr = list(np.arange(0,num_mr_points)) # defining a list with the column indices of the mass/radius values in the basic mass/radius samples
                    np.random.shuffle(idx_mr) # shuffling randomly the column indices of the mass/radius values to reduce/avoid linear correlations between the columns 
                    mr_points_count = 1
                    for k in idx_mr:
                        row_mass_info = row_mass_info + f"{mass_sample_with_noise[mr_points_count-1]},"
                        shuffled_row_mass_info = shuffled_row_mass_info + f"{mass_sample_with_noise[k]},"
                        # the last column of the row needs \n and not a comma in the end
                        if mr_points_count==num_mr_points:
                            row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]}\n"
                            shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]}\n"
                        else:
                            row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]},"
                            shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]},"    
                        mr_points_count +=1

                                            
                    # Getting the total info of the row and recording it to the final .csv file
                    row_info = row_eos_info + row_dPdE_info + row_enrg_info + row_Pc_max_mass_info +  row_Ec_max_mass_info + row_Γ_info + row_mass_info + row_radius_info
                    with open(f"{save_filename}.csv","a+") as file:
                        file.write(row_info)

                    # Getting the total info of the row and recording it to the final shuffled .csv file
                    shuffled_row_info = shuffled_row_eos_info + shuffled_row_dPdE_info + shuffled_row_enrg_info + shuffled_row_Pc_max_mass_info + shuffled_row_Ec_max_mass_info + shuffled_row_Γ_info + shuffled_row_mass_info + shuffled_row_radius_info
                    with open(f"{save_filename}_rwshuffled.csv","a+") as file:
                        file.write(shuffled_row_info)    

                    j = j + 1 # increasing the counter of the samples per polytropic EOS by 1              
        
        # Getting the final .csv file and removing all the NaN elements
        reg_df = pd.read_csv(f"{save_filename}.csv")
        reg_df_cleaned = reg_df.dropna()
        reg_df_cleaned.to_csv(f"{save_filename}.csv",index=False)

        # Printing completion message
        print(f">The recording process of regression data on the \"{save_filename}.csv\" file has been completed !!!")

        # Getting the final shuffled .csv file and removing all the NaN elements
        reg_df = pd.read_csv(f"{save_filename}_rwshuffled.csv")
        reg_df_cleaned = reg_df.dropna()
        reg_df_cleaned.to_csv(f"{save_filename}_rwshuffled.csv",index=False)

        # Printing completion message
        print(f">>The recording process of rowwise shuffled regression data on the \"{save_filename}_rwshuffled.csv\" file has also been completed !!!\n\n")


# Defining a class for plotting and sampling data for Cold Neutron Stars' EoSs from Compose and c_s parametric EoSs
class composeNSdata:
    """ 
    Handling data from the solution of TOV equations for Neutron Stars' EOSs from Compose and c_s parametric EoSs:
    1. Plotting M-R curves (2D and 3D) and EoSs curves
    2. Sampling data for regression purposes

    """
    
    # Initializing the class
    def __init__(self,rseed=10,num_segments=4,num_csgen_EoSs=500,dir_compose_EoSs="",dir_csgen_EoSs=""):
        """
        Initializing the class
        1. Appending the (given) random seed for creating c_s parametric EoSs to the self variable "rs"
        2. Appending the (given) number of mass density ρ segments to the self variable "num_segments"
        3. Appending the (given) number of mock c_s parametric EoSs to the self variable "num_csgen_EoSs"
        4. Appending the (given) directory of Compose EoSs to the self variable 'dir_compose_EoSs'
        5. Appending the (given) directory of mock c_s parametric EoSs to the self variable 'dir_csgen_EoSs'

        """

        self.rseed = rseed
        self.num_segments = num_segments
        self.num_csgen_EoSs = num_csgen_EoSs
        self.dir_compose_EoSs = dir_compose_EoSs
        self.dir_csgen_EoSs = dir_csgen_EoSs

        # Initializing the random seed
        np.random.seed(self.rseed)
        # Producing random seeds
        self.gen_random_seeds = sorted(np.random.randint(0,10*self.num_csgen_EoSs,self.num_csgen_EoSs))

        # Defining dictionaries for CompOSE EoSs as self variables

        # Microscopic Calculations
        self.microEOSs = {"unified":["APR(APR)","BL(chiral)","DDFGOS(APR)"],
             "non-unified":["APR(APR)","BBB(BHF-BBB2)","BL(chiral)"]}
        
        # Relativistic Density Functional Models
        self.relativEOSs = {"nse":["PT(GRDF2-DD2)"],
                "unified":["ABHT(QMC-RMF1)","CMGO (GDFM-I)","CMGO (GDFM-II)","GPPVA(DD2)","GPPVA(FSU2)","GPPVA(FSU2H)","GPPVA(FSU2R)","GPPVA(NL3wrL55)","GPPVA(TM1e)",
                           "GPPVA(TW)","NSXM(DDME-X1)","NSXM(DDME-X2)","NSXM(DDME-Y)","NSXM(PK1r)","SPG(M1)","SPG(M2)","SPG(M3)","SPG(M4)","SPG(M5)",
                           "XMLSLZ(DD-LZ1)","XMLSLZ(DDME-X)","XMLSLZ(DDME2)","XMLSLZ(GM1)","XMLSLZ(MTVTC)","XMLSLZ(NL3)","XMLSLZ(PK1)","XMLSLZ(PKDD)","XMLSLZ(TM1)","XMLSLZ(TW99)"],
                "non-unified":["ABHT(QMC-RMF1)","ABHT(QMC-RMF2)","ABHT(QMC-RMF3)","ABHT(QMC-RMF4)","GDTB(DDHdelta)","GM(GM1)","PCGS(PCSB0)","PCGS(PCSB1)","PCGS(PCSB2)"]}

        # Non Relativistic Density Functional Models
        self.nonrelativEOSs = {"unified":["GMSR(BSK14)","GMSR(BSK16)","GMSR(DHSL59)","GMSR(DHSL69)","GMSR(F0)","GMSR(H1)","GMSR(H2)","GMSR(H3)","GMSR(H4)","GMSR(H5)",
                              "GMSR(H7)","GMSR(LNS5)","GMSR(RATP)","GMSR(SGII)","GMSR(SLy5)","PCP(BSK22)","PCP(BSK24)","PCP(BSK25)","PCP(BSK26)","RG(KDE0v)",
                              "RG(KDE0v1)","RG(Rs)","RG(SK255)","RG(SK272)","RG(SKa)","RG(SKb)","RG(SKI2)","RG(SkI3)","RG(SkI4)","RG(SkI5)",
                              "RG(SkI6)","RG(SkMp)","RG(SkOp)","RG(SLY2)","RG(SLY230a)","RG(SLY4)","RG(SLY9)","VGBCMR(D1M)","VGBCMR(D1MStar)"]}
        


    # Method that plots a M-R 2D or 3D curve of a compose or a c_s parametric Neutron Star EOS
    def plot_MR_curve(self,filename,axis_MR,clr_caus='#1f77b4',EOS_type="compose",curve_label=None,Pc_threshold=0,Mmax_threshold=0,projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D or 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["compose","csgen"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        8. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        9. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["compose","csgen"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")

        # Allowed values for the 'Mmax_treshold' argument
        if type(Mmax_threshold)!=type(2) and type(Mmax_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Mmax_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure 
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic_compose")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "compose":
            marker_size = 3
        else:
            marker_size = 1  
        
        # Plotting only if EOS overcomes the threshold pressure and its maximum mass the maximum mass threshold
        if Pc_data[2].iloc[-1]>=Pc_threshold and max(M_data[2])>=Mmax_threshold:
            EOS_overcome = 1

            if projection=="2d": # 2D-plotting
                # Plotting the M-R data that do not violate causality
                if curve_label==None:
                    axis_MR.plot(R_data[0],M_data[0],".",ms=marker_size,color=clr_caus)
                else:
                    axis_MR.plot(R_data[0],M_data[0],".",ms=marker_size,color=clr_caus,label=curve_label)    
                # Plotting the M-R data that do violate causality
                axis_MR.plot(R_data[1],M_data[1],".",ms=marker_size,color="darkgrey")
            elif projection=="3d": # 3D-plotting
                # Plotting the M-R data that do not violate causality
                axis_MR.plot(R_data[0],Pc_data[0],M_data[0],".",ms=marker_size,color=clr_caus)
                # Plotting the projection on the M-R plane of the M-R data that do not violate causality
                axis_MR.plot(R_data[0],Pc_proj*np.ones_like(Pc_data[0]),M_data[0],"*",ms=marker_size,color=clr_caus)
                # Plotting the M-R data that do violate causality
                axis_MR.plot(R_data[1],Pc_data[1],M_data[1],".",ms=marker_size,color="darkgrey")
                # Plotting the projection on the M-R plane of the M-R data that do violate causality
                axis_MR.plot(R_data[1],Pc_proj*np.ones_like(Pc_data[1]),M_data[1],"*",ms=marker_size,color="darkgrey")

        return EOS_overcome
    
    # Method that plots the Ec-Pc 2D curve of a compose or a c_s parametric NS EOS
    def plot_EOS_curve(self,filename,axis_EOS,clr_caus='#1f77b4',EOS_type="polytropic",curve_label=None,Pc_threshold=0,Mmax_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Ec-Pc 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["compose","csgen"]
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        7. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        """
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["compose","csgen"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")

        # Allowed values for the 'Pc_treshold' argument
        if type(Pc_threshold)!=type(2) and type(Pc_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Pc_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'Mmax_treshold' argument
        if type(Mmax_threshold)!=type(2) and type(Mmax_threshold)!=type(2.5):
            raise ValueError("The value of the \"Pc_threshold\" argument must be a number. Try again.")
        elif Mmax_threshold<0:
            raise ValueError("The value of the \"Pc_threshold\" argument can not be negative. Try again.")
    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure 

        #print(EOS_type) 
        
        # Scanning and reading the file
        sol_data = file_read(filename,"hadronic_compose")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data
        M_data = sol_data[3] # getting the NS Mass data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "compose":
            marker_size = 3
        else:
            marker_size = 1
        
        # Plotting only if EOS overcomes the threshold pressure and its maximum mass the maximum mass threshold
        if Pc_data[2].iloc[-1]>=Pc_threshold and max(M_data[2])>=Mmax_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            if curve_label==None:
                axis_EOS.plot(Pc_data[0],Ec_data[0],".",ms=marker_size,color=clr_caus)
            else:
                axis_EOS.plot(Pc_data[0],Ec_data[0],".",ms=marker_size,color=clr_caus,label=curve_label)    
            # Plotting the Ec-Pc data that do violate causality
            axis_EOS.plot(Pc_data[1],Ec_data[1],".",ms=marker_size,color="darkgrey")
            

        return EOS_overcome
    
    # Method for plotting the M-R curves (2D or 3D) of the Compose Neutron Star EoSs 
    def plot_compose_MR(self,axis_MR,EoSs_type="micro",file_suffix="_sol",Pc_treshold=0,Mmax_threshold=0,projection="2d",Pc_proj=0):
        """
        Plotting the M-R curves (2D or 3D) of the Compose Neutron Star EoSs

        1. axis_MR: the axis that will include the M-R 2D or 3D curve
        2. EoSs_type: the type of the compose EoSs to be plotted. Three options are available: "micro" -> microscopic models, "rel" -> relativistic density functional models, "nrel" -> non relativistic density functional models. Default "micro".
        3. file_suffix: the suffix for the filename of the EoSs
        4. Mmax_threshold: Threshold for the maximum mass [M_sun units]. The EosS with maximum mass less than the threshold are not included in the plots.
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """

        # Allowed values for the 'EoSs_type' argument
        EoSs_type_allowedvalues = ["micro","rel","nrel"]
        if EoSs_type not in EoSs_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EoSs_type}\" for the \"EoSs_type\" argument. Allowed values are: {EoSs_type_allowedvalues}")
        
        np.random.seed(self.rseed)
        
        
        # Plotting Microscopic Compose Neutron Star EoSs
        if EoSs_type=="micro":
            EoSs_over_thresholds_unif = 0
            EoSs_over_thresholds_nonunif = 0
            EoS_models = self.microEOSs["unified"] # plotting the unified models
            for EoS_model in EoS_models:
                filename = f"{self.dir_compose_EoSs}{EoS_model}_micro_unif{file_suffix}.csv"
                if os.path.exists(filename):
                    color_EOS = np.random.random(3)
                    EoS_result = self.plot_MR_curve(filename=filename,axis_MR=axis_MR,clr_caus=color_EOS,curve_label=EoS_model+" micro unif.",Pc_threshold=Pc_treshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)
                    EoSs_over_thresholds_unif = EoSs_over_thresholds_unif + EoS_result
            
            EoS_models = self.microEOSs["non-unified"] # plotting the non-unified models
            for EoS_model in EoS_models:
                filename = f"{self.dir_compose_EoSs}{EoS_model}_micro{file_suffix}.csv"
                if os.path.exists(filename):
                    color_EOS = np.random.random(3)
                    EoS_result = self.plot_MR_curve(filename=filename,axis_MR=axis_MR,clr_caus=color_EOS,curve_label=EoS_model+" micro unif.",Pc_threshold=Pc_treshold,Mmax_threshold=Mmax_threshold,projection=projection,Pc_proj=Pc_proj)
                    EoSs_over_thresholds_nonunif = EoSs_over_thresholds_nonunif + EoS_result    

            # Print the number of EOSs over threshold pressure and maximum mass
            print("-----------------------------------------------------------------------------------------------")
            print(f"Available MICROSCOPIC COMPOSE EOSs with pressure over {Pc_treshold:.3f} [MeV*fm^3] and maximum mass greater than {Mmax_threshold} M_sun:")
            print(f"Unified: {EoSs_over_thresholds_unif}")
            print(f"Non-Unified: {EoSs_over_thresholds_nonunif}")
            print("-----------------------------------------------------------------------------------------------")



        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=18)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=18)
            axis_MR.set_xbound([4,20]) # displayed radius R values within [4,16] km
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=16)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=16)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([4,20]) # displayed radius R values within [4,20] km
            axis_MR.view_init(25,-125)        



# Defining a class for plotting and sampling data from CFL matter QS EOSs
class QSdata:
    """ 
    Handling data from the solution of TOV equations for CFL matter Quark Stars' EOSs:
    1. Plotting M-R curves (2D and 3D) and EOSs curves
    2. Sampling data for regression purposes
    """ 

    # Constructor of the class
    def __init__(self,Beff_max=250,Beff_step=5,Delta_max=250,Delta_step=10,dir_mitQS="",dir_cflQS=""):
        """
        Initializing the `QSdata` class:
        1. Setting the minimum value of the Beff parameter to 60 [MeV*fm^-3] for this class. The actual minmum allowed value is [MeV*fm^-3] # according to the MIT bag model 
        2. Setting the minimum value of the Δ parameter of the MIT bag model to 50 [MeV] for this class. 
        3. Appending the (given) maximum value of the Beff parameter to the self variable 'Beff_max'
        4. Appending the (given) step value for scanning the Beff-axis area to the self variable 'Beff_step'
        5. Appending the (given) maximum value of the Δ parameter to the self variable 'Delta_max'
        6. Appending the (given) step value for scanning the Δ-axis area to the self variable 'Delta_step'
        7. Getting the total valid CFL models and their respective Beff and Δ combinations using the 'valid_cfl_models'
        of the 'cflQSdata' class. Appending the information of the models to the self variable 'total_cfl_models_info'
        8. Getting and appending the (given) directory of the MIT-bag solution files to the self variable "dir_mitQS"
        9. Getting and appending the (given) directory of the CFL solution files to the self variable "dir_cflQS" 

        """
        
        # Minimum value of the B_eff constant in this class, the actual minimum value is 57 [MeV*fm^-3]
        # according to the MIT bag model
        self.Beff_min = 60 # MeV*fm^-3

        # Minimum value of the Δ constant of the MIT bag model for this class
        self.Delta_min = 50 # MeV

        # Allowed values for the Beff_max argument
        if Beff_max<=self.Beff_min:
            raise ValueError(f"The value of \"Beff_max\" argument must be greater than {self.Beff_min:.1f} [MeV*fm^-3]. Try again.")
        
        # Allowed values for the Delta_max argument
        if Delta_max<=self.Delta_min:
            raise ValueError(f"The value of \"Delta_max\" argument must be greater than {self.Delta_min:.1f} [MeV]. Try again.")
        
        self.Beff_max = Beff_max
        self.Beff_step = Beff_step
        self.Delta_max = Delta_max
        self.Delta_step = Delta_step
        self.total_cfl_models_info = self.valid_cfl_models()
        self.total_mitbag_models_info = self.valid_mitbag_models()
        
        self.dir_mitQS = dir_mitQS
        self.dir_cflQS = dir_cflQS


    # Method that calculates the bound of the Beff parameter, dependning on the value of Δ parameter and the mass of Strange quark m_s
    def B_eff_bound(self,Delta):
        formula = -(m_s**2*m_n**2)/(12*(np.pi**2))+(Delta**2*m_n**2)/(3*(np.pi**2))+(m_n**4)/(108*(np.pi**2))
        return formula/hbarc**3

    # Method that scans the Beff-axis area [60,Beff_max] and the Δ-axis area [Delta_min,Delta_max] and returns the valid combinations
    # of the two parameters (Beff and Δ), as well as the names of the respective models
    def valid_cfl_models(self):
        """ 
        Method that scans the Beff-axis area [60,Beff_max] and the Δ-axis area [Delta_min,Delta_max] and returns the valid combinations
        of the three parameters (Beff, Δ and m_s), as well as the names of the respective CFL models. Notice that the algorithm generates
        the CFL models as in the 'tov_solver_cflQS_par.py' script.
        """         
        
        # initializing storage lists for the models' names and the valid values of Beff, Δ and m_s parameters
        models_names = []
        valid_Beff = []
        valid_Delta = []
        valid_ms = []

        i = 1
        B_eff_val = self.Beff_min
        while B_eff_val<=self.Beff_max:
            Delta_val = self.Delta_min
            while Delta_val<=self.Delta_max:
                if B_eff_val < self.B_eff_bound(Delta_val):  
                    #print(f"CFL-{i} ({B_eff_val,Delta_val})")
                    models_names.append(f"CFL-{i}")
                    valid_ms.append(m_s)
                    valid_Beff.append(B_eff_val)
                    valid_Delta.append(Delta_val)
                    i = i + 1
                Delta_val=Delta_val+self.Delta_step    
            B_eff_val=B_eff_val+self.Beff_step    
                    

        return [models_names,valid_Beff,valid_Delta,valid_ms]
    
    # Method that scans the Beff-axis area [60,Beff_max] and returns the combinations
    # of the bag-parameter (Beff), as well as the names of the respective models
    def valid_mitbag_models(self):
        """
        Method that scans the Beff-axis area [60,Beff_max] and returns the combinations
        of the bag-parameter (Beff), as well as the names of the respective models
        """

        # Initializing storage lists for the models' names and the values of Beff bag-parameter
        models_names = []
        valid_Beff = []

        i = 1
        B_eff_val = self.Beff_min
        while B_eff_val<=self.Beff_max:
            #print(f"MITbag-{i} ({B_eff_val})")
            models_names.append(f"MITbag-{i}")
            valid_Beff.append(B_eff_val)
            B_eff_val=B_eff_val+self.Beff_step
            i = i + 1
            

        return [models_names,valid_Beff]    

    
    # Method that displays the valid generated CFL EOSs models info in a PrettyTable format
    def show_cfl_models_info(self):
        models_info = PrettyTable()

        models_info.add_column("Models",self.total_cfl_models_info[0])
        models_info.add_column("B_eff [MeV*fm^-3]",self.total_cfl_models_info[1])
        models_info.add_column("Δ [MeV]",self.total_cfl_models_info[2])
        models_info.add_column("m_s [MeV]",self.total_cfl_models_info[3])

        print(models_info)

    # Method that displays the valid generated CFL EOSs models info in a PrettyTable format
    def show_mitbag_models_info(self):
        models_info = PrettyTable()

        models_info.add_column("Models",self.total_mitbag_models_info[0])
        models_info.add_column("B_eff [MeV*fm^-3]",self.total_mitbag_models_info[1])

        print(models_info)    

    # Method that plots the valid CFL values of the Beff and Δ parameters as points in a Beff-Δ graph
    def plot_valid_cfl_combos(self,point_size=10):
        """
        Plotting the valid CFL values of the Beff and Δ parameters as points in a Beff-Δ graph
        1. point_size: the size of the points in the graph. By default the number 10 is set for the size of the points.
        """
        
        # Allowed values for the 'point_size' argument
        if type(point_size)!=type(2) and type(point_size)!=type(2.5):
            raise ValueError("The value of the \"point_size\" argument must be a number. Try again.")
        
        # Initializing a figure to include the valid CFL points of Beff and Δ parameters
        fig_valid_cfl, axis_valid_cfl = plt.subplots(1,1,figsize=(11,7))

        # Getting and printing the amount of valid CFL models
        total_cfl_models = len(self.total_cfl_models_info[0])
        print(f"Total CFL valid models: {total_cfl_models}")

        # Getting the valid values of Beff and Δ parameters
        Beff_valid_vals = self.total_cfl_models_info[1]
        Delta_valid_vals = self.total_cfl_models_info[2]

        # Plotting the bound curve m_s = 95 [MeV]
        Delta_valid_range = np.linspace(min(Delta_valid_vals)-10,max(Delta_valid_vals)+10,100)
        axis_valid_cfl.plot(Delta_valid_range,self.B_eff_bound(Delta_valid_range),"--",lw=1.5,label=r"$m_s=95$ $[MeV]$")

        # Plotting the bound curve Beff = 57 [MeV*fm^-3]
        axis_valid_cfl.plot(Delta_valid_range,57*np.ones_like(Delta_valid_range),"--",lw=1.5,label=r"$B_{eff}=57$ $[MeV*fm^{-3}]$")

        # Plotting the valid Beff and Δ points
        axis_valid_cfl.plot(Delta_valid_vals,Beff_valid_vals,".",ms=point_size,color="green",label="Valid CFL combos")

        # Coloring the area between the bound curves
        axis_valid_cfl.fill_between(Delta_valid_range, self.B_eff_bound(Delta_valid_range),57*np.ones_like(Delta_valid_range),color="lightgrey",label="Stable CFL quark matter")

        # Setting the axes bounds, labels and legend for clarity
        axis_valid_cfl.set_xbound([min(Delta_valid_range),max(Delta_valid_range)])
        axis_valid_cfl.set_xlabel(r"$\Delta$ $[MeV]$",fontsize=18)
        axis_valid_cfl.set_ylabel(r"$B_{eff}$ $[MeV\cdot fm^{-3}]$",fontsize=18)
        axis_valid_cfl.legend(fontsize=16)

        return fig_valid_cfl,axis_valid_cfl


    # Method that plots a M-R 2D or 3D curve of a MIT bag or CFL matter QS EOS
    def plot_MR_curve(self,filename,axis_MR,clr_caus='#1f77b4',clr_caus_3d='#ff7f0e',curve_label=None,projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a MIT bag or CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. clr_caus_3d: the color of the points of the M-R 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        5. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        6. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        7. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """
        

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")      
        
        # Scanning and reading the file
        sol_data = None
        if filename.count("CFL")>0:
            sol_data = file_read(filename,EOS_type="quark")
        elif filename.count("MITbag")>0:
            sol_data = file_read(filename,EOS_type="quark")   
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        if projection=="2d": # 2D-plotting
            # Plotting the M-R data that do not violate causality
            if curve_label==None:
                axis_MR.plot(R_data[0],M_data[0],lw=0.8,color=clr_caus)
            else:
                axis_MR.plot(R_data[0],M_data[0],lw=0.8,color=clr_caus,label=curve_label)   
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],M_data[1],lw=0.8,color="darkgrey")
        elif projection=="3d": # 3D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_data[0],M_data[0],lw=0.8,color=clr_caus_3d)
            # Plotting the projection on the M-R plane of the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_proj*np.ones_like(Pc_data[0]),M_data[0],"--",lw=0.8,color=clr_caus)
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_data[1],M_data[1],lw=0.8,color="darkgrey")
            # Plotting the projection on the M-R plane of the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_proj*np.ones_like(Pc_data[1]),M_data[1],"--",lw=0.8,color="darkgrey")

        return 1

    # Method that plots an EOS 2D curve of MIT bag or CFL matter QS EOS
    def plot_EOS_curve(self,filename,axis_EOS,clr_caus='#1f77b4',curve_label=None):
        """
        Reading the EOS data from a given file and plot the respective EOS 2D curve of a MIT bag or CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        """
    
        
        # Scanning and reading the file
        sol_data = None
        if filename.count("CFL")>0:
            sol_data = file_read(filename,EOS_type="quark")
        elif filename.count("MITbag")>0:
            sol_data = file_read(filename,EOS_type="quark")   
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data
            
        # Plotting the Ec-Pc data that do not violate causality
        if curve_label==None:
            axis_EOS.plot(Pc_data[0],Ec_data[0],lw=0.8,color=clr_caus)
        else:
            axis_EOS.plot(Pc_data[0],Ec_data[0],lw=0.8,color=clr_caus,label=curve_label)    
        # Plotting the Ec-Pc data that do violate causality
        axis_EOS.plot(Pc_data[1],Ec_data[1],lw=0.8,color="darkgrey")
            

        return 1

    # Method that plots the Slope (dE_dP) vs Pressure 2D curve of a MIT bag or CFL matter QS EOS
    def plot_dEdP_curve(self,filename,axis_slope,clr_caus='#1f77b4',curve_label=None):
        """
        Reading the EOS data from a given file and plot the respective Slope (dE_dP) vs Pressure 2D curve of a MIT bag or CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_slope: the axis that will include the Slope (dE_dP) vs Pressure 2D curve
        3. clr_caus: the color of the points of the Slope (dE_dP) vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        """
    
        
        # Scanning and reading the file
        sol_data = None
        if filename.count("CFL")>0:
            sol_data = file_read(filename,EOS_type="quark")
        elif filename.count("MITbag")>0:
            sol_data = file_read(filename,EOS_type="quark")   
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Plotting the c_s-Pc data that do not violate causality
        if curve_label==None:
            axis_slope.plot(Pc_data[0],dEdP_data[0],lw=0.8,color=clr_caus)
        else:
            axis_slope.plot(Pc_data[0],dEdP_data[0],lw=0.8,color=clr_caus,label=curve_label)    
        # Plotting the c_s-Pc data that do violate causality
        axis_slope.plot(Pc_data[1],dEdP_data[1],lw=0.8,color="darkgrey")
            

        return 1                 

    # Method that plots the Speed of sound vs Pressure 2D curve of a MIT bag or CFL matter QS EOS
    def plot_cs_curve(self,filename,axis_cs,clr_caus='#1f77b4',curve_label=None):
        """
        Reading the EOS data from a given file and plot the respective Speed of sound vs Pressure 2D curve of a MIT bag or CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_cs: the axis that will include the Speed of sound vs Pressure 2D curve
        3. clr_caus: the color of the points of the Speed of sound vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. curve_label: The user can include a label in the plot of the EOS curve. By default: curve_label=None, and no label is added.
        """
        
        # Scanning and reading the file
        sol_data = None
        if filename.count("CFL")>0:
            sol_data = file_read(filename,EOS_type="quark")
        elif filename.count("MITbag")>0:
            sol_data = file_read(filename,EOS_type="quark")   
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Calculating the speed of sound (c_s) values from the slope data
        cs_caus = np.sqrt(1/np.array(dEdP_data[0])) # values that do not violate causality
        cs_no_caus = np.sqrt(1/np.array(dEdP_data[1])) # values that do violate causality
            
        # Plotting the c_s-Pc data that do not violate causality
        if curve_label==None:
            axis_cs.plot(Pc_data[0],cs_caus,lw=0.8,color=clr_caus)
        else:
            axis_cs.plot(Pc_data[0],cs_caus,lw=0.8,color=clr_caus,label=curve_label)   
        # Plotting the c_s-Pc data that do violate causality
        axis_cs.plot(Pc_data[1],cs_no_caus,lw=0.8,color="darkgrey")
            

        return 1            

    # Method that plots the M-R 2D or 3D curves of MIT bag or CFL quark matter EOSs
    def plot_MR(self,axis_MR,idx_min_cfl=1,idx_max_cfl="max",idx_min_mitbag=1,idx_max_mitbag="max",tidal="yes",projection="2d",Pc_proj=0):
        """
        Plotting the M-R curves of MIT bag or CFL matter Quark Stars EOSs.
        1. axis_MR: the axis that will include the M-R curves
        2. idx_min_cfl: minimum enumeration index of CFL EOS models
        3. idx_min_cfl: maximum enumeration index of CFL EOS models
        4. idx_min_mitbag: minimum enumeration index of MIT bag EOS models
        5. idx_max_mitbag: maximum enumeration index of MIT bag EOS models
        6. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        7. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data 
        of the Quark Star. When the 3d option is selected: including additionally the pressure in center data of the Quark Star in a 3rd axis.
        8. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.

        The files to be scanned have the general format:
        1. "CFL-{number of model}_sol.csv"
        2. "MITbag-{number of model}_sol.csv"
        
        
        on par with the automated recording process on .csv files during the operation of the 'tov_solver_QS.py' or 
        the the 'tov_solver_cflQS_par.py' or the 'tov_solver_mitQS_par.py' scripts. Notice though that the algorithm of the 'cflQSdata' class, generates
        the cfl models as in the 'tov_solver_cflQS_par.py' script.
        """

        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        available_cfl_models = 0 # counter of the available CFL models files that have been found during the scan
        available_mitbag_models = 0 # counter of the available MIT bag models files that have been found during the scan

        print("-----------------------------------------------------------------------------------------------")

        # Determinimg the tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix="_tdl"
        
        # Scanning for files containing TOV solution data for CFL EOS models and plotting the respective M-R curves
        if idx_max_cfl=="max":
            idx_max_cfl = len(self.total_cfl_models_info[0])
        for i in range(idx_min_cfl,idx_max_cfl+1):
            filename = f"{self.dir_cflQS}CFL-{i}{tidal_suffix}_sol.csv"
            if os.path.exists(filename):
                MR_curve_result = self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus="cornflowerblue",clr_caus_3d="darkorange",projection=projection,Pc_proj=Pc_proj)
                available_cfl_models = available_cfl_models + MR_curve_result

        # Scanning for files containing TOV solution data for MIT bag EOS models and plotting the respective M-R curves
        if idx_max_mitbag=="max":
            idx_max_mitbag = len(self.total_mitbag_models_info[0])
        for i in range(idx_min_mitbag,idx_max_mitbag+1):
            filename = f"{self.dir_mitQS}MITbag-{i}{tidal_suffix}_sol.csv"
            if os.path.exists(filename):
                MR_curve_result = self.plot_MR_curve(filename,axis_MR=axis_MR,clr_caus="indianred",clr_caus_3d="darkgreen",projection=projection,Pc_proj=Pc_proj)
                available_mitbag_models = available_mitbag_models + MR_curve_result        
        

        # Print the number of available cfl EOSs
        print(f"Available CFL quark matter EOSs: {available_cfl_models}")
        print(f"Available MIT bag quark matter EOSs: {available_mitbag_models}")
        print("-----------------------------------------------------------------------------------------------")

        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=18)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=18)
            axis_MR.set_xbound([0,20]) # displayed radius R values within [0,20] km
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=16)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=16)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=16)
            axis_MR.set_xbound([0,20]) # displayed radius R values within [0,20] km
            axis_MR.view_init(25,-125)    

        return []
    

    # Method that plots the EOS 2D curves of MIT bag or CFL quark matter EOSs
    def plot_EOSs(self,axis_EOS,idx_min_cfl=1,idx_max_cfl="max",idx_min_mitbag=1,idx_max_mitbag="max",tidal="yes"):
        """
        Plotting the EOS curves of MIT bag or CFL matter Quark Stars EOSs.
        1. axis_EOS: the axis that will include the 2D curves of the EOSs
        2. idx_min_cfl: minimum enumaration index of CFL EOS models
        3. idx_min_cfl: maximum enumaration index of CFL EOS models
        4. idx_min_mitbag: minimum enumeration index of MIT bag EOS models
        5. idx_max_mitbag: maximum enumaration index of MIT bag EOS models
        6. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]

        The files to be scanned have the general format:
        1. "CFL-{number of model}_sol.csv"
        2. "MITbag-{number of model}_sol.csv"
        
        on par with the automated recording process on .csv files during the operation of the 'tov_solver_QS.py' or 
        the the 'tov_solver_cflQS_par.py' or the 'tov_solver_mitQS_par.py' scripts. Notice though that the algorithm of the 'cflQSdata' class, generates
        the CFL models as in the 'tov_solver_cflQS_par.py' script.
        """
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")

        available_cfl_models = 0 # counter of the available cfl models files that have been found during the scan
        available_mitbag_models = 0 # counter of the available MIT bag models files that have been found during the scan

        # Print the number of total CFL EOSs
        print("-----------------------------------------------------------------------------------------------")

        # Determinimg the tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix="_tdl"
        
        # Scanning for files containing TOV solution data for CFL EOS models and plotting the respective Ec-Pc curves
        if idx_max_cfl=="max":
            idx_max_cfl = len(self.total_cfl_models_info[0])
        for i in range(idx_min_cfl,idx_max_cfl+1):
            filename = f"{self.dir_cflQS}CFL-{i}{tidal_suffix}_sol.csv"
            if os.path.exists(filename):
                EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus="cornflowerblue")
                available_cfl_models = available_cfl_models + EOS_curve_result

        # Scanning for files containing TOV solution data for MITbag EOS models and plotting the respective Ec-Pc curves
        if idx_max_mitbag=="max":
            idx_max_mitbag = len(self.total_mitbag_models_info[0])
        for i in range(idx_min_mitbag,idx_max_mitbag+1):
            filename = f"{self.dir_mitQS}MITbag-{i}{tidal_suffix}_sol.csv"
            if os.path.exists(filename):
                EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS=axis_EOS,clr_caus="indianred")
                available_mitbag_models = available_mitbag_models + EOS_curve_result        

        

        # Print the number of available CFL EOSs
        print(f"Available CFL quark matter EOSs: {available_cfl_models}")
        print(f"Available MIT bag quark matter EOSs: {available_mitbag_models}")
        print("-----------------------------------------------------------------------------------------------")
        
        # Adding labels for clarity
        axis_EOS.set_xlabel(r"$P$ $[MeV\cdot fm^{-3}]$",fontsize=16)
        axis_EOS.set_ylabel(r"$\epsilon$ $[MeV\cdot fm^{-3}]$",fontsize=16)

        return []
    

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of a MIT bag or CFL QS EOS
    def sample_MR(self,filename,M_threshold=0,points_MR=16,violate_caus="no",noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a MIT bag or CFL Quark Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        3. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        4. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        5. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        6. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        8. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'Μ_treshold' argument
        if type(M_threshold)!=type(2) and type(M_threshold)!=type(2.5):
            raise ValueError("The value of the \"M_threshold\" argument must be a number. Try again.")
        elif M_threshold<0:
            raise ValueError("The value of the \"M_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'points_MR' argument
        if type(points_MR)!=type(2):
            raise ValueError("The value of the \"points_MR\" argument must be an integer. Try again.")
        elif points_MR<=0:
            raise ValueError("The value of the \"points_MR\" argument must be positive. Try again.")
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")

        # Allowed values for the 'noiseM_mv' argument
        if type(noiseM_mv)!=type(2) and type(noiseM_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseM_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseM_std' argument
        if type(noiseM_std)!=type(2) and type(noiseM_std)!=type(2.5):
            raise ValueError("The value of the \"noiseM_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_mv' argument
        if type(noiseR_mv)!=type(2) and type(noiseR_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseR_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_std' argument
        if type(noiseR_std)!=type(2) and type(noiseR_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")            

        mass_segments = points_MR-1 # number of Mass range segments
        obs = points_MR # number of observations in the samples

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"quark")
            
            # Getting data from the scaned file
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                M_data = list(sol_data[3][0]) # getting the NS Mass data
                R_data = list(sol_data[4][0]) # getting the NS Radius data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                M_data = list(sol_data[3][1]) # getting the NS Mass data
                R_data = list(sol_data[4][1]) # getting the NS Radius data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                M_data = list(sol_data[3][2]) # getting the NS Mass data
                R_data = list(sol_data[4][2]) # getting the NS Radius data 

            # Sampling Mass and Radius values only if there are Mass values greater than M_threshold
            if max(M_data)>=M_threshold:

                # Filtering the M_caus data to contain Mass values over M_threshold of Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_data) if mass_value>=M_threshold]
                M_data_filt = [M_data[j] for j in idx_filt]

                # Getting the respective Radius values from the R_caus data list
                R_data_filt = [R_data[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_bounds = np.linspace(min(M_data_filt),max(M_data_filt),points_MR)
                
                # Sampling Mass and Radius values at each segment
                for M_bound in M_bounds:
                    M_diff_abs = abs(M_data_filt-M_bound*np.ones_like(M_data_filt))
                    idx_min_diff = np.argmin(M_diff_abs)
                    mass = M_data_filt[idx_min_diff]
                    radius = R_data_filt[idx_min_diff]

                    # Appening to the storage lists
                    mass_sample.append(mass)
                    radius_sample.append(radius)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]
    

    # Method that samples Slope (dE_dP) data, Energy density on center data (that do not violate causality) and center pressure at maximum mass from TOV solution data files of a MIT bag or CFL matter QS EOS
    def sample_EOS(self,filename,Pc_points=[10,100,300,600,800,1000,1200,1400],violate_caus="no",noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a MIT bag or CFL matter Quark Star EOS and sampling Slope (dE_dP), Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the CFL matter Quark Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,100,300,600,800,1000,1200,1400] MeV*fm^-3.
        3. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        4. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        5. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        6. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        7. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")
        
        # Allowed values for the 'noiseSl_mv' argument
        if type(noiseSl_mv)!=type(2) and type(noiseSl_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseSl_std' argument
        if type(noiseSl_std)!=type(2) and type(noiseSl_std)!=type(2.5):
            raise ValueError("The value of the \"noiseSl_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_mv' argument
        if type(noiseEc_mv)!=type(2) and type(noiseEc_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseEc_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseEc_std' argument
        if type(noiseEc_std)!=type(2) and type(noiseEc_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")
        
        # Getting the number of pressure points
        n = len(Pc_points)

        # Initializing storage lists for the Slope (dE_dP), Energy density on center and center pressure at maximum mass values samples
        dEdP_sample = []
        enrg_dens_sample = []
        dEdP_sample_with_noise = [np.NaN]
        enrg_dens_sample_with_noise = [np.NaN]
        Pc_max_mass = np.NaN
        Ec_max_mass = np.NaN

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"quark")
            
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][0]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][0]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][0]) # getting the NS mass data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][1]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][1]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][1]) # getting the NS mass data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                Ec_data = list(sol_data[1][2]) # getting the NS Energy Density on center data
                dEdP_data = list(sol_data[2][2]) # getting the NS Slope dE_dP data
                M_data = list(sol_data[3][2]) # getting the NS mass data 

            # Sampling Slope (dE_dP) and Energy density on center values only if the EOS overcomes 
            # the value of the maximum pressure point in Pc_points list
            if Pc_data[-1]>=max(Pc_points):
                for i in range(0,n):
                    idx_press_val = Pc_data.index(Pc_points[i])
                    dEdP_sample.append(dEdP_data[idx_press_val])
                    enrg_dens_sample.append(Ec_data[idx_press_val])

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)    
                 
            # Getting the center pressure at maximum mass
            idx_max_mass = np.argmax(M_data)
            Pc_max_mass = Pc_data[idx_max_mass]
            Ec_max_mass = Ec_data[idx_max_mass]

            #print(max(M_caus),Pc_max_mass)

        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass,Ec_max_mass]
    
    # Method that samples the predicted maximum mass and the respective energy density at the center of the neutron star for a single EOS
    def sample_max_M(self,filename,violate_caus="both",tidal="yes"):
        """

        Scanning a file containing the TOV equations' solution data for a polytropic Neutron Star EOS and collecting the predicted maximum mass
        and the respective energy density at the center of the star
        1. filename: name of the file to be scanned
        2. violate_caus: wether the sampled ,ass values do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        3. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]

        """

        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Initializing variables for the maximum mass and the respective energy density at the center of the star
        max_mass = np.NaN
        Ec_dens = np.NaN
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"quark")

            if violate_caus=="no": # case 1: data that do not violate causality
                Ec_data = list(sol_data[1][0]) # getting the NS Energy Density on center data
                M_data = list(sol_data[3][0]) # getting the QS mass data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Ec_data = list(sol_data[1][1]) # getting the QS Energy Density on center data
                M_data = list(sol_data[3][1]) # getting the QS mass data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Ec_data = list(sol_data[1][2]) # getting the QS Energy Density on center data
                M_data = list(sol_data[3][2]) # getting the QS mass data


            # Getting the maximum mass and the  energy density at the center at star's center
            max_mass = max(M_data)
            idx_max_mass = np.argmax(M_data) 
            Ec_dens = Ec_data[idx_max_mass]

        return [max_mass,Ec_dens]

    
    # Method that generates and records on .csv files maximum mass and energy density at star's center data
    def gen_max_mass_data(self,save_filename,violate_caus="both",tidal="yes",idx_min_cfl=1,idx_max_cfl="max",idx_min_mitbag=1,idx_max_mitbag="max"):
        """
        Sampling and recording on .csv files maximum mass and energy density at Quark Star's center data
        1. save_filename: the name of the final .csv file, in which the data are being recorded
        2. violate_caus: wether the sampled mass values do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        3. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        4. idx_min_cfl: minimum enumeration index of CFL EOS models
        5. idx_min_cfl: maximum enumeration index of CFL EOS models
        6. idx_min_mitbag: minimum enumeration index of MIT bag EOS models
        7. idx_max_mitbag: maximum enumeration index of MIT bag EOS models

        """ 


        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")     
    
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")
        
        # Creating the file in which the regression data will be recorder and forming its headers
        headers_eos = f"EoS name," # headers for the name of the EOS
        headers_max_mass = "M_max," # headers for the maximum mass values
        headers_Ec_max_mass = "Ec(M_max)\n" # headers for the center energy density at maximum mass

        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_eos +  headers_max_mass + headers_Ec_max_mass
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)
           

        # Determinimg the tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix="_tdl"
        
        # Quark matter EOS types to be scanned
        eos_models = []
        # Getting the MITbag EOS models
        if idx_max_mitbag=="max":
            idx_max_mitbag = len(self.total_mitbag_models_info[0])
        for i in range(idx_min_mitbag,idx_max_mitbag+1):
            eos_models.append(f"MITbag-{i}")
        # Getting the CFL EOS models
        if idx_max_cfl=="max":
            idx_max_cfl = len(self.total_cfl_models_info[0])
        for i in range(idx_min_cfl,idx_max_cfl+1):
            eos_models.append(f"CFL-{i}")
        

        # Scanning the files for EOS types
        for eos_model in eos_models:
            # Getting the name of the file to be scanned
            if eos_model.count("MITbag"):
                filename = f"{self.dir_mitQS}{eos_model}{tidal_suffix}_sol.csv"
            elif eos_model.count("CFL"):
                filename = f"{self.dir_cflQS}{eos_model}{tidal_suffix}_sol.csv"

            # Getting the maximum mass and energy density at star's center data
            max_mass, Ec_dens = self.sample_max_M(filename,violate_caus=violate_caus,tidal=tidal)

            # Appending the collected data as f-string
            row_eos_model = f"{eos_model},"
            row_max_mass = f"{max_mass},"
            row_Ec_dens = f"{Ec_dens}\n" 

            # Getting the total info of the row and recording it to the final .csv file
            row_info = row_eos_model + row_max_mass + row_Ec_dens 

            with open(f"{save_filename}.csv","a+") as file:
                        file.write(row_info)  

        # Printing completion message
        print(f">The recording process of maximum mass data on the \"{save_filename}.csv\" file has been completed !!!") 
    
    # Method that generates and records on .csv files data of MIT bag or CFL matter Quark Stars for regression purposes
    def gen_reg_data(self,save_filename,samples_per_EOS=1,M_threshold=0,points_MR=16,Pc_points=[10,100,200,300,400,500,600,700,800,900,1000,1100],idx_min_cfl=1,idx_max_cfl="max",idx_min_mitbag=1,idx_max_mitbag="max",violate_caus="no",tidal="yes",noises_mv=[0,0,0,0],noises_std=[0,0,0,0]):
        """
        Getting data of MIT bag or CFL matter Quark Stars for regression purposes and recording them on .csv files
        1. save_filename: the name of the final .csv file, in which the regression data are being recorded
        2. samples_per_EOS: number of samples to be generated per CFL EOS. Each sample is recorded as a row in the final .csv file and includes the
        selected values of Speed of Sound c_s^2 = dP_dE, Energy Density at center, Mass and Radius. By default, 1 sample is generated per CFL EOS.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the EOS must have resulted in causality valid 
        Mass values greater than M_threshold
        4. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        5. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        6. idx_min_cfl: minimum enumeration index of CFL EOS models
        7. idx_min_cfl: maximum enumeration index of CFL EOS models
        8. idx_min_mitbag: minimum enumeration index of MIT bag EOS models
        9. idx_max_mitbag: maximum enumeration index of MIT bag EOS models
        By default the following points are selected: 'Pc_points' = [10,100,200,300,400,500,600,700,800,900,1000,1100] MeV*fm^-3.
        10. eos_type: type of the quark matter EOS. Allowed inputs: ["mit_bag","cfl","both"]. Default: "both".
        11. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        12. tidal: wether the files contain data of tidal deformability or not. Allowed values: ["no", "yes"]
        13. noises_mv: list containing the mean values for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dE_dP) and 4th element -> Energy Density at center. By default the mean values are set to 0.
        14. noises_std: list containing the standard deviations for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dE_dP) and 4th element -> Energy Density at center. By default the standard deviations are set to 0.
        """ 
        
        
        # Allowed values for the 'tidal' argument
        tidal_allowedvalues = ["no","yes"]
        if tidal not in tidal_allowedvalues:
            raise ValueError(f"Invalid value \"{tidal}\" for the \"tidal\" argument. Allowed values are: {tidal_allowedvalues}")

        # Allowed lentgh for the noises_mv list
        if len(noises_mv)!=4:
            raise ValueError(f"The length of the \"noises_mv\" list must be 4, but a list with length {len(noises_mv)} has been given.")

        # Allowed lentgh for the noises_std list
        if len(noises_std)!=4:
            raise ValueError(f"The length of the \"noises_std\" list must be 4, but a list with length {len(noises_std)} has been given.")
        

        # Getting the mean values and the standard deviations of the observational noises
        obs_noiseM_mv = noises_mv[0] # mean value for the noise of the Mass values
        obs_noiseR_mv = noises_mv[1] # mean value for the noise of the Radius values
        obs_noiseSl_mv = noises_mv[2] # mean value for the noise of the Slope (dE_dP) values
        obs_noiseEc_mv = noises_mv[3] # mean value for the noise of the Energy Density at center values

        obs_noiseM_std = noises_std[0] # standard deviation for the noise of the Mass values
        obs_noiseR_std = noises_std[1] # standard deviation for the noise of the Radius values
        obs_noiseSl_std = noises_std[2] # standard deviation for the noise of the Slope (dE_dP) values 
        obs_noiseEc_std = noises_std[3] # standard deviation for the noise of the Energy Density at center values

        # Getting the number of M-R points
        num_mr_points = points_MR

        # Getting the number of pressure points
        num_pc_points = len(Pc_points)

        # Creating the file in which the regression data will be recorder and forming its headers
        headers_eos = f"EoS name," # headers for the name of the EOS
        headers_slope = f"" # headers for the Slope (dP_dE) values
        headers_enrg = f"" # headers for the Energy Density at center values
        headers_Pc_max_mass = "Pc(M_max)," # headers for the center pressure at maximum mass
        headers_Ec_max_mass = "Ec(M_max)," # headers for the center energy density at maximum mass
        headers_Beff = f"Beff," # headers for the Beff parameter
        headers_Delta = f"Delta," # headers for the Delta parameter
        headers_mass = f"" # headers for the Mass values
        headers_radius = f"" # headers for the Radius values

        # Forming the headers for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
        for i in range(0,num_pc_points):
            headers_slope = headers_slope + f"dP_dE({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis
            headers_enrg = headers_enrg + f"E_c({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis

        # Forming the headers for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
        for i in range(0,num_mr_points):
            headers_mass = headers_mass + f"M_{i+1},"
            if i==points_MR-1:
                headers_radius = headers_radius + f"R_{i+1}\n" # the last column of the headers needs \n and not a comma in the end
            else:
                headers_radius = headers_radius + f"R_{i+1},"   
        
        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_eos + headers_slope + headers_enrg + headers_Pc_max_mass + headers_Ec_max_mass + headers_Beff + headers_Delta + headers_mass + headers_radius
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)

        # Creating a copy .csv file where the values in the columns of the X data are shuffled rowwise to avoid correlation between them 
        with open(f"{save_filename}_rwshuffled.csv","w") as file:
            file.write(headers_info)

        # Quark matter EOS types to be scanned
        eos_models = []
        # Getting the MITbag EOS models
        if idx_max_mitbag=="max":
            idx_max_mitbag = len(self.total_mitbag_models_info[0])
        for i in range(idx_min_mitbag,idx_max_mitbag+1):
            eos_models.append(f"MITbag-{i}")
        # Getting the CFL EOS models
        if idx_max_cfl=="max":
            idx_max_cfl = len(self.total_cfl_models_info[0])
        for i in range(idx_min_cfl,idx_max_cfl+1):
            eos_models.append(f"CFL-{i}")

        # Determinimg the tidal suffix
        tidal_suffix = ""
        if tidal=="yes":
            tidal_suffix="_tdl"    
        
        # print(eos_models)
        random.seed(23) # fixing the seed for random shuffling
        random.shuffle(eos_models)
        # print(eos_models)

        # Scanning for all selected EOS models
        for eos_model in eos_models:                       
            # Getting the name of the file to be scanned
            if eos_model.count("MITbag"):
                filename = f"{self.dir_mitQS}{eos_model}{tidal_suffix}_sol.csv"
            elif eos_model.count("CFL"):
                filename = f"{self.dir_cflQS}{eos_model}{tidal_suffix}_sol.csv"

            # Initializing the values of Beff and Delta parameters
            Beff = np.NaN
            Delta = np.NaN

            # Getting the basic sample of the Slope (dE_dP) and Energy Density at center values
            dEdP_basic_sample,enrg_basic_sample,Pc_max_mass,Ec_max_mass = self.sample_EOS(filename,Pc_points,violate_caus=violate_caus,noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0)

            # Getting the basic sample of the Mass and Radius values
            mass_basic_sample,radius_basic_sample = self.sample_MR(filename,M_threshold,points_MR,violate_caus=violate_caus,noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0)

            # Getting the values for Beff and Delta parameters if the filename exists in same path
            if os.path.exists(filename):
                if eos_model.count("MITbag")>0:
                    EOS_data = pd.read_csv(filename)
                    data_columns = EOS_data.columns

                    Beff_column = data_columns[-1]
                    Beff = float(Beff_column[3:])

                    Delta = "-"
                elif eos_model.count("CFL")>0:
                    EOS_data = pd.read_csv(filename)
                    data_columns = EOS_data.columns

                    Beff_column = data_columns[-2]
                    Beff = float(Beff_column[7:])

                    Delta_column = data_columns[-1]
                    Delta = float(Delta_column[6:])

                    # print(slope_basic_sample)
                    # print(enrg_basic_sample)
                    # print(mass_basic_sample)
                    # print(radius_basic_sample)

                # If any of the basic samples is NaN the algorithm skips the recording for this polytropic EOS and moves to the next polytrtopic EOS
                if dEdP_basic_sample[0]==np.NaN or enrg_basic_sample[0]==np.NaN or mass_basic_sample[0]==np.NaN or radius_basic_sample[0]==np.NaN:
                    break
                else:
                    j=1 # intiliazing a counter for the samples to be made per polytropic EOS
                    idx_mr = list(np.arange(0,num_mr_points)) # defining a list with the column indices of the mass/radius values in the basic mass/radius samples
                    while j<=samples_per_EOS:
                        # Adding noise to the values of the main samples and recording the resuted sample as a row in the final .csv file
                        dPdE_sample_with_noise = 1/np.array(dEdP_basic_sample) + np.random.normal(loc=obs_noiseSl_mv,scale=obs_noiseSl_std,size=num_pc_points)
                        enrg_sample_with_noise = enrg_basic_sample + np.random.normal(loc=obs_noiseEc_mv,scale=obs_noiseEc_std,size=num_pc_points)
                        mass_sample_with_noise = np.abs(mass_basic_sample + np.random.normal(loc=obs_noiseM_mv,scale=obs_noiseM_std,size=num_mr_points)) # getting the absolute value to ensure positive values for the Mass
                        radius_sample_with_noise = radius_basic_sample + np.random.normal(loc=obs_noiseR_mv,scale=obs_noiseR_std,size=num_mr_points)

                        # Initializing the row info for the basic .csv file
                        row_eos_info = f"{eos_model}," # row info for the name of EOS model
                        row_slope_info = f"" # row info for the Slope (dE_dP) values
                        row_enrg_info = f"" # row info for the Energy Density at center values
                        row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                        row_Ec_max_mass_info = f"{Ec_max_mass}," # row info for the center energy density at maximum mass
                        row_Beff = f"{Beff}," # row info for the Beff parameter
                        row_Delta = f"{Delta}," # row info for the Delta parameter
                        row_mass_info = f"" # row info for the Mass values
                        row_radius_info = f"" # rwo info for the Radius values

                        # Initializing the row info for the shuffled .csv file
                        shuffled_row_eos_info = f"{eos_model}," # row info for the name of EOS model
                        shuffled_row_slope_info = f"" # row info for the Slope (dE_dP) values
                        shuffled_row_enrg_info = f"" # row info for the Energy Density at center values
                        shuffled_row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                        shuffled_row_Ec_max_mass_info = f"{Ec_max_mass}," # row info for the center energy density at maximum mass
                        shuffled_row_Beff = f"{Beff}," # row info for the Beff parameter
                        shuffled_row_Delta = f"{Delta}," # row info for the Delta parameter 
                        shuffled_row_mass_info = f"" # row info for the Mass values
                        shuffled_row_radius_info = f"" # rwo info for the Radius values

                        # Getting the row info for the Y data (response variables) of the regression, i.e. the Slope (dE_dP) and Energy Density at center values
                        for k in range(0,num_pc_points):
                            row_slope_info = row_slope_info + f"{dPdE_sample_with_noise[k]},"
                            row_enrg_info = row_enrg_info + f"{enrg_sample_with_noise[k]},"
                            shuffled_row_slope_info = shuffled_row_slope_info + f"{dPdE_sample_with_noise[k]},"
                            shuffled_row_enrg_info = shuffled_row_enrg_info + f"{enrg_sample_with_noise[k]},"
                    
                    
                        # Getting the row info for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
                        np.random.shuffle(idx_mr) # shuffling randomly the column indices of the mass/radius values to reduce/avoid linear correlations between the columns 
                        mr_points_count = 1
                        for k in idx_mr:
                            row_mass_info = row_mass_info + f"{mass_sample_with_noise[mr_points_count-1]},"
                            shuffled_row_mass_info = shuffled_row_mass_info + f"{mass_sample_with_noise[k]},"
                            if mr_points_count==num_mr_points:
                                # the last column of the row needs \n and not a comma in the end
                                row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]}\n"
                                shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]}\n" 
                            else:
                                row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]},"
                                shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]},"
                            mr_points_count +=1        
                            
                        # Getting the total info of the row and recording it to the final .csv file
                        row_info = row_eos_info + row_slope_info + row_enrg_info + row_Pc_max_mass_info + row_Ec_max_mass_info + row_Beff + row_Delta + row_mass_info + row_radius_info
                        with open(f"{save_filename}.csv","a+") as file:
                            file.write(row_info)

                        # Getting the total info of the row and recording it to the final shuffled .csv file
                        shuffled_row_info = shuffled_row_eos_info + shuffled_row_slope_info + shuffled_row_enrg_info + shuffled_row_Pc_max_mass_info + shuffled_row_Ec_max_mass_info + shuffled_row_Beff + shuffled_row_Delta + shuffled_row_mass_info + shuffled_row_radius_info
                        with open(f"{save_filename}_rwshuffled.csv","a+") as file:
                            file.write(shuffled_row_info)    

                        j = j + 1 # increasing the counter of the samples per polytropic EOS by 1              
        
        # Getting the final .csv file and removing all the NaN elements
        reg_df = pd.read_csv(f"{save_filename}.csv")
        reg_df_cleaned = reg_df.dropna()
        reg_df_cleaned.to_csv(f"{save_filename}.csv",index=False)

        # Printing completion message
        print(f">The recording process of regression data on the \"{save_filename}.csv\" file has been completed !!!")

        # Getting the final shuffled .csv file and removing all the NaN elements
        reg_df = pd.read_csv(f"{save_filename}_rwshuffled.csv")
        reg_df_cleaned = reg_df.dropna()
        reg_df_cleaned.to_csv(f"{save_filename}_rwshuffled.csv",index=False)

        # Printing completion message
        print(f">>The recording process of rowwise shuffled regression data on the \"{save_filename}_rwshuffled.csv\" file has also been completed !!!\n\n")


# Class for sampling data for classification of Exotic Stars
class classifying_data:
    """
    Class for sampling data for classification of Exotic Stars
    """

    # Initializing the class
    def __init__(self,dir_polyNS,dir_mitQS,dir_cflQS):
        """
        Initializing the "classifying_data" class
        1. Appending the (given) directory for the polytropic EoSs data as self variable "dir_polyNS"
        2. Appending the (given) directory for the MIT-bag EoSs data as self variable "dir_mitQS"
        3. Appending the (given) directory for the CFL EoSs data as self variable "dir_cflQS"
        """

        self.dir_polyNS = dir_polyNS
        self.dir_mitQS = dir_mitQS
        self.dir_cflQS = dir_cflQS

    # Method for sampling the X (expalnatory) data of classification
    def sampling_X_data(self,filename,eos_type="hadronic",M_threshold=0,points_M=10,violate_caus="no",noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0,noisePc_mv=0,noisePc_std=0,noisek2_mv=0,noisek2_std=0,noiseL_mv=0,noiseL_std=0):
        """
        Scanning file containing the TOV equations' solution data for a polytropic Neutron Star or a MIT bag or CFL Quark Star EOS and sampling Mass, Radius, L (love number), k2 and Pressure values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. EOS_type: type of EoS, the file of which is going to be scanned. Allowed values: ["hadronic","quark"]
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        4. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass and the Radius to the boundary points of these segments, and stores these values for the Mass and the Radius samples, respectively.
        By default, 16 points are selected to be sampled.
        5. violate_caus: wether the sampled M-R points do or do not violate causality, or both. Allowed values: ["no","yes","both"]
        6. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        8. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        9. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the "eos_type" argument
        eos_type_allowedvalues = ["hadronic","quark"]
        if eos_type not in eos_type_allowedvalues:
            raise ValueError(f"Invalid input \"{eos_type}\" for the \"eos_type\" argument. Allowed inputs are: {eos_type_allowedvalues}.")
        
        # Allowed values for the 'Μ_treshold' argument
        if type(M_threshold)!=type(2) and type(M_threshold)!=type(2.5):
            raise ValueError("The value of the \"M_threshold\" argument must be a number. Try again.")
        elif M_threshold<0:
            raise ValueError("The value of the \"M_threshold\" argument can not be negative. Try again.")
        
        # Allowed values for the 'points_MR' argument
        if type(points_M)!=type(2):
            raise ValueError("The value of the \"points_MR\" argument must be an integer. Try again.")
        elif points_M<=0:
            raise ValueError("The value of the \"points_MR\" argument must be positive. Try again.")
        
        # Allowed values for the 'linear_behavior' argument
        violate_caus_allowedvalues = ["no","yes","both"]
        if violate_caus not in violate_caus_allowedvalues:
            raise ValueError(f"Invalid value \"{violate_caus}\" for the \"violate_caus\" argument. Allowed values are: {violate_caus_allowedvalues}")

        # Allowed values for the 'noiseM_mv' argument
        if type(noiseM_mv)!=type(2) and type(noiseM_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseM_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseM_std' argument
        if type(noiseM_std)!=type(2) and type(noiseM_std)!=type(2.5):
            raise ValueError("The value of the \"noiseM_std\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_mv' argument
        if type(noiseR_mv)!=type(2) and type(noiseR_mv)!=type(2.5):
            raise ValueError("The value of the \"noiseR_mv\" argument must be a number. Try again.")

        # Allowed values for the 'noiseR_std' argument
        if type(noiseR_std)!=type(2) and type(noiseR_std)!=type(2.5):
            raise ValueError("The value of the \"noiseR_std\" argument must be a number. Try again.")            

        mass_segments = points_M-1 # number of Mass range segments
        obs = points_M # number of observations in the samples

        eos_include = 0 # check index for inclusion of the EoS

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        Pc_sample = []
        k2_sample = []
        Lambda_sample = []

        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        Pc_sample_with_noise = [np.NaN]
        k2_sample_with_noise = [np.NaN]
        Lambda_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,eos_type)
            
            # Getting data from the scaned file
            if violate_caus=="no": # case 1: data that do not violate causality
                Pc_data = list(sol_data[0][0]) # getting the Star's pressure on center data
                M_data = list(sol_data[3][0]) # getting the Star's Mass data
                R_data = list(sol_data[4][0]) # getting the Star's Radius data
                k2_data = list(sol_data[6][0]) # getting the Star's k2 data
                Lambda_data = list(sol_data[7][0]) # getting the Star's Lambda (love number) data
            elif violate_caus=="yes": # case 2: data that do violate causality
                Pc_data = list(sol_data[0][1]) # getting the NS pressure on center data
                M_data = list(sol_data[3][1]) # getting the NS Mass data
                R_data = list(sol_data[4][1]) # getting the NS Radius data
                k2_data = list(sol_data[6][1]) # getting the Star's k2 data
                Lambda_data = list(sol_data[7][1]) # getting the Star's Lambda (love number) data
            elif violate_caus=="both": # case 3: both data that do and do not violate causality
                Pc_data = list(sol_data[0][2]) # getting the NS pressure on center data
                M_data = list(sol_data[3][2]) # getting the NS Mass data
                R_data = list(sol_data[4][2]) # getting the NS Radius data
                k2_data = list(sol_data[6][2]) # getting the Star's k2 data
                Lambda_data = list(sol_data[7][2]) # getting the Star's Lambda (love number) data

            # Sampling Mass and Radius values only if there are Mass values greater than M_threshold
            if max(M_data)>=M_threshold:

                eos_include = 1

                # Filtering the M_caus data to contain Mass values over M_threshold of Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_data) if mass_value>=M_threshold]
                M_data_filt = [M_data[j] for j in idx_filt]
                R_data_filt = [R_data[j] for j in idx_filt] # Getting the respective Radius values
                Pc_data_filt = [Pc_data[j] for j in idx_filt] # Getting the respective central pressure Pc values
                k2_data_filt = [k2_data[j] for j in idx_filt] # Getting the respective k2 values
                Lambda_data_filt = [Lambda_data[j] for j  in idx_filt] # Getting the respective Lambda values

                # Strict Monotonicity Filter (Solves the noise problem)
                M_data_clean = [M_data_filt[0]]
                R_data_clean = [R_data_filt[0]]
                Pc_data_clean = [Pc_data_filt[0]]
                k2_data_clean = [k2_data_filt[0]]
                Lambda_data_clean = [Lambda_data_filt[0]]
                idx_clean = [0]
                for i in range(1, len(M_data_filt)):
                    if M_data_filt[i] > M_data_clean[-1]:
                        M_data_clean.append(M_data_filt[i])
                        R_data_clean.append(R_data_filt[i])
                        Pc_data_clean.append(Pc_data_filt[i])
                        k2_data_clean.append(k2_data_filt[i])
                        Lambda_data_clean.append(Lambda_data_filt[i])
                        idx_clean.append(i)

                
                
                # Getting the Mass bounds of the Mass range segments
                M_bounds = np.linspace(min(M_data_clean),max(M_data_clean),points_M)
                
                # Sampling Mass and Radius values at each segment
                for M_bound in M_bounds:

                    # M_diff_abs = abs(M_data_filt-M_bound*np.ones_like(M_data_filt))
                    # idx_min_abs = np.argmin(M_diff_abs)
                    # mass = M_data_filt[idx_min_abs]
                    # radius = R_data_filt[idx_min_abs]
                    # Pc = Pc_data_filt[idx_min_abs]
                    # k2 = k2_data_filt[idx_min_abs]
                    # Lambda = Lambda_data_filt[idx_min_abs]

                    RM_func = pchip(M_data_clean,R_data_clean)
                    PcM_func = pchip(M_data_clean,Pc_data_clean)
                    k2M_func = pchip(M_data_clean,k2_data_clean)
                    LambdaM_func = pchip(M_data_clean,Lambda_data_clean)

                    mass = M_bound
                    radius = RM_func(mass)
                    Pc = PcM_func(mass)
                    k2 = k2M_func(mass)
                    Lambda = LambdaM_func(mass)

                    # Appening to the storage lists
                    mass_sample.append(mass)
                    radius_sample.append(radius)
                    Pc_sample.append(Pc)
                    k2_sample.append(k2)
                    Lambda_sample.append(Lambda)

                # Adding noise to the Mass, Radius, Pc (central pressure), k2 and Lambda samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)
                Pc_sample_with_noise = Pc_sample + np.random.normal(loc=noisePc_mv,scale=noisePc_std,size=obs)
                k2_sample_with_noise = k2_sample + np.random.normal(loc=noisek2_mv,scale=noisek2_std,size=obs)
                Lambda_sample_with_noise = Lambda_sample + np.random.normal(loc=noiseL_mv,scale=noiseL_std,size=obs)


        return [mass_sample_with_noise,radius_sample_with_noise,Pc_sample_with_noise,k2_sample_with_noise,Lambda_sample_with_noise,eos_include]

    # Method for generating files and recording on them data for classification
    def gen_class_data(self,save_filename,Γ_choices_codec=["A","B","C","D"],Γ_choices=[1,2,3,4],num_segments=4,Beff_max_mitQS=250,Beff_step_mitQS=0.2,Beff_max_cflQS=250,Beff_step_cflQS=5,Delta_max_cflQS=250,Delta_step_cflQS=5,points_MR=10,M_threshold=0.2,noises_mv=[0,0,0,0,0],noises_std=[0,0,0,0,0]):
        """
        Getting the data of polytropic Neutron Stars, MIT-bag and CFL Quark Stars and recordinng them for classification
        1. save_filename: the name of the final .csv file, in which the regression data are being recorded
        2. The list of available coded choices of the Γ parameter
        3. The list of available choices of the Γ parameter
        4. The number of mass density ρ segments
        5. The maximum value of the Beff parameter (for MIT-bag models)
        6. The step value for scanning the Beff-axis area (for MIT-bag models)
        7. The maximum value of the Beff parameter (for CFL models)
        8. The step value for scanning the Beff-axis area (for CFL models)
        9. The maximum value of the Δ parameter (for CFL models only)
        10. The step value for scanning the Δ-axis area (for CFL models only)
        11. points_MR: number of equally spaced points (with respect to Mass-axis) to be selected and sampled. The algorithm divides the range of Mass data that do not violate causality into 
        points_MR-1 segments with equal length. Then it selects the closest values of the Mass, Radius, Pc, k2 and Lambda to the boundary points of these segments, and stores these values, forming the respective samples.
        By default, 10 points are selected to be sampled.
        12. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        13. noises_mv: list containing the mean values for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Central Pressure (Pc), 4th element -> k2 and 5th element -> Lambda (love number). By default the mean values are set to 0.
        14. noises_std: list containing the standard deviations for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Central Pressure (Pc), 4th element -> k2 and 5th element -> Lambda (love number). By default the standard deviations are set to 0.
        """  

        # Getting the mean values and the standard deviations of the observational noises
        obs_noiseM_mv = noises_mv[0] # mean value for the noise of the Mass values
        obs_noiseR_mv = noises_mv[1] # mean value for the noise of the Radius values
        obs_noisePc_mv = noises_mv[2] # mean value for the noise of the Central Pressure (Pc) values
        obs_noisek2_mv = noises_mv[3] # mean value for the noise of the k2 values
        obs_noiseL_mv = noises_mv[4] # mean value for the noise of the Lambda values

        obs_noiseM_std = noises_std[0] # standard deviation for the noise of the Mass values
        obs_noiseR_std = noises_std[1] # standard deviation for the noise of the Radius values
        obs_noisePc_std = noises_std[2] # standard deviation for the noise of the Central Pressure (Pc) values 
        obs_noisek2_std = noises_std[3] # standard deviation for the noise of the k2 values
        obs_noiseL_std = noises_std[4] # standard deviation for the noise of the Lambda values

        # Creating the file in which the regression data will be recorder and forming its headers
    
        headers_mass = "M," # headers for the Mass values
        headers_radius = "R," # headers for the Radius values
        headers_Pc = "Pc," # headers for the central pressure
        headers_k2 = "k2," # headers for the k2 value
        headers_Lambda = "Lambda," # headers for the Lambda value
        headers_eos_name = "EoS name," # headers for the name of the EOS
        headers_eos_idx = "EoS index," # headers for the index of the EOS
        headers_eos_count = "EoS counter," # headers for the counter of the EOS
        headers_category = "Category," # headers for the category of the EOS
        headers_supercategory = "Supercategory\n" # headeres for the type of the Star (named supercategory here)

        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_mass + headers_radius + headers_Pc + headers_k2 + headers_Lambda + headers_eos_name + headers_eos_idx + headers_eos_count + headers_category + headers_supercategory
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)

        #Initializing the counter and index counter of EoS models
        EoS_index = 1
        EoS_counter = 0 

        ################# NEUTRON STARS SCANNING #################

        # The main EOSs for polytropic parameterization
        main_EOSs = ["HLPS-2","HLPS-3"]

        # Getting the combos of Γ parameter
        Γ_combos = polyNSdata(Γ_choices_codec=Γ_choices_codec,Γ_choices=Γ_choices,num_segments=num_segments).Γ_total_combos_sorted

        for main_EOS in main_EOSs:
            for Γ_combo in Γ_combos:
                filename = f"{self.dir_polyNS}{main_EOS}_{Γ_combo}L_tdl_sol.csv"
                mass_sample,radius_sample,Pc_sample,k2_sample,Lambda_sample,eos_include = self.sampling_X_data(filename,eos_type="hadronic",M_threshold=M_threshold,noiseM_mv=obs_noiseM_mv,noiseM_std=obs_noiseM_std,noiseR_mv=obs_noiseR_mv,noiseR_std=obs_noiseR_std,noisePc_mv=obs_noisePc_mv,noisePc_std=obs_noisePc_std,noisek2_mv=obs_noisek2_mv,noisek2_std=obs_noisek2_std,noiseL_mv=obs_noiseL_mv,noiseL_std=obs_noiseL_std)
                # Updating the EOS counter
                EoS_counter = EoS_counter + eos_include
                for i in range(len(mass_sample)):
                    row_mass = f"{mass_sample[i]}," # row info for Mass
                    row_radius = f"{radius_sample[i]}," # row info for Radius
                    row_Pc = f"{Pc_sample[i]}," # row info for Central Pressure
                    row_k2 = f"{k2_sample[i]}," # row info for k2
                    row_Lambda = f"{Lambda_sample[i]}," # row info for Lambda
                    row_eos_name = f"{main_EOS}_{Γ_combo}L," # row info for the name of the EOS
                    row_eos_idx = f"{EoS_index}," #row info for the index of the EOS
                    row_eos_count = f"{EoS_counter}," #row info for the counter of the EOS
                    row_category = f"{main_EOS}," # row info for the category of the EOS
                    row_supercategory = "hadronic\n" # row info for the type of the Star (named supercategory here)

                    # Recording the row info to the .csv file
                    row_info = row_mass + row_radius + row_Pc + row_k2 + row_Lambda + row_eos_name + row_eos_idx + row_eos_count + row_category + row_supercategory
                    with open(f"{save_filename}.csv","a+") as file:
                        file.write(row_info)

                # Updating the EOS index
                EoS_index = EoS_index + 1        

                  

        # Printing completion message
        print(f">>The recording process of hadronic classification data on the \"{save_filename}.csv\" file has been completed !!!")                


        ################# MIT-BAG QUARK STARS SCANNING #################

        # Getting the MIT-bag models based on the given Beff (max and step) values
        mitbag_models_names, mitbag_models_Beff = QSdata(Beff_max=Beff_max_mitQS,Beff_step=Beff_step_mitQS).total_mitbag_models_info

        for j in range(len(mitbag_models_names)):
            filename = f"{self.dir_mitQS}{mitbag_models_names[j]}_tdl_sol.csv"
            mass_sample,radius_sample,Pc_sample,k2_sample,Lambda_sample,eos_include = self.sampling_X_data(filename,eos_type="quark",M_threshold=M_threshold,noiseM_mv=obs_noiseM_mv,noiseM_std=obs_noiseM_std,noiseR_mv=obs_noiseR_mv,noiseR_std=obs_noiseR_std,noisePc_mv=obs_noisePc_mv,noisePc_std=obs_noisePc_std,noisek2_mv=obs_noisek2_mv,noisek2_std=obs_noisek2_std,noiseL_mv=obs_noiseL_mv,noiseL_std=obs_noiseL_std)
            # Updating the EOS counter
            EoS_counter = EoS_counter + eos_include
            for i in range(len(mass_sample)):
                row_mass = f"{mass_sample[i]}," # row info for Mass
                row_radius = f"{radius_sample[i]}," # row info for Radius
                row_Pc = f"{Pc_sample[i]}," # row info for Central Pressure
                row_k2 = f"{k2_sample[i]}," # row info for k2
                row_Lambda = f"{Lambda_sample[i]}," # row info for Lambda
                row_eos_name = f"{mitbag_models_names[j]} ({mitbag_models_Beff[j]})," # row info for the name of the EOS
                row_eos_idx = f"{EoS_index}," #row info for the index of the EOS
                row_eos_count = f"{EoS_counter}," #row info for the counter of the EOS
                row_category = f"MIT-bag," # row info for the category of the EOS
                row_supercategory = "quark\n" # row info for the type of the Star (named supercategory here)

                # Recording the row info to the .csv file
                row_info = row_mass + row_radius + row_Pc + row_k2 + row_Lambda + row_eos_name + row_eos_idx + row_eos_count + row_category + row_supercategory
                with open(f"{save_filename}.csv","a+") as file:
                    file.write(row_info)

            # Updating the EOS index
            EoS_index = EoS_index + 1    

        # Printing completion message
        print(f">>The recording process of MIT-bag quark classification data on the \"{save_filename}.csv\" file has been completed !!!")            

        ################# CFL QUARK STARS SCANNING #################

        # Getting the CFL models based on the given Beff (max and step) and Δ (max and step) values
        cfl_models_names, cfl_models_Beff, cfl_models_Delta,_ = QSdata(Beff_max=Beff_max_cflQS,Beff_step=Beff_step_cflQS,Delta_max=Delta_max_cflQS,Delta_step=Delta_step_cflQS).total_cfl_models_info

        for j in range(len(cfl_models_names)):
            filename = f"{self.dir_cflQS}{cfl_models_names[j]}_tdl_sol.csv"
            mass_sample,radius_sample,Pc_sample,k2_sample,Lambda_sample,eos_include = self.sampling_X_data(filename,eos_type="quark",M_threshold=M_threshold,noiseM_mv=obs_noiseM_mv,noiseM_std=obs_noiseM_std,noiseR_mv=obs_noiseR_mv,noiseR_std=obs_noiseR_std,noisePc_mv=obs_noisePc_mv,noisePc_std=obs_noisePc_std,noisek2_mv=obs_noisek2_mv,noisek2_std=obs_noisek2_std,noiseL_mv=obs_noiseL_mv,noiseL_std=obs_noiseL_std)
            # Updating the EOS counter
            EoS_counter = EoS_counter + eos_include
            for i in range(len(mass_sample)):
                row_mass = f"{mass_sample[i]}," # row info for Mass
                row_radius = f"{radius_sample[i]}," # row info for Radius
                row_Pc = f"{Pc_sample[i]}," # row info for Central Pressure
                row_k2 = f"{k2_sample[i]}," # row info for k2
                row_Lambda = f"{Lambda_sample[i]}," # row info for Lambda
                row_eos_name = f"{cfl_models_names[j]} ({cfl_models_Beff[j]}_{cfl_models_Delta[j]})," # row info for the name of the EOS
                row_eos_idx = f"{EoS_index}," #row info for the index of the EOS
                row_eos_count = f"{EoS_counter}," #row info for the counter of the EOS
                row_category = f"CFL," # row info for the category of the EOS
                row_supercategory = "quark\n" # row info for the type of the Star (named supercategory here)

                # Recording the row info to the .csv file
                row_info = row_mass + row_radius + row_Pc + row_k2 + row_Lambda + row_eos_name + row_eos_idx + row_eos_count + row_category + row_supercategory
                with open(f"{save_filename}.csv","a+") as file:
                    file.write(row_info)

            # Updating the EOS index
            EoS_index = EoS_index + 1    

        # Printing completion message
        print(f">>The recording process of CFL quark classification data on the \"{save_filename}.csv\" file has been completed !!!") 


        # Getting the final .csv file and removing all the NaN elements
        class_df = pd.read_csv(f"{save_filename}.csv")
        class_df_cleaned = class_df.dropna()
        class_df_cleaned.to_csv(f"{save_filename}.csv",index=False)

        print(f">>The \"{save_filename}.csv\" file has been cleaned from NaN elements !!!")
        # Printing completion message
        print(f">>The total recording process of classification data on the \"{save_filename}.csv\" file has been completed !!!")  

        class_df_unq = class_df_cleaned.drop_duplicates(subset=["M","R","Pc","k2","Lambda"], keep="first")
        class_df_unq.to_csv(f"{save_filename}_unq.csv",index=False)

        

        # Printing completion message of unique points files
        print(f">>An additional \"{save_filename}_unq.csv\" file keeping only the unique rows (M,R,Pc,k2,Lambda) has also been created !!!")                       
