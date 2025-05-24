# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py8b
# Name: ExoticStarsDataHandling2.py

# Description: 
# Module offering classes and functions for plotting MR curves, EOS curves and sampling data
# from the solution data of the TOV equations for Exotic Stars (Neutron Stars, Quark Stars)

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

# Defining useful constants
ρ_sat = 2.7*10**14 # nuclear saturation density [g/cm^3]
c = 3e8 # vaccuum speed of light in m/s
MeV_to_J = 1.60218e-13 # 1MeV value in Joules
m_s = 95 # MeV, mass of Strange quark
m_n = 939.565 # MeV, mass of Neutron
hbarc = 197.327 # MeV*fm, hbar*c constant

# Function that reads .csv files containing solution data of the TOV equations 
def file_read(filename,EOS_type="main"):
    """
    Reading a .csv file containing the solution data of TOV equations for a main EOS or a polytropic Neutron Star EOS or a cfl matter Quark Star EOS.
    Obtaining and seperating the data in terms of violation of the causality (c_s/c<=1). Returning the data as a nested list.
    1. filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
    module script. To scan on another folder, the exact path of the file must be provided.
    2. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
    Allowed values: ["main","polytropic","cfl"]
    """

    # Allowed values for the 'EOS_type' argument
    EOS_type_allowedvalues = ["main","polytropic","cfl"]
    if EOS_type not in EOS_type_allowedvalues:
        raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type_allowedvalues}")
        
    if EOS_type=="main" or EOS_type=="cfl":
        # The indices for the columns are on par 
        # with the auto-recording of data on .csv files
        # during the operation of 'the tov_solver_NS.py' or 
        # the 'tov_solver_NS_par.py' or the 'tov_solver_QS.py'
        # or the 'tov_solver_cflQS_par.py' scripts
        polyNS_data = pd.read_csv(filename)
        Pc_data_raw = polyNS_data.iloc[:,1]
        Ec_data_raw = polyNS_data.iloc[:,2]
        dE_dP_data_raw = polyNS_data.iloc[:,3]
        M_data_raw = polyNS_data.iloc[:,4]
        R_data_raw = polyNS_data.iloc[:,5]
                
        # position of slope dE_dP values that do not violate causality
        idx_caus = [i for i, value in enumerate(dE_dP_data_raw) if value>=1]
        # position of slope dE_dP values that do violate causality 
        idx_no_caus = [i for i, value in enumerate(dE_dP_data_raw) if value<1]
                
        # Obtaining the data that do not violate causality
        Pc_caus = [Pc_data_raw[i] for i in idx_caus]
        Ec_caus = [Ec_data_raw[i] for i in idx_caus]
        dE_dP_caus = [dE_dP_data_raw[i] for i in idx_caus]
        M_caus = [M_data_raw[i] for i in idx_caus]
        R_caus = [R_data_raw[i] for i in idx_caus]
                
        # Obtaining the data that do violate causality
        Pc_no_caus = [Pc_data_raw[i] for i in idx_no_caus]
        Ec_no_caus = [Ec_data_raw[i] for i in idx_no_caus]
        dE_dP_no_caus = [dE_dP_data_raw[i] for i in idx_no_caus]
        M_no_caus = [M_data_raw[i] for i in idx_no_caus]
        R_no_caus = [R_data_raw[i] for i in idx_no_caus]

    elif EOS_type=="polytropic":
        # The indices for the columns are on par 
        # with the auto-recording of data on .csv files
        # during the operation of 'the tov_solver_polyNS_par.py' or 
        # the 'the tov_solver_polyNS_par2.py' scripts
        polyNS_data = pd.read_csv(filename)
        Pc_data_raw = polyNS_data.iloc[:,1]
        Ec_data_raw = polyNS_data.iloc[:,2]
        dE_dP_data_raw = polyNS_data.iloc[:,4]
        M_data_raw = polyNS_data.iloc[:,5]
        R_data_raw = polyNS_data.iloc[:,6]
                
        # position of slope dE_dP values that do not violate causality
        idx_caus = [i for i, value in enumerate(dE_dP_data_raw) if value>=1]
        # position of slope dE_dP values that do violate causality 
        idx_no_caus = [i for i, value in enumerate(dE_dP_data_raw) if value<1]
                
        # Obtaining the data that do not violate causality
        Pc_caus = [Pc_data_raw[i] for i in idx_caus]
        Ec_caus = [Ec_data_raw[i] for i in idx_caus]
        dE_dP_caus = [dE_dP_data_raw[i] for i in idx_caus]
        M_caus = [M_data_raw[i] for i in idx_caus]
        R_caus = [R_data_raw[i] for i in idx_caus]
                
        # Obtaining the data that do violate causality
        Pc_no_caus = [Pc_data_raw[i] for i in idx_no_caus]
        Ec_no_caus = [Ec_data_raw[i] for i in idx_no_caus]
        dE_dP_no_caus = [dE_dP_data_raw[i] for i in idx_no_caus]
        M_no_caus = [M_data_raw[i] for i in idx_no_caus]
        R_no_caus = [R_data_raw[i] for i in idx_no_caus]
        
    # Nested list containg the seperated data of the scanned file
    file_data = [[Pc_caus,Pc_no_caus,Pc_data_raw],[Ec_caus,Ec_no_caus,Ec_data_raw],[dE_dP_caus,dE_dP_no_caus,dE_dP_data_raw],[M_caus,M_no_caus,M_data_raw],[R_caus,R_no_caus,R_data_raw]]

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
    def plot_MR_curve(self,main_EOS_name,axis_MR,clr_caus,EOS_type="main",projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a main Neutron Star's EOS
        1. main_EOS_name: the name of the main Neutron Star EOS 
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D or 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main"]
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.   
        """
        
        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")

        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")
        

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")    

        # Creating the name of the file to be scanned
        filename=f"{main_EOS_name}_sol.csv"  
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type)
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        # Setting the line width of the curve
        line_width = 1.2    
        
        # Plotting the EOS in 2d or 3d space
        if projection=="2d": # 2D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],M_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],M_data[1],lw=line_width,color="darkgrey")
        elif projection=="3d": # 3D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_data[0],M_data[0],lw=line_width,color=clr_caus)
            # Plotting the projection on the M-R plane of the M-R data that do not violate causality
            axis_MR.plot(R_data[0],Pc_proj*np.ones_like(Pc_data[0]),M_data[0],"--",lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
            # Plotting the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_data[1],M_data[1],lw=line_width,color="darkgrey")
            # Plotting the projection on the M-R plane of the M-R data that do violate causality
            axis_MR.plot(R_data[1],Pc_proj*np.ones_like(Pc_data[1]),M_data[1],"--",lw=line_width,color="darkgrey")

    # Method that plots the Ec-Pc 2D curve of a main NS EOS
    def plot_EOS_curve(self,main_EOS_name,axis_EOS,clr_caus,EOS_type="main"):
        """
        Reading the EOS data from a given file and plot the respective Ec-Pc 2D curve of a main Neutron Star's EOS
        1. main_EOS_name: the name of the main Neutron Star EOS 
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main"]
        """

        # Allowed values for the 'main_EOS_name' argument
        if main_EOS_name not in self.main_EOSs_names:
            raise ValueError(f"Invalid value \"{main_EOS_name}\" for the \"main_EOS_name\" argument. Allowed values are: {self.main_EOSs_names}")
        
        # Allowed values for the 'projection' argument
        EOS_type_allowedvalues = ["main"]
        if EOS_type not in EOS_type_allowedvalues:
            raise ValueError(f"Invalid value \"{EOS_type}\" for the \"EOS_type\" argument. Allowed values are: {EOS_type}")
    
        
        # Creating the name of the file to be scanned
        filename=f"{main_EOS_name}_sol.csv"
      
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type)
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data

        # Setting the line width of the curve depending on the type of EOS
        line_width = 1.2
        
        # Plotting the EOS 
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_EOS.plot(Pc_data[0],Ec_data[0],lw=line_width,color=clr_caus,label=f"{main_EOS_name}")
        # Plotting the Ec-Pc data that do violate causality
        axis_EOS.plot(Pc_data[1],Ec_data[1],lw=line_width,color="darkgrey")


    # Method the plots the M-R curves of selected main Neutron Star EOSs
    def plot_select_MR(self,main_EOS_list,colors_list,axis_MR,projection="2d",Pc_proj=0):
        """
        Plotting the M-R curves of selected main Neutron Star EOSs
        1. main_EOS_list: list with the names of the main Neutron Star EOS, the respective M-R curves of which are going to be plotted
        2. colors_list: list with the colors for the M-R curves
        3. axis_MR: the axis that will include the M-R 2D or 3D curves
        4. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        5. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'. 
        """ 
        
        # Iterative process scanning for plotting the respective M-R curves of the given main EOSs
        for i in range(0,len(main_EOS_list)):
            self.plot_MR_curve(main_EOS_list[i],axis_MR,colors_list[i],EOS_type="main",projection=projection,Pc_proj=Pc_proj)

    # Method the plots the Ec-Pc curves of selected main Neutron Star EOSs
    def plot_select_EOS(self,main_EOS_list,colors_list,axis_EOS):
        """
        Plotting the Ec-Pc curves of selected main Neutron Star EOSs
        1. main_EOS_list: list with the names of the main Neutron Star EOS, the respective Ec-Pc curves of which are going to be plotted
        2. colors_list: list with the colors for the M-R curves
        3. axis_EOS: the axis that will include the M-R 2D or 3D curves
        """ 
        
        # Iterative process for plotting the respective Ec-Pc curves of the given main EOSs 
        for i in range(0,len(main_EOS_list)):
            self.plot_EOS_curve(main_EOS_list[i],axis_EOS,colors_list[i],EOS_type="main")

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of a main NS EOS
    def sample_MR(self,filename,Pc_threshold=0,M_threshold=0,points_dist=[3,3,3,3,3],noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a main Neutron Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the sampling of Mass and Radius values. By default its value is set to 0.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        4. points_dist: list of random points to be selected. The algorithm divides the range of Mass data that do not violate causality into 
        as many segments as the length of the list 'points_dist'. Then it selects randomly (and uniformly), as many points per segment,
        as the respective value of the list's element, that corresponds to that segment. By default, the list [3,3,3,3,3] is given
        as input for the 'points_dist' argument, i.e. the Mass range is divided into 5 segments and 3 points are randomly selected per segment,
        to create the sample of Mass and Radius values from the scanned file.
        5. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        6. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        8. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
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

        mass_segments = len(points_dist) # number of Mass range segments
        obs = np.sum(points_dist) # number of obsarvations in the sample

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"polytropic")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            M_data = sol_data[3] # getting the NS Mass data
            R_data = sol_data[4] # getting the NS Radius data
            
            # Obtaining the pressure, mass and radius values that do not violate causality
            Pc_caus = Pc_data[0] # getting the NS pressure on center data that do not violate causality
            M_caus = M_data[0]
            R_caus = R_data[0]

            # Sampling Mass and Radius values only if the causality part of the EOS overcomes the threshold pressure
            # and there are Mass values more than M_threshold
            if Pc_caus[-1]>=Pc_threshold and max(M_caus)>=M_threshold:

                # Filtering the M_caus data to contain Mass values over M_threshold Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_caus) if mass_value>=M_threshold]
                M_caus_filt = [M_caus[j] for j in idx_filt]

                # Getting the respective Radius values from the R_caus data list
                R_caus_filt = [R_caus[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_range = np.linspace(min(M_caus_filt),max(M_caus_filt),mass_segments+1)
                
                # Sampling Mass and Radius values at each segment
                for i in range(0,mass_segments):
                    # Index position of Mass values in M_caus list that lie inside the interval [M_range[i],M_range[i+1]]
                    idx_seg = [j for j, mass_value in enumerate(M_caus_filt) if (mass_value>=M_range[i])*(mass_value<=M_range[i+1])]

                    # Mass values in M_caus_filt list that lie inside the interval [M_range[i],M_range[i+1]]
                    M_seg_data = [M_caus_filt[j] for j in idx_seg]

                    # Checking if the M_seg_data list contains less elements than the random choices to be made
                    # This is crucial since we need any element of the list to be (randomly) selected only once
                    if len(M_seg_data)<points_dist[i]:
                        raise ValueError(f"In the Mass segment {i+1} there are {len(M_seg_data)} available Mass values and {points_dist[i]} different random choices requested to be made. Try again.")

                    # Radius values in R_caus_filt list that correspond to the Mass values of M_seg_data list
                    R_seg_data = [R_caus_filt[j] for j in idx_seg]
                    
                    # Sample of Mass values from the current Mass range segment
                    mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    while check_same_value(mass_seg_sample):
                        mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    #print(mass_seg_sample)
                    mass_seg_sample = np.sort(mass_seg_sample) # sorting the mass values in the segment's sample in ascending order
                    
                    # Getting the sample of the respective Radius values
                    radius_seg_sample = []
                    for mass in mass_seg_sample:
                        idx_mass = M_seg_data.index(mass)
                        radius = R_seg_data[idx_mass]
                        radius_seg_sample.append(radius)
                    #print(radius_seg_sample)

                    # Appening to the storage lists
                    mass_sample.append(mass_seg_sample)
                    radius_sample.append(radius_seg_sample)
        
                # Combining the samples of each Mass segment into total samples for Mass and Radius
                mass_sample = np.concatenate((mass_sample),axis=None)
                radius_sample = np.concatenate((radius_sample),axis=None)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]
    
    # Method that samples Slope (dE_dP), Energy density on center data (that do not violate causality) and center pressure at maximum mass from TOV solution data files of a main NS EOS
    def sample_EOS(self,filename,Pc_points=[10,25,50,75,100,200,300,400,500,600,700,800],noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a main Neutron Star EOS and sampling Slope (dE_dP) and Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,300,400,500,600,700,800] MeV*fm^-3.
        3. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        4. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        5. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        6. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
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

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"polytropic")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            Ec_data = sol_data[1] # getting the NS Energy Density on center data
            dEdP_data = sol_data[2] # getting the NS Slope data
            M_data = sol_data[3] # getting the NS mass data

            # Getting the data that do not violate causality
            Pc_caus = Pc_data[0]
            Ec_caus = Ec_data[0]
            dEdP_caus = dEdP_data[0]
            M_caus = M_data[0]

            # Sampling Slope (dE_dP) and Energy density on center values only if the causality part of the EOS overcomes 
            # the value of the maximum pressure point plus 50
            if Pc_caus[-1]>=max(Pc_points)+50:
                for i in range(0,n):
                    idx_press_val = Pc_caus.index(Pc_points[i])
                    dEdP_sample.append(dEdP_caus[idx_press_val])
                    enrg_dens_sample.append(Ec_caus[idx_press_val])
                 
                # Getting the center pressure at maximum mass
                idx_max_mass = np.argmax(M_caus)
                Pc_max_mass = Pc_caus[idx_max_mass]

                #print(max(M_caus),Pc_max_mass)

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)

        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass]

# Defining a class for plotting and sampling data from polytropic NS EOSs
class polyNSdata:
    """ 
    Handling data from the solution of TOV equations for polytropic Neutron Stars' EOSs:
    1. Plotting M-R curves (2D and 3D) and EOSs curves
    2. Sampling data for regression purposes
    """

    # Constructor of the class
    def __init__(self,Γ_choices_codec=["A","B","C","D"],Γ_choices=[1,2,3,4],num_segments=4):
        """
        Initializing the `polyNSdata` class:
        1. Appending the (given) list of available coded choices of the Γ parameter to the self variable 'Γ_choices_codec'
        2. Appending the (given) list of available choices of the Γ parameter to the self variable 'Γ_choices'
        3. Appending the (given) number of mass density ρ segments to the self variable 'num_segments'
        4. Getting and appending the total Γ combinations using the 'Γ_total_combos' method of the 'polyNSdata' class
        """
        self.Γ_choices_codec = Γ_choices_codec
        self.Γ_choices = Γ_choices
        self.num_segments = num_segments
        self.Γ_total_combos_sorted = self.Γ_total_combos()



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
    
    # Method that plots a M-R 2D or 3D curve of a main or a polytropic NS EOS
    def plot_MR_curve(self,filename,axis_MR,clr_caus,EOS_type="main",Pc_threshold=0,projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D or 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots.
        6. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        7. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
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

        # Allowed values for the 'projection' argument
        projection_allowedvalues = ["2d","3d"]
        if projection not in projection_allowedvalues:
            raise ValueError(f"Invalid value \"{projection}\" for the \"projection\" argument. Allowed values are: {projection_allowedvalues}")
        
        # Allowed values for the 'Pc_proj' argument
        if type(Pc_proj)!=type(2) and type(Pc_proj)!=type(2.5):
            raise ValueError("The value of the \"Pc_proj\" argument must be a number. Try again.")    

        EOS_overcome = 0 # index that checks if the EOS overcomes the threshold pressure  
        
        # Scanning and reading the file
        sol_data = file_read(filename,EOS_type)
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8    
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1

            if projection=="2d": # 2D-plotting
                # Plotting the M-R data that do not violate causality
                axis_MR.plot(R_data[0],M_data[0],lw=line_width,color=clr_caus)
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

    # Method that plots an EOS 2D curve of a main or a polytropic NS EOS
    def plot_EOS_curve(self,filename,axis_EOS,clr_caus,EOS_type="main",Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective EOS 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
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
        sol_data = file_read(filename,EOS_type)
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data

        # Setting the line width of the curve depending on the type of EOS
        if EOS_type == "main":
            line_width = 2.5
        else:
            line_width = 0.8
        
        # Plotting only if EOS overcomes the threshold pressure
        if Pc_data[2].iloc[-1]>=Pc_threshold:
            EOS_overcome = 1
            
            # Plotting the Ec-Pc data that do not violate causality
            axis_EOS.plot(Pc_data[0],Ec_data[0],lw=line_width,color=clr_caus)
            # Plotting the Ec-Pc data that do violate causality
            axis_EOS.plot(Pc_data[1],Ec_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome

    # Method that plots the Slope (dE_dP) vs Pressure 2D curve of a main or a polytropic NS EOS
    def plot_dEdP_curve(self,filename,axis_slope,clr_caus,EOS_type="main",Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Slope (dE_dP) vs Pressure 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_slope: the axis that will include the Slope (dE_dP) vs Pressure 2D curve
        3. clr_caus: the color of the points of the Slope (dE_dP) vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
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
        sol_data = file_read(filename,EOS_type)
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
            axis_slope.plot(Pc_data[0],dEdP_data[0],lw=line_width,color=clr_caus)
            # Plotting the c_s-Pc data that do violate causality
            axis_slope.plot(Pc_data[1],dEdP_data[1],lw=line_width,color="darkgrey")
            

        return EOS_overcome                 

    # Method that plots the Speed of sound vs Pressure 2D curve of a main or a polytropic NS EOS
    def plot_cs_curve(self,filename,axis_cs,clr_caus,EOS_type="main",Pc_threshold=0):
        """
        Reading the EOS data from a given file and plot the respective Speed of sound vs Pressure 2D curve of a main or a polytropic Neutron Star's EOS, when the EOS overcomes the given threshold pressure
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_cs: the axis that will include the Speed of sound vs Pressure 2D curve
        3. clr_caus: the color of the points of the Speed of sound vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. EOS_type: the user can select wether the scanned file contains the TOV solution data for a main EOS or a polytropic Neutron Star EOS.
        Allowed values: ["main","polytropic"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
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
        sol_data = file_read(filename,EOS_type)
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
            axis_cs.plot(Pc_data[0],cs_caus,lw=line_width,color=clr_caus)
            # Plotting the c_s-Pc data that do violate causality
            axis_cs.plot(Pc_data[1],cs_no_caus,lw=line_width,color="darkgrey")
            

        return EOS_overcome             


    # Method that plots the M-R 2D or 3D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_MR_single(self,axis_MR,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="no",Pc_threshold=0,projection="2d",Pc_proj=0):
        """ 
        Plotting the M-R curves of polytropic Neutron Stars for a certain main EOS and of the corresponding main EOS.
        1. axis_MR: the axis that will include the M-R 2D or 3D curves
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the M-R curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' is selected the aformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        6. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        7. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.

        The files to be scanned have the general format:
        1. "{mainEOSname}_{Γ_combination}_sol.csv": containing the solution data of TOV equations for EOSs with non-linear behavior at last mass density segment
        2. "{mainEOSname}_{Γ_combination}L_sol.csv": containing the solution data of TOV equations for EOSs with linear behavior at last mass density segment (if selected by the user)
        3. "{mainEOSname}_sol.csv": containing the solution data of TOV equations for the main EOS 
        
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
        
        
        # Defining the colors of the plots
        if mainEOSname=="HLPS-2":
            clr_main = "darkgreen" # color for the M-R curve of the main HLPS-2 EOS
            clr_poly = "rosybrown" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "darkred" # color for the M-R curve of the polytropic EOS with linear behavior
        elif mainEOSname=="HLPS-3":
            clr_main = "gold" # color for the M-R curve of the main HLPS-3 EOS
            clr_poly = "purple" # color for the M-R curve of the polytropic EOS with non-linear behavior
            clr_poly2 = "cornflowerblue" # color for the M-R curve of the polytropic EOS with linear behavior    
        
        EOSs_over_threshold = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (non-linear behavior)
        EOSs_over_threshold_lin = 0 # counter for the polytropic EOSs that overcome the Pc_threshold (linear behavior)

        # Scanning and plotting the TOV solution's M-R data for polytropic EOSs
        for Γ_combo in self.Γ_total_combos_sorted:
            # Checking if the "all" option is selected for the "first_segment value" argument
            if first_segment_value=="all" or Γ_combo[0]==first_segment_value:
                # Scanning for data with EOS non-linear behavior at the last mass density segment
                filename = f"{mainEOSname}_{Γ_combo}_sol.csv"
                # Checking wether the file exists or not, and plottting the data if it exists
                if os.path.exists(filename):
                    MR_curve_result = self.plot_MR_curve(filename,axis_MR,clr_poly,"polytropic",Pc_threshold,projection,Pc_proj)
                    EOSs_over_threshold = EOSs_over_threshold + MR_curve_result

                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{mainEOSname}_{Γ_combo}L_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        MR_curve_result = self.plot_MR_curve(filename,axis_MR,clr_poly2,"polytropic",Pc_threshold,projection,Pc_proj)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + MR_curve_result     

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print("-----------------------------------------------------------------------------------------------")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3]:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print("-----------------------------------------------------------------------------------------------")

        # Scanning and plotting TOV solution's M-R data for the main EOSs
        filename = f"{mainEOSname}_sol.csv"
        # Checking wether the file exists or not, and plottting the data if it exists
        if os.path.exists(filename):
            self.plot_MR_curve(filename,axis_MR,clr_main,"main",Pc_threshold,projection,Pc_proj)  

        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=14)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([4,16]) # displayed radius R values within [4,16] km
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=14)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([4,20]) # displayed radius R values within [4,16] km
            axis_MR.view_init(25,-125)    

        return []          


    # Method that plots the M-R 2D or 3D curves of polytropic EOSs for both main EOSs (HLPS-2 and HLPS-3)
    def plot_MR_both(self,axis_MR,first_segment_value="all",linear_behavior="no",Pc_threshold=0,projection="2d",Pc_proj=0):
        """ 
        Plotting the M-R curves of polytropic Neutron Stars for both HLPS-2 and HLPS-3 main EOSs and the corresponding main EOSs.
        1. axis_MR: the axis that will include the M-R 2D or 3D curves
        2. first_segment_value: allowing the user to draw only the M-R curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' is selected the aformentioned constraint is not applied.
        3. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        4. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data 
        of the Neutron Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.
        """
        self.plot_MR_single(axis_MR,"HLPS-2",first_segment_value,linear_behavior,Pc_threshold,projection,Pc_proj)
        self.plot_MR_single(axis_MR,"HLPS-3",first_segment_value,linear_behavior,Pc_threshold,projection,Pc_proj)

        return []


    # Method that plots the EOS 2D curves of polytropic EOSs for a single main EOS (HLPS-2 or HLPS-3) - by default HLPS-2
    def plot_EOSs_single(self,axis_EOS,mainEOSname="HLPS-2",first_segment_value="all",linear_behavior="no",Pc_threshold=0):
        """ 
        Plotting the EOS curves of polytropic Neutron Stars for a certain main EOS and the corresponding main EOS curve.
        1. axis_EOS: the axis that will include the 2D curves of the EOSs
        2. mainEOSname: the name of the main EOS, by default 'HLPS-2'.
        3. first_segment_value: allowing the user to draw only the M-R curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        4. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        5. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
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
                # Scanning for data with EOS non-linear behavior at the last mass density segment
                filename = f"{mainEOSname}_{Γ_combo}_sol.csv"
                # Checking wether the file exists or not, and plottting the data if it exists
                if os.path.exists(filename):
                    EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS,clr_poly,"polytropic",Pc_threshold)
                    EOSs_over_threshold = EOSs_over_threshold + EOS_curve_result

                # Scanning for data with EOS linear behavior at the last mass density segment (if selected)
                if linear_behavior=="yes":
                    filename = f"{mainEOSname}_{Γ_combo}L_sol.csv"
                    # Checking wether the file exists or not, and plottting the data if it exists
                    if os.path.exists(filename):
                        EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS,clr_poly2,"polytropic",Pc_threshold)
                        EOSs_over_threshold_lin = EOSs_over_threshold_lin + EOS_curve_result

        # Print the number of EOSs over threshold pressure
        print(f"Γ coded value on first mass density segment -> {first_segment_value}")
        print("-----------------------------------------------------------------------------------------------")
        print(f"Available POLYTROPIC EOSs with {mainEOSname} as MAIN EOS function, with pressure over {Pc_threshold:.3f} [MeV*fm^3]:")
        print(f"Non-linear behavior -> {EOSs_over_threshold}")
        print(f"Linear behavior -> {EOSs_over_threshold_lin}")
        print("-----------------------------------------------------------------------------------------------")

        # Scanning and plotting TOV solution's Ec-Pc data for the main EOS
        filename = f"{mainEOSname}_sol.csv"
        # Checking wether the file exists or not, and plottting the data if it exists
        if os.path.exists(filename):
            self.plot_EOS_curve(filename,axis_EOS,clr_main,"main",Pc_threshold)
            

        # Adding labels for clarity, as well as setting the scale of both axes to logarithmic
        axis_EOS.set_xlabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)
        axis_EOS.set_ylabel(r"$E_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)
        axis_EOS.set_xscale("log")
        axis_EOS.set_yscale("log")

        return []


    # Method that plots the EOS 2D curves of polytropic EOSs for both main EOSs (HLPS-2 and HLPS-3)
    def plot_EOSs_both(self,axis_EOS,first_segment_value="all",linear_behavior="no",Pc_threshold=0):
        """ 
        Plotting the M-R curves of polytropic Neutron Stars for both HLPS-2 and HLPS-3 main EOSs and the corresponding main EOSs.
        1. axis_EOS: the axis that will include the 2D curves of the EOSs
        2. first_segment_value: allowing the user to draw only the M-R curves with a certain value of Γ in the first segment of mass density
        (from the available coded choices). When the option 'all' the afformentioned constraint is not applied.
        3. linear_behavior: allowing the user to scan additionally for files containing TOV equations solution data from EOSs that display 
        linear rather than polytropic behavior at the last segment of mass density (in order to avoid the violation of causality). Allowed values: ["yes","no"]
        4. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the plots. By default its value is set to 0.
        """
        self.plot_EOSs_single(axis_EOS,"HLPS-2",first_segment_value,linear_behavior,Pc_threshold)
        self.plot_EOSs_single(axis_EOS,"HLPS-3",first_segment_value,linear_behavior,Pc_threshold)

        return []
    

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of a polytropic NS EOS
    def sample_MR(self,filename,Pc_threshold=0,M_threshold=0,points_dist=[3,3,3,3,3],noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a polytropic Neutron Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_threshold: Threshold of the maximum pressure in [MeV*fm^-3]. The EOSs with maximum pressure less than the threshold are not included
        in the sampling of Mass and Radius values. By default its value is set to 0.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        4. points_dist: list of random points to be selected. The algorithm divides the range of Mass data that do not violate causality into 
        as many segments as the length of the list 'points_dist'. Then it selects randomly (and uniformly), as many points per segment,
        as the respective value of the list's element, that corresponds to that segment. By default, the list [3,3,3,3,3] is given
        as input for the 'points_dist' argument, i.e. the Mass range is divided into 5 segments and 3 points are randomly selected per segment,
        to create the sample of Mass and Radius values from the scanned file.
        5. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        6. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        7. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        8. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
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

        mass_segments = len(points_dist) # number of Mass range segments
        obs = np.sum(points_dist) # number of obsarvations in the sample

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"polytropic")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            M_data = sol_data[3] # getting the NS Mass data
            R_data = sol_data[4] # getting the NS Radius data
            
            # Obtaining the pressure, mass and radius values that do not violate causality
            Pc_caus = Pc_data[0] # getting the NS pressure on center data that do not violate causality
            M_caus = M_data[0]
            R_caus = R_data[0]

            # Sampling Mass and Radius values only if the causality part of the EOS overcomes the threshold pressure
            # and there are Mass values more than M_threshold
            if Pc_caus[-1]>=Pc_threshold and max(M_caus)>=M_threshold:

                # Filtering the M_caus data to contain Mass values over M_threshold Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_caus) if mass_value>=M_threshold]
                M_caus_filt = [M_caus[j] for j in idx_filt]

                # Getting the respective Radius values from the R_caus data list
                R_caus_filt = [R_caus[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_range = np.linspace(min(M_caus_filt),max(M_caus_filt),mass_segments+1)
                
                # Sampling Mass and Radius values at each segment
                for i in range(0,mass_segments):
                    # Index position of Mass values in M_caus list that lie inside the interval [M_range[i],M_range[i+1]]
                    idx_seg = [j for j, mass_value in enumerate(M_caus_filt) if (mass_value>=M_range[i])*(mass_value<=M_range[i+1])]

                    # Mass values in M_caus_filt list that lie inside the interval [M_range[i],M_range[i+1]]
                    M_seg_data = [M_caus_filt[j] for j in idx_seg]

                    # Checking if the M_seg_data list contains less elements than the random choices to be made
                    # This is crucial since we need any element of the list to be (randomly) selected only once
                    if len(M_seg_data)<points_dist[i]:
                        raise ValueError(f"In the Mass segment {i+1} there are {len(M_seg_data)} available Mass values and {points_dist[i]} different random choices requested to be made. Try again.")

                    # Radius values in R_caus_filt list that correspond to the Mass values of M_seg_data list
                    R_seg_data = [R_caus_filt[j] for j in idx_seg]
                    
                    # Sample of Mass values from the current Mass range segment
                    mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    while check_same_value(mass_seg_sample):
                        mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    #print(mass_seg_sample)
                    mass_seg_sample = np.sort(mass_seg_sample) # sorting the mass values in the segment's sample in ascending order
                    
                    # Getting the sample of the respective Radius values
                    radius_seg_sample = []
                    for mass in mass_seg_sample:
                        idx_mass = M_seg_data.index(mass)
                        radius = R_seg_data[idx_mass]
                        radius_seg_sample.append(radius)
                    #print(radius_seg_sample)

                    # Appening to the storage lists
                    mass_sample.append(mass_seg_sample)
                    radius_sample.append(radius_seg_sample)
        
                # Combining the samples of each Mass segment into total samples for Mass and Radius
                mass_sample = np.concatenate((mass_sample),axis=None)
                radius_sample = np.concatenate((radius_sample),axis=None)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]
    

    # Method that samples Slope (dE_dP), Energy density on center data (that do not violate causality) and center pressure at maximum mass from TOV solution data files of a polytropic NS EOS
    def sample_EOS(self,filename,Pc_points=[10,25,50,75,100,200,400,800],noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a polytropic Neutron Star EOS and sampling Slope (dE_dP) and Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,400,800] MeV*fm^-3.
        3. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        4. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope (dE_dP) sample. By default its value is set to 0.
        5. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        6. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
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

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"polytropic")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            Ec_data = sol_data[1] # getting the NS Energy Density on center data
            dEdP_data = sol_data[2] # getting the NS Slope data
            M_data = sol_data[3] # getting the NS mass data

            # Getting the data that do not violate causality
            Pc_caus = Pc_data[0]
            Ec_caus = Ec_data[0]
            dEdP_caus = dEdP_data[0]
            M_caus = M_data[0]

            # Sampling Slope (dE_dP) and Energy density on center values only if the causality part of the EOS overcomes 
            # the value of the maximum pressure point plus 50
            if Pc_caus[-1]>=max(Pc_points)+50:
                for i in range(0,n):
                    idx_press_val = Pc_caus.index(Pc_points[i])
                    dEdP_sample.append(dEdP_caus[idx_press_val])
                    enrg_dens_sample.append(Ec_caus[idx_press_val])
                 
                # Getting the center pressure at maximum mass
                idx_max_mass = np.argmax(M_caus)
                Pc_max_mass = Pc_caus[idx_max_mass]

                #print(max(M_caus),Pc_max_mass)

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)

        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass]

    # Method that generates and records on .csv files data of polytropic Neutron Stars for regression purposes
    def gen_reg_data(self,save_filename,samples_per_EOS=1,M_threshold=0,points_dist=[3,3,3,3,3],Pc_points=[10,25,50,75,100,200,400,800],noises_mv=[0,0,0,0],noises_std=[0,0,0,0]):
        """
        Getting data of polytropic Neutron Stars for regression purposes and recording them on .csv files
        1. save_filename: the name of the final .csv file, in which the regression data are being recorded
        2. samples_per_EOS: number of samples to be generated per polytropic EOS. Each sample is recorded as a row in the final .csv file and includes the
        selected values of Speed of Sound = dP_dE, Energy Density at center, Mass and Radius. By default, 1 sample is generated per polytropic EOS.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the EOS must have resulted in causality valid 
        Mass values greater than M_threshold
        4. points_dist: list of random points to be selected. The algorithm divides the range of Mass data that do not violate causality into 
        as many segments as the length of the list 'points_dist'. Then it selects randomly (and uniformly), as many points per segment,
        as the respective value of the list's element, that corresponds to that segment. By default, the list [3,3,3,3,3] is given
        as input for the 'points_dist' argument, i.e. the Mass range is divided into 5 segments and 3 points are randomly selected per segment.
        5. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,25,50,75,100,200,400,800] MeV*fm^-3.
        6. noises_mv: list containing the mean values for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the mean values are set to 0.
        7. noises_std: list containing the standard deviations for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the standard deviations are set to 0.
        """ 

        # Allowed lentgh for the noises_mv list
        if len(noises_mv)!=4:
            raise ValueError(f"The length of the \"noises_mv\" list must be 4, but a list with length {len(noises_mv)} has been given.")

        # Allowed lentgh for the noises_std list
        if len(noises_std)!=4:
            raise ValueError(f"The length of the \"noises_std\" list must be 4, but a list with length {len(noises_std)} has been given.")

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
        num_mr_points = np.sum(points_dist)

        # Getting the number of pressure points
        num_pc_points = len(Pc_points)

        # Creating the file in which the regression data will be recorder and forming its headers
        headers_slope = f"" # headers for the Slope (dP_dE) values
        headers_enrg = f"" # headers for the Energy Density at center values
        headers_Pc_max_mass = "Pc(M_max)," # headers for the center pressure at maximum mass
        headers_Γ = f"" # headers for the values of the polytropic parameter Γ in the pressure segments 
        headers_mass = f"" # headers for the Mass values
        headers_radius = f"" # headers for the Radius values
        

        # Forming the headers for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
        for i in range(0,num_pc_points):
            headers_slope = headers_slope + f"dP_dE({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis
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
        headers_info = headers_slope + headers_enrg + headers_Pc_max_mass + headers_Γ + headers_mass + headers_radius
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)

        # Creating a copy .csv file where the values in the columns of the X data are shuffled rowwise to avoid correlation between them 
        with open(f"{save_filename}_rwshuffled.csv","w") as file:
            file.write(headers_info)    

        # Linear behavior signs
        linear_behavior_signs = ["","L"]

        # Main EOSs to be included
        main_EOSs = ["HLPS-2","HLPS-3"]

        # Scanning all the files for all main EOSs
        for main_EOS in main_EOSs:
            # Scanning both for linear and non-linear behavior of the polytropic EOSs at pressure's last segment
            for linear_sign in linear_behavior_signs:
                # Scanning all the combinations of the Γ parameter
                for Γ_combo in self.Γ_total_combos_sorted:
                    # Getting the name of the file to be scanned
                    filename = f"{main_EOS}_{Γ_combo}{linear_sign}_sol.csv"

                    # Getting the basic sample of the Slope (dE_dP) and Energy Density at center values
                    dEdP_basic_sample,enrg_basic_sample,Pc_max_mass = self.sample_EOS(filename,Pc_points,noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0)

                    # Getting the basic sample of the Mass and Radius values
                    mass_basic_sample,radius_basic_sample = self.sample_MR(filename,max(Pc_points)+50,M_threshold,points_dist,noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0)
                    
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
                            row_slope_info = f"" # row info for the Slope (dP_dE) values
                            row_enrg_info = f"" # row info for the Energy Density at center values
                            row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                            row_Γ_info = f"" # row info for the values of the polytropic parameter Γ in the pressure segments 
                            row_mass_info = f"" # row info for the Mass values
                            row_radius_info = f"" # rwo info for the Radius values
                            

                            # Initializing the row info for the shuffled .csv file
                            shuffled_row_slope_info = f"" # row info for the Slope (dP_dE) values
                            shuffled_row_enrg_info = f"" # row info for the Energy Density at center values
                            shuffled_row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass
                            shuffled_row_Γ_info = f"" # row info for the values of the polytropic parameter Γ in the pressure segments 
                            shuffled_row_mass_info = f"" # row info for the Mass values
                            shuffled_row_radius_info = f"" # rwo info for the Radius values
                            

                            # Getting the row info for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
                            for k in range(0,num_pc_points):
                                row_slope_info = row_slope_info + f"{dPdE_sample_with_noise[k]},"
                                row_enrg_info = row_enrg_info + f"{enrg_sample_with_noise[k]},"
                                shuffled_row_slope_info = shuffled_row_slope_info + f"{dPdE_sample_with_noise[k]},"
                                shuffled_row_enrg_info = shuffled_row_enrg_info + f"{enrg_sample_with_noise[k]},"

                            # Getting the row info for the Γ parameter data
                            for k in range(0,self.num_segments):
                                row_Γ_info = row_Γ_info + f"{self.Γ_decode(Γ_combo[k])},"
                                shuffled_row_Γ_info = shuffled_row_Γ_info + f"{self.Γ_decode(Γ_combo[k])},"    
                            
                            # Getting the row info for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
                            np.random.shuffle(idx_mr) # shuffling randomly the column indices of the mass/radius values to reduce/avoid linear correlations between the columns 
                            mr_points_count = 1
                            for k in idx_mr:
                                row_mass_info = row_mass_info + f"{mass_sample_with_noise[mr_points_count-1]},"
                                shuffled_row_mass_info = shuffled_row_mass_info + f"{mass_sample_with_noise[k]},"
                                # the last column of the row needs \n and not a comma in the end
                                if mr_points_count==num_mr_points-1:
                                    row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]}\n"
                                    shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]}\n"
                                else:
                                    row_radius_info = row_radius_info + f"{radius_sample_with_noise[mr_points_count-1]},"
                                    shuffled_row_radius_info = shuffled_row_radius_info + f"{radius_sample_with_noise[k]},"    
                                mr_points_count +=1    
                            
                            # Getting the total info of the row and recording it to the final .csv file
                            row_info = row_slope_info + row_enrg_info + row_Pc_max_mass_info + row_Γ_info + row_mass_info + row_radius_info
                            with open(f"{save_filename}.csv","a+") as file:
                                file.write(row_info)

                            # Getting the total info of the row and recording it to the final shuffled .csv file
                            shuffled_row_info = shuffled_row_slope_info + shuffled_row_enrg_info + shuffled_row_Pc_max_mass_info + shuffled_row_Γ_info + shuffled_row_mass_info + shuffled_row_radius_info
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



# Defining a class for plotting and sampling data from CFL matter QS EOSs
class cflQSdata:
    """ 
    Handling data from the solution of TOV equations for CFL matter Quark Stars' EOSs:
    1. Plotting M-R curves (2D and 3D) and EOSs curves
    2. Sampling data for regression purposes
    """ 

    # Constructor of the class
    def __init__(self,Beff_max=250,Beff_step=5,Delta_max=250,Delta_step=10):
        """
        Initializing the `cflQSdata` class:
        1. Setting the minimum value of the Beff parameter to 60 [MeV*fm^-3] for this class. The actual minmum allowed value is [MeV*fm^-3] # according to the MIT bag model 
        2. Setting the minimum value of the Δ parameter of the MIT bag model to 50 [MeV] for this class. 
        3. Appending the (given) maximum value of the Beff parameter to the self variable 'Beff_max'
        4. Appending the (given) step value for scanning the Beff-axis area to the self variable 'Beff_step'
        5. Appending the (given) maximum value of the Δ parameter to the self variable 'Delta_max'
        6. Appending the (given) step value for scanning the Δ-axis area to the self variable 'Delta_step'
        7. Getting the total valid CFL models and their respective Beff and Δ combinations using the 'valid_cfl_models'
        of the 'cflQSdata' class. Appending the information of the models to the self variable 'total_cfl_models_info'

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
        for B_eff_val in range(self.Beff_min,self.Beff_max+self.Beff_step,self.Beff_step):
            for Delta_val in range(self.Delta_min,self.Delta_max+self.Delta_step,self.Delta_step):
                if B_eff_val <self. B_eff_bound(Delta_val):
                    models_names.append(f"CFL-{i}")
                    valid_ms.append(m_s)
                    valid_Beff.append(B_eff_val)
                    valid_Delta.append(Delta_val)
                    i = i + 1

        return [models_names,valid_Beff,valid_Delta,valid_ms]
    
    # Method that displays the valid generated CFL EOSs models info in a PrettyTable format
    def show_models_info(self):
        models_info = PrettyTable()

        models_info.add_column("Models",self.total_cfl_models_info[0])
        models_info.add_column("B_eff [MeV*fm^-3]",self.total_cfl_models_info[1])
        models_info.add_column("Δ [MeV]",self.total_cfl_models_info[2])
        models_info.add_column("m_s [MeV]",self.total_cfl_models_info[3])

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
        fig_valid_cfl, axis_valid_cfl = plt.subplots(1,1,figsize=(10,10))

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
        axis_valid_cfl.set_xlabel(r"$\Delta$ $[MeV]$",fontsize=14)
        axis_valid_cfl.set_ylabel(r"$B_{eff}$ $[MeV\cdot fm^{-3}]$",fontsize=14)
        axis_valid_cfl.legend()

        return fig_valid_cfl,axis_valid_cfl


    # Method that plots a M-R 2D or 3D curve of a CFL matter QS EOS
    def plot_MR_curve(self,filename,axis_MR,clr_caus,clr_caus_3d,projection="2d",Pc_proj=0):
        """
        Reading the EOS data from a given file and plot the respective M-R 2D or 3D curve of a CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_MR: the axis that will include the M-R 2D or 3D curve
        3. clr_caus: the color of the points of the M-R 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        4. clr_caus_3d: the color of the points of the M-R 3D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        5. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the Neutron
        Star. When the 3d option is selected: including additionally the pressure in center data of the Neutron Star in a 3rd axis.
        6. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
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
        sol_data = file_read(filename,"cfl")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        M_data = sol_data[3] # getting the NS Mass data
        R_data = sol_data[4] # getting the NS Radius data

        if projection=="2d": # 2D-plotting
            # Plotting the M-R data that do not violate causality
            axis_MR.plot(R_data[0],M_data[0],lw=0.8,color=clr_caus)
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

    # Method that plots an EOS 2D curve of CFL matter QS EOS
    def plot_EOS_curve(self,filename,axis_EOS,clr_caus):
        """
        Reading the EOS data from a given file and plot the respective EOS 2D curve of a CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_EOS: the axis that will include the 2D curve of the EOS
        3. clr_caus: the color of the points of the EOS 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """
    
        
        # Scanning and reading the file
        sol_data = file_read(filename,"cfl")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        Ec_data = sol_data[1] # getting the NS energy density on center data
            
        # Plotting the Ec-Pc data that do not violate causality
        axis_EOS.plot(Pc_data[0],Ec_data[0],lw=0.8,color=clr_caus)
        # Plotting the Ec-Pc data that do violate causality
        axis_EOS.plot(Pc_data[1],Ec_data[1],lw=0.8,color="darkgrey")
            

        return 1

    # Method that plots the Slope (dE_dP) vs Pressure 2D curve of a CFL matter QS EOS
    def plot_dEdP_curve(self,filename,axis_slope,clr_caus):
        """
        Reading the EOS data from a given file and plot the respective Slope (dE_dP) vs Pressure 2D curve of a CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_slope: the axis that will include the Slope (dE_dP) vs Pressure 2D curve
        3. clr_caus: the color of the points of the Slope (dE_dP) vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """
    
        
        # Scanning and reading the file
        sol_data = file_read(filename,"cfl")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Plotting the c_s-Pc data that do not violate causality
        axis_slope.plot(Pc_data[0],dEdP_data[0],lw=0.8,color=clr_caus)
        # Plotting the c_s-Pc data that do violate causality
        axis_slope.plot(Pc_data[1],dEdP_data[1],lw=0.8,color="darkgrey")
            

        return 1                 

    # Method that plots the Speed of sound vs Pressure 2D curve of a CFL matter QS EOS
    def plot_cs_curve(self,filename,axis_cs,clr_caus):
        """
        Reading the EOS data from a given file and plot the respective Speed of sound vs Pressure 2D curve of a CFL matter Quark Star's EOS
        1.filename: name of the file to be read. By default the scanning is performed in the folder that contains the 'ExoticStarsDataHandling'
        module script. To scan on another folder, the exact path of the file must be provided.
        2. axis_cs: the axis that will include the Speed of sound vs Pressure 2D curve
        3. clr_caus: the color of the points of the Speed of sound vs Pressure 2D curve that do not violate causality. The rest points (that violate causality) are plotted
        with 'darkgrey' color.
        """
        
        # Scanning and reading the file
        sol_data = file_read(filename,"cfl")
        Pc_data = sol_data[0] # getting the NS pressure on center data
        dEdP_data = sol_data[2] # getting the NS Slope (dE_dP) on center data

        # Calculating the speed of sound (c_s) values from the slope data
        cs_caus = np.sqrt(1/np.array(dEdP_data[0])) # values that do not violate causality
        cs_no_caus = np.sqrt(1/np.array(dEdP_data[1])) # values that do violate causality
            
        # Plotting the c_s-Pc data that do not violate causality
        axis_cs.plot(Pc_data[0],cs_caus,lw=0.8,color=clr_caus)
        # Plotting the c_s-Pc data that do violate causality
        axis_cs.plot(Pc_data[1],cs_no_caus,lw=0.8,color="darkgrey")
            

        return 1            

    # Method that plots the M-R 2D or 3D curves of CFL quark matter EOSs
    def plot_MR(self,axis_MR,projection="2d",Pc_proj=0):
        """
        Plotting the M-R curves of CFL matter Quark Stars EOSs.
        1. axis_MR: the axis that will include the M-R curves
        2. projection: projection of the axis that plots the M-R curves. Values: ["2d","3d"]. By default: 2d projection and plot of the Mass and Radius data of the CFL Quark Star
        Star. When the 3d option is selected: including additionally the pressure in center data of the Quark Star in a 3rd axis.
        3. Pc_proj: the pressure of a plane parallel to the M-R plane, on which the 2d-projections of the 3d M-R curves will be displayed, when the "3d" option is selected
        for the 'projection' argument. By default the value 0 is appended to the argument 'Pc_proj'.

        The files to be scanned have the general format:
        1. "CFL-{number of model}_sol.csv":
        
        on par with the automated recording process on .csv files during the operation of the 'tov_solver_QS.py' or 
        the the 'tov_solver_cflQS_par.py' scripts. Notice though that the algorithm of the 'cflQSdata' class, generates
        the cfl models as in the 'tov_solver_cflQS_par.py' script.
        """
        

        total_cfl_models = len(self.total_cfl_models_info[0]) # total valid generated CFL models
        available_cfl_models = 0 # counter of the available CFL models files that have been found during the scan

        # Print the number of total cfl EOSs
        print("-----------------------------------------------------------------------------------------------")
        print(f"Total CFL quark matter EOSs: {total_cfl_models}")

        for i in range(0,total_cfl_models):
            model_name = self.total_cfl_models_info[0][i]
            filename = f"{model_name}_sol.csv"
            if os.path.exists(filename):
                MR_curve_result = self.plot_MR_curve(filename,axis_MR,"darksalmon","darkorange",projection,Pc_proj)
                available_cfl_models = available_cfl_models + MR_curve_result
        

        # Print the number of available cfl EOSs
        print(f"Available CFL quark matter EOSs: {available_cfl_models}")
        print("-----------------------------------------------------------------------------------------------")

        # Adding labels and setting axes scale for clarity
        if projection=="2d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=14)
            axis_MR.set_ylabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([0,20]) # displayed radius R values within [0,20] km
        elif projection=="3d":
            axis_MR.set_xlabel(r"R $[km]$",fontsize=14)
            axis_MR.set_ylabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)
            axis_MR.set_zlabel(r"$M$ $(M_\odot)$",fontsize=14)
            axis_MR.set_xbound([0,20]) # displayed radius R values within [0,20] km
            axis_MR.view_init(25,-125)    

        return []
    

    # Method that plots the EOS 2D curves of CFL quark matter EOSs
    def plot_EOSs(self,axis_EOS):
        """
        Plotting the EOS curves of CFL matter Quark Stars EOSs.
        1. axis_EOS: the axis that will include the 2D curves of the EOSs

        The files to be scanned have the general format:
        1. "CFL-{number of model}_sol.csv":
        
        on par with the automated recording process on .csv files during the operation of the 'tov_solver_QS.py' or 
        the the 'tov_solver_cflQS_par.py' scripts. Notice though that the algorithm of the 'cflQSdata' class, generates
        the CFL models as in the 'tov_solver_cflQS_par.py' script.
        """
        
        total_cfl_models = len(self.total_cfl_models_info[0]) # total valid generated cfl models
        available_cfl_models = 0 # counter of the available cfl models files that have been found during the scan

        # Print the number of total CFL EOSs
        print("-----------------------------------------------------------------------------------------------")
        print(f"Total CFL quark matter EOSs: {total_cfl_models}")

        for i in range(0,total_cfl_models):
            model_name = self.total_cfl_models_info[0][i]
            filename = f"{model_name}_sol.csv"
            if os.path.exists(filename):
                EOS_curve_result = self.plot_EOS_curve(filename,axis_EOS,"darksalmon")
                available_cfl_models = available_cfl_models + EOS_curve_result

        

        # Print the number of available CFL EOSs
        print(f"Available CFL quark matter EOSs: {available_cfl_models}")
        print("-----------------------------------------------------------------------------------------------")
        
        # Adding labels for clarity
        axis_EOS.set_xlabel(r"$P_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)
        axis_EOS.set_ylabel(r"$E_c$ $[MeV\cdot fm^{-3}]$",fontsize=14)

        return []
    

    # Method that samples Mass and Radius data (that do not violate causality) from TOV solution data files of a CFL QS EOS
    def sample_MR(self,filename,M_threshold=0,points_dist=[3,3,3,3,3],noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0):
        """
        Scanning file containing the TOV equations' solution data for a CFL Quark Star EOS and sampling Mass and Radius values,
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the scanned file must contain causality valid Mass values greater than M_threshold
        3. points_dist: list of random points to be selected. The algorithm divides the range of Mass data that do not violate causality into 
        as many segments as the length of the list 'points_dist'. Then it selects randomly (and uniformly), as many points per segment,
        as the respective value of the list's element, that corresponds to that segment. By default, the list [3,3,3,3,3] is given
        as input for the 'points_dist' argument, i.e. the Mass range is divided into 5 segments and 3 points are randomly selected per segment,
        to create the sample of Mass and Radius values from the scanned file.
        4. noiseM_mv: mean value of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        5. noiseM_std: standard deviation of the normal distributed observational noise for the values of the Mass sample. By default its value is set to 0.
        6. noiseR_mv: mean value of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        7. noiseR_std: standard deviation of the normal distributed observational noise for the values of the Radius sample. By default its value is set to 0.
        """ 
        
        # Allowed values for the 'Μ_treshold' argument
        if type(M_threshold)!=type(2) and type(M_threshold)!=type(2.5):
            raise ValueError("The value of the \"M_threshold\" argument must be a number. Try again.")
        elif M_threshold<0:
            raise ValueError("The value of the \"M_threshold\" argument can not be negative. Try again.")

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

        mass_segments = len(points_dist) # number of Mass range segments
        obs = np.sum(points_dist) # number of obsarvations in the sample

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        mass_sample_with_noise = [np.NaN]
        radius_sample_with_noise = [np.NaN]

        # Initializing storage lists for the Mass and Radius values sample
        mass_sample = []
        radius_sample = []
        
        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"cfl")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            M_data = sol_data[3] # getting the NS Mass data
            R_data = sol_data[4] # getting the NS Radius data
            
            # Obtaining the pressure, mass and radius values that do not violate causality
            M_caus = M_data[0]
            R_caus = R_data[0]

            # Sampling Mass and Radius values only if there are causality valid Mass values more than M_threshold
            if max(M_caus)>=M_threshold:

                # Filtering the M_caus data to contain Mass values over M_threshold of Solar Mass
                idx_filt = [j for j, mass_value in enumerate(M_caus) if mass_value>=M_threshold]
                M_caus_filt = [M_caus[j] for j in idx_filt]

                # Getting the respective Radius values from the R_caus data list
                R_caus_filt = [R_caus[j] for j in idx_filt]
                
                # Getting the Mass bounds of the Mass range segments
                M_range = np.linspace(min(M_caus_filt),max(M_caus_filt),mass_segments+1)
                
                # Sampling Mass and Radius values at each segment
                for i in range(0,mass_segments):
                    # Index position of Mass values in M_caus list that lie inside the interval [M_range[i],M_range[i+1]]
                    idx_seg = [j for j, mass_value in enumerate(M_caus_filt) if (mass_value>=M_range[i])*(mass_value<=M_range[i+1])]

                    # Mass values in M_caus_filt list that lie inside the interval [M_range[i],M_range[i+1]]
                    M_seg_data = [M_caus_filt[j] for j in idx_seg]

                    # Checking if the M_seg_data list contains less elements than the random choices to be made
                    # This is crucial since we need any element of the list to be (randomly) selected only once
                    if len(M_seg_data)<points_dist[i]:
                        raise ValueError(f"In the Mass segment {i+1} there are {len(M_seg_data)} available Mass values and {points_dist[i]} different random choices requested to be made. Try again.")

                    # Radius values in R_caus_filt list that correspond to the Mass values of M_seg_data list
                    R_seg_data = [R_caus_filt[j] for j in idx_seg]
                    
                    # Sample of Mass values from the current Mass range segment
                    mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    while check_same_value(mass_seg_sample):
                        mass_seg_sample = random.choices(M_seg_data,k=points_dist[i])
                    #print(mass_seg_sample)
                    mass_seg_sample = np.sort(mass_seg_sample) # sorting the mass values in the segment's sample in ascending order
                    
                    # Getting the sample of the respective Radius values
                    radius_seg_sample = []
                    for mass in mass_seg_sample:
                        idx_mass = M_seg_data.index(mass)
                        radius = R_seg_data[idx_mass]
                        radius_seg_sample.append(radius)
                    #print(radius_seg_sample)

                    # Appening to the storage lists
                    mass_sample.append(mass_seg_sample)
                    radius_sample.append(radius_seg_sample)
        
                # Combining the samples of each Mass segment into total samples for Mass and Radius
                mass_sample = np.concatenate((mass_sample),axis=None)
                radius_sample = np.concatenate((radius_sample),axis=None)

                # Adding noise to the Mass and Radius samples
                mass_sample_with_noise = mass_sample + np.random.normal(loc=noiseM_mv,scale=noiseM_std,size=obs)
                radius_sample_with_noise = radius_sample + np.random.normal(loc=noiseR_mv,scale=noiseR_std,size=obs)

        return [mass_sample_with_noise,radius_sample_with_noise]
    

    # Method that samples Slope (dE_dP) and Energy density on center data (that do not violate causality), center pressure at maximum mass from TOV solution data files of a CFL matter QS EOS
    def sample_EOS(self,filename,Pc_points=[10,100,300,600,800,1000,1200,1400],noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0):
        """
        Scanning a file containing the TOV equations' solution data for a CFL matter Quark Star EOS and sampling Slope (dE_dP) and Energy Density at center values and center pressure at maximum mass
        that do not violate causality. Artificial observational noise (following normal distribution) can be added to the values of the samples.
        1. filename: name of the file to be scanned
        2. Pc_points: values (points) of pressure in center of the CFL matter Quark Star, on which the algorithm will collect the values of Slope (dE_dP) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,50,100,200,400,800] MeV*fm^-3.
        3. noiseSl_mv: mean value of the normal distributed observational noise for the values of the Slope sample. By default its value is set to 0.
        4. noiseSl_std: standard deviation of the normal distributed observational noise for the values of the Slope sample. By default its value is set to 0.
        5. noiseEc_mv: mean value of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        6. noiseEc_std: standard deviation of the normal distributed observational noise for the values of the Energy Density at center sample. By default its value is set to 0.
        """ 
        
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

        # Initializing storage lists for the Slope (dE_dP) and Energy density on center values samples
        dEdP_sample = []
        enrg_dens_sample = []
        dEdP_sample_with_noise = [np.NaN]
        enrg_dens_sample_with_noise = [np.NaN]
        Pc_max_mass = np.NaN

        # Scanning for the file
        if os.path.exists(filename):
            sol_data = file_read(filename,"cfl")
            Pc_data = sol_data[0] # getting the NS pressure on center data
            Ec_data = sol_data[1] # getting the NS Energy Density on center data
            Slope_data = sol_data[2] # getting the NS Slope data
            M_data = sol_data[3] # getting the NS Mass data

            # Getting the data that do not violate causality
            Pc_caus = Pc_data[0]
            Ec_caus = Ec_data[0]
            Slope_caus = Slope_data[0]
            M_caus = M_data[0]

            # Sampling Slope (dE_dP) and Energy density on center values only if the causality part of the EOS overcomes 
            # the value of the maximum pressure point plus 50
            if Pc_caus[-1]>=max(Pc_points)+50:
                for i in range(0,n):
                    idx_press_val = Pc_caus.index(Pc_points[i])
                    dEdP_sample.append(Slope_caus[idx_press_val])
                    enrg_dens_sample.append(Ec_caus[idx_press_val])
                 
                # Getting the center pressure at maximum mass
                idx_max_mass = np.argmax(M_caus)
                Pc_max_mass = Pc_caus[idx_max_mass]

                #print(max(M_caus),Pc_max_mass)

                # Adding noise to the Mass and Radius samples
                dEdP_sample_with_noise = dEdP_sample + np.random.normal(loc=noiseSl_mv,scale=noiseSl_std,size=n)
                enrg_dens_sample_with_noise = enrg_dens_sample + np.random.normal(loc=noiseEc_mv,scale=noiseEc_std,size=n)

        return [dEdP_sample_with_noise,enrg_dens_sample_with_noise,Pc_max_mass]
    
    # Method that generates and records on .csv files data of CFL matter Quark Stars for regression purposes
    def gen_reg_data(self,save_filename,samples_per_EOS=1,M_threshold=0,points_dist=[3,3,3,3,3],Pc_points=[10,100,300,600,800,1000,1200,1400],noises_mv=[0,0,0,0],noises_std=[0,0,0,0]):
        """
        Getting data of CFL matter Quark Stars for regression purposes and recording them on .csv files
        1. save_filename: the name of the final .csv file, in which the regression data are being recorded
        2. samples_per_EOS: number of samples to be generated per CFL EOS. Each sample is recorded as a row in the final .csv file and includes the
        selected values of Speed of Sound = dP_dE, Energy Density at center, Mass and Radius. By default, 1 sample is generated per CFL EOS.
        3. M_threshold: Threshold of Mass values. In order for the algorithm to create Mass and Radius samples, the EOS must have resulted in causality valid 
        Mass values greater than M_threshold
        4. points_dist: list of random points to be selected. The algorithm divides the range of Mass data that do not violate causality into 
        as many segments as the length of the list 'points_dist'. Then it selects randomly (and uniformly), as many points per segment,
        as the respective value of the list's element, that corresponds to that segment. By default, the list [3,3,3,3,3] is given
        as input for the 'points_dist' argument, i.e. the Mass range is divided into 5 segments and 3 points are randomly selected per segment.
        5. Pc_points: values (points) of pressure in center of the polytropic Neutron Star, on which the algorithm will collect the values of Slope (dP_dE) and Energy Density.
        By default the following points are selected: 'Pc_points' = [10,100,300,600,800,1000,1200,1400] MeV*fm^-3.
        6. noises_mv: list containing the mean values for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the mean values are set to 0.
        7. noises_std: list containing the standard deviations for the artificial observational noise that is added to the sample values of the following: 1st element-> Mass, 2nd element -> Radius, 3rd element -> Slope (dP_dE) and 4th element -> Energy Density at center. By default the standard deviations are set to 0.
        """ 

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
        num_mr_points = np.sum(points_dist)

        # Getting the number of pressure points
        num_pc_points = len(Pc_points)

        # Creating the file in which the regression data will be recorder and forming its headers
        headers_slope = f"" # headers for the Slope (dP_dE) values
        headers_enrg = f"" # headers for the Energy Density at center values
        headers_Pc_max_mass = "Pc(M_max)," # headers for the center pressure at maximum mass 
        headers_mass = f"" # headers for the Mass values
        headers_radius = f"" # headers for the Radius values

        # Forming the headers for the Y data (response variables) of the regression, i.e. the Slope (dP_dE) and Energy Density at center values
        for i in range(0,num_pc_points):
            headers_slope = headers_slope + f"dE_dP({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis
            headers_enrg = headers_enrg + f"E_c({Pc_points[i]})," # the values of the pressure are icluded inside the paranthesis

        # Forming the headers for the X data (explanatory variables) of the regression, i.e. the Mass and Radius values
        for i in range(0,num_mr_points):
            headers_mass = headers_mass + f"M_{i+1},"
            if i==np.sum(points_dist)-1:
                headers_radius = headers_radius + f"R_{i+1}\n" # the last column of the headers needs \n and not a comma in the end
            else:
                headers_radius = headers_radius + f"R_{i+1},"   
        
        # Forming the total info of the headers and the name of the recording .csv file
        headers_info = headers_slope + headers_enrg + headers_Pc_max_mass + headers_mass + headers_radius
        with open(f"{save_filename}.csv","w") as file:
            file.write(headers_info)

        # Creating a copy .csv file where the values in the columns of the X data are shuffled rowwise to avoid correlation between them 
        with open(f"{save_filename}_rwshuffled.csv","w") as file:
            file.write(headers_info)    

        for cfl_model_name in self.total_cfl_models_info[0]:
            # Getting the name of the file to be scanned
            filename = f"{cfl_model_name}_sol.csv"

            # Getting the basic sample of the Slope (dE_dP) and Energy Density at center values
            dEdP_basic_sample,enrg_basic_sample,Pc_max_mass = self.sample_EOS(filename,Pc_points,noiseSl_mv=0,noiseSl_std=0,noiseEc_mv=0,noiseEc_std=0)

            # Getting the basic sample of the Mass and Radius values
            mass_basic_sample,radius_basic_sample = self.sample_MR(filename,M_threshold,points_dist,noiseM_mv=0,noiseM_std=0,noiseR_mv=0,noiseR_std=0)
                    
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
                    row_slope_info = f"" # row info for the Slope (dE_dP) values
                    row_enrg_info = f"" # row info for the Energy Density at center values
                    row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass 
                    row_mass_info = f"" # row info for the Mass values
                    row_radius_info = f"" # rwo info for the Radius values

                    # Initializing the row info for the shuffled .csv file
                    shuffled_row_slope_info = f"" # row info for the Slope (dE_dP) values
                    shuffled_row_enrg_info = f"" # row info for the Energy Density at center values
                    shuffled_row_Pc_max_mass_info = f"{Pc_max_mass}," # row info for the center pressure at maximum mass 
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
                    row_info = row_slope_info + row_enrg_info + row_Pc_max_mass_info + row_mass_info + row_radius_info
                    with open(f"{save_filename}.csv","a+") as file:
                        file.write(row_info)

                    # Getting the total info of the row and recording it to the final shuffled .csv file
                    shuffled_row_info = shuffled_row_slope_info + shuffled_row_enrg_info + shuffled_row_Pc_max_mass_info + shuffled_row_mass_info + shuffled_row_radius_info
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