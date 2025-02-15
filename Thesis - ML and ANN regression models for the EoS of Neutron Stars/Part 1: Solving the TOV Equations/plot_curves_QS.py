# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs and classification of Exotic Stars using ML and ANNs models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script 6
# Name: plot_curves_QS.py

# Description: 
# -> Plotting the EOSs curves of Quark Stars
# -> Plotting the respective M-R curves 

# Abbrevations:
# QS -> Quark Star


# Importing useful modules
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
import eos_lib_QS

print("\n>PLOTTING THE EOSs AND M-R CURVES OF NEUTRON STARS\n")
print('=============================================================')

fig1, axes_EOS = plt.subplots(1,2,figsize=(15,6))
fig2, axes_MR = plt.subplots(1,2,figsize=(15,6))

j=1
for i in range(0,12):
    EOS_info = eos_lib_QS.eos_list_cfl[i]
    EOS_name_cfl = EOS_info[0]
    EOS_clf_B_eff = EOS_info[3]
    EOS_cfl_Delta = EOS_info[4]
    filename = f"{EOS_name_cfl}_sol.csv"

    # Checking if the file exists and obtain its data if so
    if os.path.exists(filename):
        EOS_data = pd.read_csv(filename)
        Pc_data = EOS_data.iloc[:,1]
        E_data = EOS_data.iloc[:,2]
        M_data = EOS_data.iloc[:,4]
        R_data = EOS_data.iloc[:,5]

        idx_max = np.argmax(M_data)
        Pc_data_stable = Pc_data[0:idx_max+1]
        E_data_stable = E_data[0:idx_max+1]
        M_data_stable = M_data[0:idx_max+1]
        R_data_stable = R_data[0:idx_max+1]

        Pc_data_unstable = Pc_data[idx_max+1:-1]
        E_data_unstable = E_data[idx_max+1:-1]
        M_data_unstable = M_data[idx_max+1:-1]
        R_data_unstable = R_data[idx_max+1:-1]
        
        if j<=9:
            axes_EOS[0].plot(Pc_data_stable,E_data_stable,"-",label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")
            axes_MR[0].plot(R_data_stable,M_data_stable,".",ms=3,label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")

            axes_EOS[0].plot(Pc_data_unstable,E_data_unstable,'--',color="darkgrey")
            axes_MR[0].plot(R_data_unstable,M_data_unstable,'--',color="darkgrey")
            
        else:
            axes_EOS[0].plot(Pc_data_stable,E_data_stable,'--',label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")
            axes_MR[0].plot(R_data_stable,M_data_stable,'--',label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")

            axes_EOS[0].plot(Pc_data_unstable,E_data_unstable,'--',color="darkgrey")
            axes_MR[0].plot(R_data_unstable,M_data_unstable,'--',color="darkgrey")
        j=j+1    

j=1
for i in range(12,23):
    EOS_info = eos_lib_QS.eos_list_cfl[i]
    EOS_name_cfl = EOS_info[0]
    EOS_clf_B_eff = EOS_info[3]
    EOS_cfl_Delta = EOS_info[4]
    filename = f"{EOS_name_cfl}_sol.csv"
   
    # Checking if the file exists and obtain its data if so
    if os.path.exists(filename):
        EOS_data = pd.read_csv(filename)
        Pc_data = EOS_data.iloc[:,1]
        E_data = EOS_data.iloc[:,2]
        M_data = EOS_data.iloc[:,4]
        R_data = EOS_data.iloc[:,5]

        idx_max = np.argmax(M_data)
        Pc_data_stable = Pc_data[0:idx_max+1]
        E_data_stable = E_data[0:idx_max+1]
        M_data_stable = M_data[0:idx_max+1]
        R_data_stable = R_data[0:idx_max+1]

        Pc_data_unstable = Pc_data[idx_max+1:-1]
        E_data_unstable = E_data[idx_max+1:-1]
        M_data_unstable = M_data[idx_max+1:-1]
        R_data_unstable = R_data[idx_max+1:-1]
        
        if j<=9:
            axes_EOS[1].plot(Pc_data_stable,E_data_stable,"-",label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")
            axes_MR[1].plot(R_data_stable,M_data_stable,".",ms=3,label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")

            axes_EOS[1].plot(Pc_data_unstable,E_data_unstable,'--',color="darkgrey")
            axes_MR[1].plot(R_data_unstable,M_data_unstable,'--',color="darkgrey")
            
        else:
            axes_EOS[1].plot(Pc_data_stable,E_data_stable,'--',label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")
            axes_MR[1].plot(R_data_stable,M_data_stable,'--',label=f"{EOS_name_cfl} ({EOS_clf_B_eff},{EOS_cfl_Delta})")

            axes_EOS[1].plot(Pc_data_unstable,E_data_unstable,'--',color="darkgrey")
            axes_MR[1].plot(R_data_unstable,M_data_unstable,'--',color="darkgrey")    
        j=j+1   

# Adding labels and legends on the EOSs graphs
axes_EOS[0].set_title("a)",fontsize=16)
axes_EOS[1].set_title("b)",fontsize=16)
axes_EOS[0].set_xlabel(r"$P_c$ $[MeV*fm^{-3}]$",fontsize=14)
axes_EOS[1].set_xlabel(r"$P_c$ $[MeV*fm^{-3}]$",fontsize=14)
axes_EOS[0].set_ylabel(r"$E$ $[MeV*fm^{-3}]$",fontsize=14)
axes_EOS[1].set_ylabel(r"$E$ $[MeV*fm^{-3}]$",fontsize=14)
axes_EOS[0].legend(fontsize=10)
axes_EOS[1].legend(fontsize=10)
axes_EOS[0].set_facecolor(color=[0.9,0.9,0.9])
axes_EOS[1].set_facecolor(color=[0.9,0.9,0.9])


# Adding labels and legends on the M-R graphs
axes_MR[0].set_title("a)",fontsize=16)
axes_MR[1].set_title("b)",fontsize=16)
axes_MR[0].set_xlabel(r"$R$ $[km]$",fontsize=14)
axes_MR[1].set_xlabel(r"$R$ $[km]$",fontsize=14)
axes_MR[0].set_ylabel(r"$M$ $(M_\odot)$",fontsize=14)
axes_MR[1].set_ylabel(r"$M$ $(M_\odot)$",fontsize=14)
#axes_MR[0].set_xbound([0,16])
#axes_MR[1].set_xbound([0,16])
axes_MR[0].legend(fontsize=10)
axes_MR[1].legend(fontsize=10)
axes_MR[0].set_facecolor(color=[0.9,0.9,0.9])
axes_MR[1].set_facecolor(color = [0.9,0.9,0.9])


# Saving the figures
fig1.savefig("EOS_curves_QS_cfl_branches.pdf",dpi=200)
print("\nEOS figure saved sucessfully !!!")
fig2.savefig("M_R_curves_QS_cfl_branches.pdf",dpi=200)        
print("M-R figure saved sucessfully !!!\n")

print('=============================================================')
