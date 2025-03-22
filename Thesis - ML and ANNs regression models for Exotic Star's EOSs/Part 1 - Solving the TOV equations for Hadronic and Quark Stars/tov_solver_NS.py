# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs and classification of Exotic Stars using ML and ANNs models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py2a
# Name: tov_solver_NS.py

# Description: 
# -> Solving the TOV equations for the Neutron Stars' EoSs included in the 'eos_lib_NS.py' script
# -> Storaging the solutions in .csv files

# Abbrevations:
# NS -> Neutron Star


# Importing useful modules
import numpy as np
import sympy as smp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys as sys
from prettytable import PrettyTable
import eos_lib_NS

# Defining a function that returns the system of the TOV equations
def tov_eq(r,y,EOS):
    # Appending values
    P = y[0] # value of the pressure at radius r
    M = y[1] # value of the mass at radius r
    e = EOS(P) # value of the energy's density at radius r

    # Defining the system of TOV equations
    dP_dr = -1.474*(e*M/r**2)*(1+P/e)*(1+11.2e-6*r**3*P/M)/(1-2.948*M/r)
    dM_dr = 11.2e-6*r**2*e
    return [dP_dr,dM_dr] 


# Defining a function that solves the system of TOV equations
# for a given EOS
def tov_sol(p_0,r_step,EOS_name,EOS_func,EOS_func_sym,i,n):
    # Pressure and mass initial values - initial conditions
    Pc = p_0
    P_0 = Pc # initial pressure = pressure in center of NS
    M_0 = 10**-12 # mass in center of the NS
    initial_values = [P_0,M_0]

    # Defining the slope of the selected EOS as a function
    p = smp.symbols("p")
    def EOS_slope(p_value):
        return smp.diff(EOS_func_sym(p),p).subs(p,p_value)
        
    # Calculate the slope, associated with the speed of sound at the center of the NS
    dE_dP = EOS_slope(Pc)
        
        
    # Bounds of NS radius r interval for the 1st step 
    # in the solving process
    r_min: float = 1e-9
    r_max: float = 1e-2
        
    # Printing the progress of the solving process
    NS_area = "Core"
    show_progress = sys.stdout
    show_progress.write("\r" + " " *100) # clearing previous output
    show_progress.flush()
    show_progress.write(f"\rUsed initial values: {i+1}/{n} - Center Press:{Pc:.2f} - Press:{P_0} ({NS_area})")
    show_progress.flush()

    # Useful storage lists
    M_vals = np.array([]) # mass of NS values
    R_vals = np.array([]) # radius of the NS values
    P_vals = np.array([]) # pressure in the NS

    # Crust-core pressure bound
    p_bound_crust = 0.184
    if EOS_name == 'PS':
        p_bound_crust = 0.696    
        
    # Solving the TOV equations for the current pressure in center
    k=1
    while P_0>10**-12:
        # Changing the core EoS to a crust EoS if the crust of the NS has been reached
        if 9.34375*10**-5 < P_0 < p_bound_crust:
            NS_area = "Crust"
            EOS_func = eos_lib_NS.eos_crust1
        elif 4.1725*10**-8 < P_0 < 9.34375*10**-5:
            EOS_func = eos_lib_NS.eos_crust2
        elif 1.44875*10**-11 < P_0 < 4.1725*10**-8:
            EOS_func = eos_lib_NS.eos_crust3
        elif P_0 < 1.44875*10**-11:
            EOS_func = eos_lib_NS.eos_crust4    

        # Solving the TOV equations system in the [r_min,r_max] radius interval  
        solution = solve_ivp(tov_eq,(r_min,r_max),initial_values,method="LSODA",args=(EOS_func,),atol=10**-12,rtol=10**-8)
        # Updating the initial values to be used in the next [r_min,r_max] interval
        initial_values[0] = solution.y[0][-1] # pressure
        initial_values[1] = solution.y[1][-1] # mass

        # Check if the new initial mass is negative or zero
        # and break if so
        if initial_values[1]<=0:
            break

        # Check if the new initial pressure is equal to the previous
        # initial pressure and break if so
        if initial_values[0]==P_0:
            break
            
        # Update the bounds r_min, r_max and the initial value of pressure P_0
        r_min = solution.t[-1]
        r_max = r_min + r_step
        P_0 = initial_values[0]

        # Update the progress info
        show_progress.write(f"\rUsed initial values: {i+1}/{n} - Center Press:{Pc:.2f} - Press:{P_0} ({NS_area})")
        show_progress.flush()

        # Appending values to the storage lists
        M_vals = np.append(M_vals,solution.y[1][~np.isnan(solution.y[1])])
        R_vals = np.append(R_vals,solution.t)
        P_vals = np.append(P_vals,solution.y[0][~np.isnan(solution.y[0])])

        k = k+1
    if P_vals[-1]<0:
        idx = np.argwhere(P_vals<0)[0 ,0]
        P_vals = np.delete (P_vals ,np.s_[idx::],0)
        M_vals = np.delete (M_vals , np.s_[ idx::],0)
        R_vals = np.delete (R_vals, np.s_[idx::],0)

    # Storaging the solution's data in a list
    sol_data = [min(P_vals),Pc,EOS(Pc),dE_dP,max(M_vals),R_vals[-1]]
    # Store the solution's data in the .csv file 
    filename = f'{EOS_name}_sol.csv'
    with open(filename,"a+") as file:
        file.write(f"{min(P_vals)}, {Pc}, {EOS(Pc)},{dE_dP},{max(M_vals)}, {R_vals[-1]}\n")
    return sol_data


# Appending the EOSs info to lists
EOS_list = PrettyTable()
EOS_names = []
EOS_formulas = []
EOS_formulas_sym = []
for EOS_info in eos_lib_NS.eos_list_core:
    EOS_names.append(EOS_info[0])
    EOS_formulas.append(EOS_info[1])
    EOS_formulas_sym.append(EOS_info[2])
EOS_list.add_column("Offered EOSs models",EOS_names)    

# Printing the solution process info
print("\n>SOLVING THE TOV EQUATIONS SYSTEM FOR A NEUTRON STAR\n")
print('=============================================================')

# Showing the offered EOSs
print(EOS_list)
print('-------------------------------------------------------------')

# Asking the user to select an EOS  
EOS_name = input("Give the EOS name:  ")
while EOS_names.count(EOS_name)==0:
    EOS_name = input("EOS not found, give another name:  ")
EOS = EOS_formulas[EOS_names.index(EOS_name)] # numerical formula of the selected EOS
EOS_sym = EOS_formulas_sym[EOS_names.index(EOS_name)] # symbolic formula of the selected EOS
print('-------------------------------------------------------------')

# Printing the EOS symbolic formula
print("Solving for: %s"%EOS_name)
print("EOS formula:")
p = smp.symbols("p")
formula = f"E(p)={EOS_sym(p)}"
print(formula)
print('-------------------------------------------------------------')

# Create a .csv file to save the solution data
filename = f'{EOS_name}_sol.csv'
with open(filename,"w") as file:
    file.write("P_out, P_c, E, dE/dP, M, R\n")
print("Solution data being saved in %s file"%filename)
print('-------------------------------------------------------------')

# Defining lists that contain initial values for the pressure
# in the center of the NS
P_in_center_1: np.ndarray = np.arange(1.5,5,0.1)
P_in_center_2: np.ndarray = np.arange(5,1201,1)
# Combining the lists of the initial pressure values
P_in_center_total: np.ndarray = np.concatenate((P_in_center_1,P_in_center_2),axis=None)
# Printing the pressure info
print("Minimum center pressure: %.2e [MeV/fm^3]"%P_in_center_total[0])
print("Maximum center pressure: %.2e [MeV/fm^3]"%P_in_center_total[-1])

# Defining a list to store the data of the solution
n = len(P_in_center_total)
total_results = np.zeros((n,6))
print("Total initial values of pressure in NS center: %d"%n)
print('-------------------------------------------------------------')

# Ask the user to give the radius step (in km) for the solving process
R_step = float(input("Give the radius step in km:  "))
print("Radius step length in solution process: %.3e [km]"%R_step)
print('-------------------------------------------------------------')
for i in range(0,n):
    total_results[i] = tov_sol(P_in_center_total[i],R_step,EOS_name,EOS,EOS_sym,i,n)    
print('\n=============================================================\n\n')

# Printing the results
E = total_results[:,2] # NS center energy density data
print(">NS Center energy density data:")
print(E)
Pc = total_results[:,1] # NS center pressure data
print("\n>NS Center pressure data:")
print(Pc)
M = total_results[:,4] # NS mass data
print("\n>NS Mass data:")
print(M)
R = total_results[:,5] # NS radius data
print("\n>NS Radius data:")
print(R)

# Plotting and saving the M-R diagram of the NS
plt.plot(R,M)
plt.title("M-R plot for %s EOS"%EOS_name)
plt.xlabel(r"$R$ $[km]$")
plt.ylabel(r"$M$ $(M_\odot)$")
plt.savefig(f"{EOS_name}_M_R.pdf",dpi=200)

print("\n>The M-R figure has been saved !!!")
