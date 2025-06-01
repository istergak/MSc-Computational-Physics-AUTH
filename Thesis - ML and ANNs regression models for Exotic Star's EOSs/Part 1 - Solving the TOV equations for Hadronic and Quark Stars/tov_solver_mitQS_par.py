# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py6b
# Name: tov_solver_mitQS_par.py

# Description: 
# -> Parallel solving of the TOV equations for MIT bag EOS models of Quark Stars
# -> Storaging the solutions in .csv files

# Abbrevations:
# QS -> Quark Star


# Importing useful modules
import numpy as np
import sympy as smp
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys as sys
import os
from prettytable import PrettyTable
import multiprocessing
import time 

# Defining useful consants
m_s = 95 # MeV, mass of Strange quark
m_n = 939.565 # MeV, mass of Neutron
hbarc = 197.327 # MeV*fm, hbar*c constant
B_ult_min = 57 # MeV*fm^-3, minimum value of the B constant of the MIT bag model

# Defining a function that finds all the MIT bag models, provided
# a maximum and a step for the bag parameter B. The minimum of B is fixed at 60 MeV*fm^-3
def MIT_bag_models(B_max,B_step):
    models_names = []
    B_values = []

    i = 1
    for B_val in range(60,B_max+B_step,B_step):
        models_names.append(f"MITbag-{i}")
        B_values.append(B_val)
        i=i+1

    return [models_names,B_values]            

# Define a function that returns and prints all the given combinations of B_eff and Δ parameters
# in groups, depending on the result of the Beff_Delta_valid_combos() function
# as well as the selected number of combinations per group
def Beff_Delta_groups_maker(MITbag_combos,num_per_group):
    models_names = MITbag_combos[0]
    B_values = MITbag_combos[1]

    total_combos = len(models_names)

    # Forming and printing the groups of Γ combinations
    mod = total_combos%num_per_group
    upper_limit = total_combos-mod

    models_names_groups = []
    B_values_groups = []
    

    for i in range(0,upper_limit,num_per_group):
        models_names_groups.append(models_names[i:i+num_per_group])
        B_values_groups.append(B_values[i:i+num_per_group])
    if mod!=0:
        models_names_groups.append(models_names[upper_limit:total_combos])
        B_values_groups.append(B_values[upper_limit:total_combos])
    m = len(models_names_groups)
    for j in range(0,m):
        group_info = PrettyTable()
        group_info.add_column("Models",models_names_groups[j])
        group_info.add_column("B [MeV*fm^-3]",B_values_groups[j])
        print(f"MIT-bag  Group {j+1}:")
        print(group_info,"\n")
    
    return [models_names_groups,B_values_groups]

# Defining the generator for MIT-bag quark matter EoSs of a QS 
# as a function of the pressure P and the bag parameter B

# Numerical definition of EOS
def EOS_MITbag(p,B):
    e = 3*p + 4*B
    return e

# Numerical definition of the slope dE/dP of EOS
def EOS_MITbag_slope(p):
    e = 3
    return e


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
def tov_sol(p_0,r_step,B_val,EOS_name):
    # Defining the formula of the EOS function
    def EOS_formula(P):
        return EOS_MITbag(P,B_val)
    
    # Calculating the mass density, energy density, EOS slope (dE/dP) and speed of sound c_s at NS center
    Pc = p_0 # pressure at QS center
    E_c = EOS_formula(Pc) # energy density at QS center
    dE_dP = EOS_MITbag_slope(Pc) # EOS slope at QS center
  
     
    # Pressure and mass initial values (initial conditions)       
    P_0 = Pc # initial pressure = pressure in center of NS
    M_0 = 10**-12 # mass in center of the NS
    initial_values = [P_0,M_0]
            
    # Bounds of NS radius r interval for the 1st step 
    # in the solving process
    r_min: float = 1e-9
    r_max: float = 1e-2

    # Useful storage lists
    P_out = None
    M_vals = []
    R_final = None
    
            
    # Solving the TOV equations for the main core and crust section
    EOS_func = EOS_formula
    while P_0>10**-12:

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

        # Appending values to the M_max, R_final, P_final variables
        P_out = solution.y[0][-1]
        if solution.y[1][-1]!=np.nan:
            M_vals.append(solution.y[1][-1])
        if solution.t[-1]!=np.nan:
            R_final = solution.t[-1]
 

    
    # if P_vals[-1]<0:
    #     idx = np.argwhere(P_vals<0)[0 ,0]
    #     P_vals = np.delete (P_vals ,np.s_[idx::],0)
    #     M_vals = np.delete (M_vals , np.s_[ idx::],0)
    #     R_vals = np.delete (R_vals, np.s_[idx::],0)
    
    # Store the solution's data in the .csv file 
    filename = f'{EOS_name}_sol.csv'
    with open(filename,"a+") as file: 
        file.write(f"{P_out}, {Pc}, {E_c}, {dE_dP},{max(M_vals)}, {R_final}\n")

# Defining the worker function for the parallel solving 
def tov_sol_worker(task_id,EOS_name,B_value,progress_queue):

    # Pressure ranges
    P_in_center_1: np.ndarray = np.arange(1,5,0.1) # lower pressure range
    P_in_center_2: np.ndarray = np.arange(5,1501,1) # higher pressure range
    P_in_center_total: np.ndarray = np.concatenate((P_in_center_1,P_in_center_2),axis=None)

    # Radius step in the scan of the NS
    R_step = 0.001 # km

    # Total number of initial values
    total_values = len(P_in_center_total)

    filename = f'{EOS_name}_sol.csv'
    with open(filename,"w") as file:
        file.write(f"P_out, P_c, E_c, dE_dP, M, R, B={B_value:.1f}\n")

    for i in range(0,total_values):
        P_0 = P_in_center_total[i]

        # Send progress update to the queue (task_id, current number of used initial values, total number of initial values)
        progress_queue.put((task_id,EOS_name,i + 1, total_values,P_0))
        
        # Calling the tov_sol() function
        tov_sol(P_0,R_step,B_value,EOS_name)

def print_progress(progress_queue, num_tasks):
    progress = {i: (0, 1, 2, 3) for i in range(num_tasks)}  # Dictionary to track (step, total_steps) for each task
    
    # Starting the cpu execution time measurement
    start_time = time.time()
    while True:
        message = progress_queue.get()
        
        if message == "DONE":
            break
        
        task_id, EOS_name,step, total_steps,P_0 = message
        progress[task_id] = (EOS_name,step, total_steps,P_0)
        
        # Clear the line and update the progress of all tasks in the same spot
        sys.stdout.write("\r")  # Carriage return to the beginning of the line
        sys.stdout.write(" | ".join([f"{progress[i][0]}:{progress[i][3]:.1f}" for i in range(num_tasks)]))
        sys.stdout.flush()
    
    # Terminating the cpu execution time measurement
    end_time = time.time()
    cpu_time_total = (end_time - start_time) # total execution time in seconds
    cpu_time_res = cpu_time_total%60 # remaining seconds if we express execution time in minutes
    cpu_time_total_mins = (cpu_time_total - cpu_time_res)/60 # minutes of the total execution time
    print('\n-------------------------------------------------------------------------------')
    print("\nAll tasks have been completed......")
    print("\nElapsed time: %.1f\'%.2f\""%(cpu_time_total_mins,cpu_time_res))
    print('\n-------------------------------------------------------------------------------')

def task_giver(models_selected,B_selected_group,progress_queue):
    n = len(models_selected)
    # Appending the tasks to be parallelized
    tasks = []
    for i in range(0,n):
        tasks.append(pool.apply_async(tov_sol_worker, args=(i,models_selected[i],B_selected_group[i],progress_queue))) 


        
    

if __name__ == "__main__":
    # Printing the solution process info
    print("\n>SOLVING THE TOV EQUATIONS SYSTEM FOR A (CFL MATTER) QUARK STAR \n")
    print('===============================================================================')
    print('>>Determination of the range for the B_eff and Δ parameters:')
    print('-------------------------------------------------------------------------------')
    
    print("Minimum value of bag parameter B in our study: 60.00 [MeV*fm^-3]")
    # Ask the user to give the maximum value for the Beff parameter
    B_max_val = int(input("Give the maximum value of the bag parameter B in [MeV*fm^-3]:  "))
    while B_max_val<=57:
        B_max_val = int(input("The value must be greater than 57 [MeV*fm^-3]. Try again:  "))

    # Ask the user to give the step for the scan of the Beff area
    B_step_val = int(input("Give the step for the scan of the B parameter area in [MeV*fm^-3]:  "))
    while B_step_val<=0:
        B_step_val = int(input("The value must be positive. Try again:  "))
    print('-------------------------------------------------------------------------------')

    MIT_bag_combos_results = MIT_bag_models(B_max_val,B_step_val)
    print("Total valid combinations:  %d"%len(MIT_bag_combos_results[0]))
    print('-------------------------------------------------------------------------------')

    # Ask the user to give the number of valid Beff and Δ combinations per group
    number_per_group = int(input("Give the number of valid MIT-bag models per group:  "))
    while number_per_group<=0:
        number_per_group = int(input("The number must be positive. Try again:  "))
    print('-------------------------------------------------------------------------------')
    
    print(">>Groups of models:")
    print('-------------------------------------------------------------------------------\n')
    # Getting and printing the groups of Γ combinations
    models_names_groups,B_groups = Beff_Delta_groups_maker(MIT_bag_combos_results,number_per_group)
    l = len(models_names_groups)
    
    print('-------------------------------------------------------------------------------')
    group_choosing = int(input("Choose the group to be solved:  "))
    while group_choosing<1 or group_choosing>l:
        group_choosing = int(input("Group not found. Choose another group to be solved:  "))
    print('-------------------------------------------------------------------------------') 
    models_names_group_select = models_names_groups[group_choosing-1]
    B_group_select = B_groups[group_choosing-1]
    num_tasks = len(models_names_group_select)
        

    # Create a multiprocessing Manager to share data between processes
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()


    try:
        # Start the progress printing process
        progress_process = multiprocessing.Process(target=print_progress, args=(progress_queue, num_tasks))
        progress_process.start()

        # Create a pool of worker processes
        pool = multiprocessing.Pool()
        
        # Supervising the solving process
        print("Supervising the solving process (current initial pressure values are being shown):")
        # Example: Each task has a different number of steps
        tasks = task_giver(models_names_group_select,B_group_select,progress_queue)

        # Close the pool and wait for the workers to finish
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\nTerminating the process due to keyboard interruption...")
        
        # Terminate the worker processes
        pool.terminate()
        pool.join()

        # Ensure progress process is terminated
        progress_queue.put("DONE")
        progress_process.terminate()
        progress_process.join()

    # Signal the progress printer process to stop if not interrupted
    if progress_process.is_alive():
        progress_queue.put("DONE")
        progress_process.join()