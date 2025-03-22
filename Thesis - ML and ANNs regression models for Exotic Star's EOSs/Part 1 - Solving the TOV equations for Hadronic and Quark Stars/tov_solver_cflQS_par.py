# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py6
# Name: tov_solver_cflQS_par.py

# Description: 
# -> Parallel solving of the TOV equations for CFL EOS models of Quark Stars
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
Beff_min = 57 # MeV*fm^-3, minimum value of the B_eff constant of the MIT bag model
Delta_min = 50 # MeV, minimum value of the Δ constant of the MIT bag model

# Defining the bound of the B_eff parameter for stable CFL phase of the quark matter as 
# a function of the Δ parameter
def B_eff_bound(Delta):
    formula = -(m_s**2*m_n**2)/(12*(np.pi**2))+(Delta**2*m_n**2)/(3*(np.pi**2))+(m_n**4)/(108*(np.pi**2))
    return formula/hbarc**3

# Defining a function that finds all the CFL models with valid combinations of the Β_eff and Δ parameters
# for stable CFL quark matter, depending on the maximum selected values of B_eff and Δ respectively,
# as well as the step in scanning the B_eff axis and Δ axis

def valid_cfl_models(Beff_max,Beff_step,Delta_max,Delta_step):
    models_names = []
    valid_Beff = []
    valid_Delta = []
    valid_ms = []

    i = 1
    for B_eff_val in range(60,Beff_max+Beff_step,Beff_step):
        for Delta_val in range(Delta_min,Delta_max+Delta_step,Delta_step):
            if B_eff_val < B_eff_bound(Delta_val):
                models_names.append(f"CFL-{i}")
                valid_ms.append(m_s)
                valid_Beff.append(B_eff_val)
                valid_Delta.append(Delta_val)
                i = i + 1

    return [models_names,valid_Beff,valid_Delta,valid_ms]            

# Define a function that returns and prints all the given combinations of B_eff and Δ parameters
# in groups, depending on the result of the Beff_Delta_valid_combos() function
# as well as the selected number of combinations per group
def Beff_Delta_groups_maker(Beff_Delta_combos,num_per_group):
    models_names = Beff_Delta_combos[0]
    valid_Beff = Beff_Delta_combos[1]
    valid_Delta = Beff_Delta_combos[2]
    valid_ms = Beff_Delta_combos[3]

    total_combos = len(models_names)

    # Forming and printing the groups of Γ combinations
    mod = total_combos%num_per_group
    upper_limit = total_combos-mod

    models_names_groups = []
    valid_Beff_groups = []
    valid_Delta_groups = []
    valid_ms_groups = []
    

    for i in range(0,upper_limit,num_per_group):
        models_names_groups.append(models_names[i:i+num_per_group])
        valid_Beff_groups.append(valid_Beff[i:i+num_per_group])
        valid_Delta_groups.append(valid_Delta[i:i+num_per_group])
        valid_ms_groups.append(valid_ms[i:i+num_per_group])
    if mod!=0:
        models_names_groups.append(models_names[upper_limit:total_combos])
        valid_Beff_groups.append(valid_Beff[upper_limit:total_combos])
        valid_Delta_groups.append(valid_Delta[upper_limit:total_combos])
        valid_ms_groups.append(valid_ms[upper_limit:total_combos])
    m = len(models_names_groups)
    for j in range(0,m):
        group_info = PrettyTable()
        group_info.add_column("Models",models_names_groups[j])
        group_info.add_column("B_eff [MeV*fm^-3]",valid_Beff_groups[j])
        group_info.add_column("Δ [MeV]",valid_Delta_groups[j])
        group_info.add_column("m_s [MeV]",valid_ms_groups[j])
        print(f"Γ Group {j+1}:")
        print(group_info,"\n")
    
    return [models_names_groups,valid_Beff_groups,valid_Delta_groups,valid_ms_groups]

# Defining the generator for stable CFL quark matter EoSs of a QS 
# as a function of the pressure P and the parameters B_eff and Δ

# Numerical definition
def EOS_CFL_stable(p,B_eff,Delta):
    a = -m_s**2/6 + 2*Delta**2/3
    μ_sq = -3*a + np.sqrt(4/3*(np.pi**2)*(B_eff+p)*hbarc**3 + 9*a**2)
    e = 3*p + 4*B_eff - 9*a*μ_sq/(hbarc**3*np.pi**2)
    return e

# Symbolic definition
def EOS_CFL_stable_sym(p,B_eff,Delta):
    a = -m_s**2/6 + 2*Delta**2/3
    μ_sq = -3*a + smp.sqrt(4/3*(np.pi**2)*(B_eff+p)*hbarc**3 + 9*a**2)
    e = 3*p + 4*B_eff-9*a * μ_sq/(hbarc**3*np.pi**2)
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
def tov_sol(p_0,r_step,Beff_val,Delta_val,EOS_name):
    # Defining the formula of the EOS function
    def EOS_formula(P):
        return EOS_CFL_stable(P,Beff_val,Delta_val)
    
    # Defining the slope of the EOS function
    p = smp.symbols("p")
    def EOS_formula_slope(P_val):
        return smp.diff(EOS_CFL_stable_sym(p,Beff_val,Delta_val),p).subs(p,P_val)

    # Calculating the mass density, energy density, EOS slope (dE/dP) and speed of sound c_s at NS center
    Pc = p_0 # pressure at QS center
    E_c = EOS_formula(Pc) # energy density at QS center
    dE_dP = EOS_formula_slope(Pc) # EOS slope at QS center
  
     
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
def tov_sol_worker(task_id,EOS_name,Beff_value,Delta_value,ms_value,progress_queue):

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
        file.write(f"P_out, P_c, E_c, dE_dP, M, R, B_eff={Beff_value:.1f},Delta={Delta_value:.1f}\n")

    for i in range(0,total_values):
        P_0 = P_in_center_total[i]

        # Send progress update to the queue (task_id, current number of used initial values, total number of initial values)
        progress_queue.put((task_id,EOS_name,i + 1, total_values,P_0))
        
        # Calling the tov_sol() function
        tov_sol(P_0,R_step,Beff_value,Delta_value,EOS_name)

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

def task_giver(models_selected,Beff_selected_group,Delta_selected_group,ms_selected_group,progress_queue):
    n = len(models_selected)
        
    # Appending the tasks to be parallelized
    tasks = []
    for i in range(0,n):
        tasks.append(pool.apply_async(tov_sol_worker, args=(i,models_selected[i],Beff_selected_group[i],Delta_selected_group[i],ms_selected_group[i],progress_queue))) 


        
    

if __name__ == "__main__":
    # Printing the solution process info
    print("\n>SOLVING THE TOV EQUATIONS SYSTEM FOR A (CFL MATTER) QUARK STAR \n")
    print('===============================================================================')
    print('>>Determination of the range for the B_eff and Δ parameters:')
    print('-------------------------------------------------------------------------------')
    
    print("Minimum value of B_eff parameter in our study: 60.00 [MeV*fm^-3]")
    # Ask the user to give the maximum value for the Beff parameter
    Beff_max_val = int(input("Give the maximum value of the B_eff parameter in [MeV*fm^-3]:  "))
    while Beff_max_val<=57:
        Beff_max_val = int(input("The value must be greater than 57 [MeV*fm^-3]. Try again:  "))

    # Ask the user to give the step for the scan of the Beff area
    Beff_step_val = int(input("Give the step for the scan of the Beff area in [MeV*fm^-3]:  "))
    while Beff_step_val<=0:
        Beff_step_val = int(input("The value must be positive. Try again:  "))
    print('-------------------------------------------------------------------------------')
    
    print("Minimum value of Δ parameter in our study: %.2f [MeV]"%Delta_min)
    # Ask the user to give the maximum value for the Δ parameter
    Delta_max_val = int(input("Give the maximum value of the Δ parameter in [MeV]:  "))
    while Delta_max_val<=50:
        Delta_max_val = int(input("The value must be greater than 50 [MeV]. Try again:  "))

    # Ask the user to give the step for the scan of the Δ area
    Delta_step_val = int(input("Give the step for the scan of the Δ area in [MeV]:  "))
    while Beff_step_val<=0:
        Beff_step_val = int(input("The value must be positive. Try again:  "))         
    print('-------------------------------------------------------------------------------')

    valid_combos_results = valid_cfl_models(Beff_max_val,Beff_step_val,Delta_max_val,Delta_step_val)
    print("Total valid combinations:  %d"%len(valid_combos_results[0]))
    print('-------------------------------------------------------------------------------')

    # Ask the user to give the number of valid Beff and Δ combinations per group
    number_per_group = int(input("Give the number of valid Beff and Δ combinations per group:  "))
    while number_per_group<=0:
        number_per_group = int(input("The number must be positive. Try again:  "))
    print('-------------------------------------------------------------------------------')
    
    print(">>Groups of models:")
    print('-------------------------------------------------------------------------------\n')
    # Getting and printing the groups of Γ combinations
    models_names_groups,valid_Beff_groups,valid_Delta_groups,valid_ms_groups = Beff_Delta_groups_maker(valid_combos_results,number_per_group)
    l = len(models_names_groups)
    
    print('-------------------------------------------------------------------------------')
    group_choosing = int(input("Choose the group to be solved:  "))
    while group_choosing<1 or group_choosing>l:
        group_choosing = int(input("Group not found. Choose another group to be solved:  "))
    print('-------------------------------------------------------------------------------') 
    models_names_group_select = models_names_groups[group_choosing-1]
    valid_Beff_group_select = valid_Beff_groups[group_choosing-1]
    valid_Delta_group_select = valid_Delta_groups[group_choosing-1]
    valid_ms_group_select = valid_ms_groups[group_choosing-1]
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
        tasks = task_giver(models_names_group_select,valid_Beff_group_select,valid_Delta_group_select,valid_ms_group_select,progress_queue)

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
