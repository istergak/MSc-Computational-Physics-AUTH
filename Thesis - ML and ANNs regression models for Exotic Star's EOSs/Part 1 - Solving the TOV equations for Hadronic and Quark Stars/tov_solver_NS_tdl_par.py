# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py2b (Version 2 - Tidal Deformability Calculation Included)
# Name: tov_solver_NS_tdl_par.py

# Description: 
# -> Parallel solving of the TOV equations for a selected number of the Neutron Stars' EoSs included in the 'eos_lib_NS.py' script
# -> Storaging the solutions in .csv files

# Abbrevations:
# NS -> Neutron Star


# Importing useful modules
import numpy as np
import sympy as smp
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys as sys
import os
from prettytable import PrettyTable
import eos_lib_NS
import multiprocessing
import time

# Defining useful symbols
pp = smp.symbols("pp",real=True)

# Defining a function that returns the system of the TOV equations
def tov_eq(r,z,EOS,dEOS,p_bound_crust):
    # Changing the core EoS to a crust EoS if the crust of the NS has been reached
    if 9.34375*10**-5 < z[0] < p_bound_crust:
        EOS = eos_lib_NS.eos_crust1
        dEOS = eos_lib_NS.deos_crust1_num
    elif 4.1725*10**-8 < z[0] < 9.34375*10**-5:
        EOS = eos_lib_NS.eos_crust2
        dEOS = eos_lib_NS.deos_crust2_num
    elif 1.44875*10**-11 < z[0] < 4.1725*10**-8:
        EOS = eos_lib_NS.eos_crust3
        dEOS = eos_lib_NS.deos_crust3_num
    elif z[0] < 1.44875*10**-11:
        EOS = eos_lib_NS.eos_crust4
        dEOS = eos_lib_NS.deos_crust3_num   

    # Appending values
    P = z[0] # value of the pressure at radius r
    M = z[1] # value of the mass at radius r
    y = z[2] # value of y at radius r
    e = EOS(P) # value of the energy's density at radius r
    dedp = dEOS(P) # value of the energy's density first derivative with respect to pressure at radius r

    # Necessary functions for tidal deformability

    # F(r) function
    F = (1-1.474*11.2e-6*(r**2)*(e-P))*((1-2.948*M/r)**(-1))
    # r^2Q(r) function
    r2Q = 1.474*11.2e-6*(r**2)*(5*e+9*P+(e+P)/(1/dedp))*((1-2.948*M/r)**(-1))-6*((1-2.948*M/r)**(-1))-4*(1.474**2*M**2)/(r**2)*((1+11.2e-6*r**3*P/M)**2)*((1-2.948*M/r)**(-2))

    # Defining the system of TOV equations (including a third equation for tidal)
    dP_dr = -1.474*(e*M/r**2)*(1+P/e)*(1+11.2e-6*r**3*P/M)/(1-2.948*M/r)
    dM_dr = 11.2e-6*r**2*e
    dy_dr = (-y*y-y*F-r2Q)/r
    return [dP_dr,dM_dr,dy_dr] 


# Defining a function that solves the system of TOV equations
# for a given EOS
def tov_sol(p_0,r_step,EOS_name,EOS_func,dEOS_func):
    # Pressure and mass initial values - initial conditions
    Pc = p_0
    E_c = EOS_func(p_0)
    P_0 = Pc # initial pressure = pressure in center of NS
    M_0 = 10**-12 # mass in center of the NS
    y_0 = 2 # y value at center of the NS
    initial_values = [P_0,M_0,y_0]

    # # Defining the slope of the selected EOS as a function
    # p = smp.symbols("p")
    # def EOS_slope(p_value):
    #     return smp.diff(EOS_func_sym(p),p).subs(p,p_value)
        
    # Calculate the slope, associated with the speed of sound at the center of the NS
    dE_dP = dEOS_func(Pc)
        
        
    # Bounds of NS radius r interval for the 1st step 
    # in the solving process
    r_min: float = 1e-9
    r_max: float = 1e-2

    # Useful storage lists/variables
    P_out = None
    M_vals = []
    R = None
    yR = None

    # Crust-core pressure bound
    p_bound_crust = 0.184
    if EOS_name == 'PS':
        p_bound_crust = 0.696    
        
    # Solving the TOV equations for the current pressure in center
    while P_0>10**-12: 

        # Solving the TOV equations system in the [r_min,r_max] radius interval  
        solution = solve_ivp(tov_eq,(r_min,r_max),initial_values,method="RK45",args=(EOS_func,dEOS_func,p_bound_crust),atol=10**-12,rtol=10**-8)
        # Updating the initial values to be used in the next [r_min,r_max] interval
        initial_values[0] = solution.y[0][-1] # pressure
        initial_values[1] = solution.y[1][-1] # mass
        initial_values[2] = solution.y[2][-1] # y

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
            R = solution.t[-1]
        if solution.y[2][-1]!=np.nan:
            yR = solution.y[2][-1]    

    # if P_vals[-1]<0:
    #     idx = np.argwhere(P_vals<0)[0 ,0]
    #     P_vals = np.delete (P_vals ,np.s_[idx::],0)
    #     M_vals = np.delete (M_vals , np.s_[ idx::],0)
    #     R_vals = np.delete (R_vals, np.s_[idx::],0)

    # Calculating tidal parameters
    C = max(M_vals)/R # compactness
    beta = 1.474*C
    k2_term1 = (8/5)*beta**5*(1-2*beta )**2*(2-yR+2*beta*(yR-1))
    k2_term2 = 2*beta*(6-3*yR+3*beta*(5*yR-8))
    k2_term3 = 4*beta**3*(13 -11*yR+ beta*(3*yR-2)+2*beta**2*(1+yR))
    k2_term4 = 3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
    k2 = k2_term1/(k2_term2+k2_term3+k2_term4)
    Lambda = 2/3*k2*(R**5/max(M_vals)**5)

    # Store the solution's data in the .csv file 
    filename = f'{EOS_name}_tdl_sol.csv'
    with open(filename,"a+") as file:
        file.write(f"{P_out}, {Pc}, {E_c},{dE_dP},{max(M_vals)}, {R}, {yR}, {k2}, {Lambda}\n")

# Defining the worker function for the parallel solving 
def tov_sol_worker(task_id,EOS_name,EOS_function,EOS_function_sym,progress_queue):
    # Calculating the derivative of EOS
    dEOS_function = smp.lambdify(pp,EOS_function_sym(pp).diff(),"numpy")

    # Pressure ranges
    P_in_center_1: np.ndarray = np.arange(1.8,5,0.1) # lower pressure range
    P_in_center_2: np.ndarray = np.arange(5,3001,1) # higher pressure range
    P_in_center_total: np.ndarray = np.concatenate((P_in_center_1,P_in_center_2),axis=None)

    # Radius step in the scan of the NS
    R_step = 0.01

    # Total number of initial values
    total_values = len(P_in_center_total)

    filename = f'{EOS_name}_tdl_sol.csv'
    with open(filename,"w") as file:
        file.write("P_out, P_c, E, dE/dP, M, R, yR, k2, L\n")

    for i in range(total_values):
        P_0 = P_in_center_total[i]

        # Send progress update to the queue (task_id, current number of used initial values, total number of initial values)
        progress_queue.put((task_id,EOS_name,i + 1, total_values,P_0))

        
        # Calling the tov_sol() function
        tov_sol(P_0,R_step,EOS_name,EOS_function,dEOS_function)

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

def task_giver(EOS_names,EOS_functions,EOS_functions_sym,progress_queue):
    n = len(EOS_names)
        
    # Appending the tasks to be parallelized
    tasks = []
    for i in range(0,n):
        tasks.append(pool.apply_async(tov_sol_worker, args=(i,EOS_names[i], EOS_functions[i], EOS_functions_sym[i],progress_queue)))  
    

if __name__ == "__main__":
    # Printing the solution process info
    print("\n>SOLVING THE TOV EQUATIONS SYSTEM FOR A NEUTRON STAR\n")
    print('===============================================================================')
    
    # Asking the user for the number of EOSs to be solved
    num_tasks = int(input("Give the number of EOS for parallel solving: "))

    print('-------------------------------------------------------------------------------')
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

    # Showing the offered EOSs
    print(EOS_list)
    print('-------------------------------------------------------------------------------')

    # Asking the user to select the EOSs to be solved
    EOS_names_select = []
    EOS_formulas_select = []
    EOS_formulas_sym_select = []
    for i in range(0,num_tasks):
        EOS_name = input(f"Give the EOS {i+1} name:  ")
        while EOS_names.count(EOS_name)==0:
            EOS_name = input(f"EOS not found, give another name for EOS {i+1}:  ")
        EOS = EOS_formulas[EOS_names.index(EOS_name)] # numerical formula of the selected EOS
        EOS_sym = EOS_formulas_sym[EOS_names.index(EOS_name)] # symbolic formula of the selected EOS

        EOS_names_select.append(EOS_name)
        EOS_formulas_select.append(EOS)
        EOS_formulas_sym_select.append(EOS_sym)
    print('-------------------------------------------------------------------------------')    

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
        tasks = task_giver(EOS_names_select,EOS_formulas_select,EOS_formulas_sym_select,progress_queue)

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