# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs and classification of Exotic Stars using ML and ANNs models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script 2c
# Name: tov_solver_polyNS_par.py

# Description: 
# -> Parallel solving of the TOV equations for polytropic constructed EoSs of a Neutron Star
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


# Definition of useful constants
ρ_sat = 2.7*10**14 # nuclear saturation density [g/cm^3]
c = 3e8 # vaccuum speed of light in m/s
MeV_to_J = 1.60218e-13 # 1MeV value in Joules

# Define a function to return Γ codec value
def Γ_codec(Γ,Γ_choices):
    n = len(Γ_choices)
    for i in range(0,n):
        codec = chr(65+i)
        if Γ==Γ_choices[i]:
            break
    return codec

# Define a function that decodes the selected combination of Γ
def Γ_decode(Γ_select_coded,Γ_choices):
    n = len(Γ_select_coded)
    m = len(Γ_choices)
    Γ_select_decoded = []
    for i in range(0,n):
        Γ_result = "Codec letter not found"
        for j in range(0,m):
            askii_codec = chr(65+j)
            if Γ_select_coded[i]==askii_codec:
                Γ_result = Γ_choices[j]
                break
        Γ_select_decoded.append(Γ_result)

    return Γ_select_decoded        


# Define a function that returns all the possible selections of Γ
# as groups, depending on the number of choices, the number of mass density
# segments and the number of combinations per group
def Γ_groups_maker(Γ_choices_codec,num_segments,num_per_group):
    
    # Count the given number of choices for Γ
    n = len(Γ_choices_codec)
    
    # Find all the possible combinations of Γ
    total_combo = n**num_segments
    i = 1
    Γ_selections = []
    Γ_choose = random.choices(Γ_choices_codec,k=num_segments)
    Γ_selection = ""
    for j in range(0,len(Γ_choose)):
        Γ_selection = Γ_selection + Γ_choose[j]
    Γ_selections.append(Γ_selection)
    i=i+1
    while i<=total_combo:
        Γ_choose = random.choices(Γ_choices_codec,k=num_segments)
        Γ_selection = ""
        for j in range(0,len(Γ_choose)):
            Γ_selection = Γ_selection + Γ_choose[j]
        if Γ_selections.count(Γ_selection)!=0:
            continue
        else:
            Γ_selections.append(Γ_selection)
            i=i+1

    # Sorting all the combinations in ascending order
    Γ_selections_sort = sorted(Γ_selections)
    
    # Forming and printing the groups of Γ combinations
    mod = total_combo%num_per_group
    upper_limit = total_combo-mod
    Γ_groups = []
    for i in range(0,upper_limit,num_per_group):
        Γ_groups.append(Γ_selections_sort[i:i+num_per_group])
    if mod!=0:
        Γ_groups.append(Γ_selections_sort[upper_limit:total_combo])
    m = len(Γ_groups)
    for j in range(0,m):
        print(f"Γ Group {j+1}:")
        print(Γ_groups[j],"\n")
    
    return Γ_groups

# Converter of mass density from units g/cm^3 to MeV/fm^3 units
def conv_to_MeV(value):
    # Multiply by 10^3 to convert to SI units (kg/m^3)
    result = value*10**3
    # Multiply by c^2 to convert to J/m^3
    result = result*c**2
    # Divide by the MeV_to_J constant to convert to MeV/m^3
    result = result/MeV_to_J
    # Convert m to fm, for the units to be MeV/fm^3
    result = result*10**(-45)

    return result

# Defining the polytrope relation between pressure P and mass density ρ as a function
def P_poly(ρ,K,Γ):
    return K*ρ**Γ

# Defining the inverse polytrope relation between pressure P and mass density ρ as a function
def ρ_poly(P,K,Γ):
    return pow(P/K,1/Γ)

# Defining the K constant parameter as a function of pressure P and and mass density
def K_calc(P,ρ,Γ):
    return P/ρ**Γ

# Defining the total inverse polytrope relation between pressure P and mass density ρ (all presure segments are included)
def ρ_poly_piecewise(P,Pi_vals,Ki_vals,Γi_vals):
    points = len(Pi_vals)
    n = points -1 # number of pressure (or mass density) segments

    conditions = []
    functions = []

    for i in range(0,n):
        if i==n-1:
            conditions.append((P>=Pi_vals[i])*(P<=Pi_vals[i+1]))
        else:
            conditions.append((P>=Pi_vals[i])*(P<Pi_vals[i+1]))   
        functions.append(ρ_poly(P,Ki_vals[i],Γi_vals[i]))

    return np.piecewise(P,conditions,functions)

# Defining the polytropic behavior of EOS for a single segment in high mass densities and high pressures
def EOS_polytrope(P,K_i,Γ_i,ρ_im1,P_im1,E_im1):
    if Γ_i!=1:
        result = (E_im1/ρ_im1 - P_im1/(ρ_im1*(Γ_i-1)))*pow(P/K_i,1/Γ_i) + P/(Γ_i-1)
    else:
        result = E_im1/ρ_im1*P/K_i + np.log(1/ρ_im1)*P - P*np.log(K_i/P)

    return result

# Defining the total polytropic behavior of EOS in high mass densities and high pressures (all presure segments are included)
def EOS_polytrope_piecewise(P,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals):
    points = len(ρ_bounds)
    n = points-1 # number of pressure (or mass density) segments

    conditions = []
    functions = []

    for i in range(0,n):
        if i==n-1:
            conditions.append((P>=Pi_vals[i])*(P<=Pi_vals[i+1]))
        else:
            conditions.append((P>=Pi_vals[i])*(P<Pi_vals[i+1]))
        functions.append(EOS_polytrope(P,Ki_vals[i],Γi_vals[i],ρ_bounds[i],Pi_vals[i],Ei_vals[i]))

    return np.piecewise(P,conditions,functions)

# Defining the slope of the polytropic EOS for a single segment in high mass densities and high pressures
def EOS_slope_polytrope(P,K_i,Γ_i,ρ_im1,P_im1,E_im1):
    if Γ_i!=1:
        result = (1/(K_i*Γ_i))*(E_im1/ρ_im1 - P_im1/(ρ_im1*(Γ_i-1)))*pow(P/K_i,1/Γ_i-1) + 1/(Γ_i-1)
    else:
        result =  E_im1/(ρ_im1*K_i) + np.log(1/ρ_im1) + 1 - np.log(K_i/P)

    return result

# Defining the slope of the total polytropic EOS in high mass densities and high pressures (all presure segments are included)
def EOS_slope_polytrope_piecewise(P,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals):
    points = len(ρ_bounds)
    n = points -1 # number of pressure (or mass density) segments

    conditions = []
    functions = []

    for i in range(0,n):
        conditions.append((P>Pi_vals[i])*(P<Pi_vals[i+1]))
        functions.append(EOS_slope_polytrope(P,Ki_vals[i],Γi_vals[i],ρ_bounds[i],Pi_vals[i],Ei_vals[i]))

    for i in range(0,points):
        conditions.append((P==Pi_vals[i]))
        functions.append(np.nan)

    return np.piecewise(P,conditions,functions)           

# Defining the crust EOS as a piecewise function
def EOS_crust_piecewise(P):
    conditions = []
    functions = []

    # Append the condition and function of the 1st layer of the NS crust
    conditions.append((P>9.34375*10**-5)*(P<=0.184))
    functions.append(eos_lib_NS.eos_crust1(P))

    # Append the condition and function of the 2nd layer of the NS crust
    conditions.append((P>4.1725*10**-8)*(P<=9.34375*10**-5))
    functions.append(eos_lib_NS.eos_crust2(P))

    # Append the condition and function of the 3rd layer of the NS crust
    conditions.append((P>1.44875*10**-11)(P<= 4.1725*10**-8))
    functions.append(eos_lib_NS.eos_crust3(P))

    # Append the condition and function of the 4th layer of the NS crust
    conditions.append((P <= 1.44875*10**-11))
    functions.append(eos_lib_NS.eos_crust4(P))

    return np.piecewise(P,conditions,functions)


# Defining the total EOS of the NS as a piecewise function
def total_EOS(P,main_EOS,P_poly_change,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals):
    conditions = []
    functions = []

    # Append the polytrope EOS's condition and function
    conditions.append((P>=P_poly_change))
    functions.append(EOS_polytrope_piecewise(P,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals))

    # Append the main EOS's condition and function
    conditions.append((P>0.184)*(P<P_poly_change))
    functions.append(main_EOS(P))

    # Append the crust EOS's condition and function
    conditions.append((P<=0.184))
    functions.append(EOS_crust_piecewise(P))

    return np.piecewise(P,conditions,functions)

# Define a function to return the values of K_i, P_i and E_i given a
# starting pressure P_0, the bounds of the segments of 
# mass density ρ and the selected values of Γ for the segments of ρ
def poly_param_calc(P_0,E_0,ρ_bounds,Γ_selections):
    # Definition of storage lists
    results = []
    Pi_vals = []
    Pi_vals.append(P_0)
    Ei_vals = []
    Ei_vals.append(E_0)
    Ki_vals = []
    
    # Number of segments
    n = len(Γ_selections)

    # For loop to calculate the values of Pi,Γi and Ki
    for i in range(0,n):
        Γi = Γ_selections[i]

        Ki = K_calc(Pi_vals[i],ρ_bounds[i],Γi)
        Ki_vals.append(Ki)

        Pi = P_poly(ρ_bounds[i+1],Ki,Γi)
        Pi_vals.append(Pi)

        Ei = EOS_polytrope(Pi_vals[i+1],Ki,Γi,ρ_bounds[i],Pi_vals[i],Ei_vals[i])
        Ei_vals.append(Ei)

    results = [Pi_vals,Ei_vals,Ki_vals]
    return results


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
def tov_sol(p_0,ρ_sat,P_sat,r_step,total_EOS_name,main_EOS_func,main_EOS_func_sym,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals):
    # Defining the polytropic EOS aa a function of pressure
    def polytropic_EOS(P):
        return EOS_polytrope_piecewise(P,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals)
    
    # Defining the slope of the main EOS
    p = smp.symbols("p")
    def main_EOS_slope(p_value):
        return smp.diff(main_EOS_func_sym(p),p).subs(p,p_value)

    # Calculating the mass density, energy density and EOS slope (dE/dP) at NS center
    Pc = p_0 # pressure at NS center
    E_c = None # energy density at NS center
    rho_c = None # mass density ρ*c^2 at NS center
    rho_frac = None # mass density to nuclear saturation density
    dE_dP = None # EOS slope at NS center
    
    if Pc>=P_sat: # polytropic core area
        E_c = polytropic_EOS(Pc) 
        rho_c = ρ_poly_piecewise(Pc,Pi_vals,Ki_vals,Γi_vals)
        rho_frac = rho_c/ρ_sat 
        dE_dP = EOS_slope_polytrope_piecewise(Pc,ρ_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals)
    elif Pc<P_sat: # main core area
        E_c = main_EOS_func(Pc) 
        rho_c = np.nan
        rho_frac = "-"
        dE_dP = main_EOS_slope(Pc)
         
    # Pressure and mass initial values (initial conditions)       
    P_0 = Pc # initial pressure = pressure in center of NS
    M_0 = 10**-12 # mass in center of the NS
    initial_values = [P_0,M_0]
            
    # Bounds of NS radius r interval for the 1st step 
    # in the solving process
    r_min: float = 1e-9
    r_max: float = 1e-2

    # Useful storage lists
    P_final = None
    M_vals = []
    R_final = None
    
    # Solving the TOV equations for the polytropic core section of pressure
    n = len(Pi_vals)
    k = n-1
    for i in range(k,0,-1):
        def current_poly_EOS(P):
            return EOS_polytrope(P,Ki_vals[i-1],Γi_vals[i-1],ρ_bounds[i-1],Pi_vals[i-1],Ei_vals[i-1])
        EOS_func = current_poly_EOS
        # Solving for the current pressure section (Pi_vals[i-1],Pi_vals[i])
        while Pi_vals[i-1]<P_0<Pi_vals[i]:
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
            P_final = solution.y[0][-1]
            if solution.y[1][-1]!=np.nan:
                M_vals.append(solution.y[1][-1])
            if solution.t[-1]!=np.nan:
                R_final = solution.t[-1]
 
    # Solving the TOV equations for the main core and crust section
    EOS_func = main_EOS_func
    while P_0>10**-12:
        # Changing the core main EoS to the crust EoS if the crust of the NS star has been reached 
        # (pressure less than 0.184 MeV/fm^3)
        if 9.34375*10**-5 < P_0 < 0.184:
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

        # Appending values to the M_max, R_final, P_final variables
        P_final = solution.y[0][-1]
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
    filename = f'{total_EOS_name}_sol.csv'
    with open(filename,"a+") as file:
        file.write(f"{P_final:.3f}, {Pc:.3f}, {E_c:.3f}, {rho_frac}, {dE_dP:.3f}, {max(M_vals)}, {R_final}\n")

# Defining the worker function for the parallel solving 
def tov_sol_worker(task_id,Γ_selection_coded,Γ_choices,num_segments,main_EOS_name,main_EOS,main_EOS_sym,progress_queue):
    # Make the total EOS name
    total_EOS_name = f"{main_EOS_name}_{Γ_selection_coded}"

    # Getting the changing pressure from the main EOS to the polytropic behavior
    if main_EOS_name=="HLPS-2":
        P_sat = 1.722 # in MeV/fm^3
    elif main_EOS_name=="HLPS-3":
        P_sat = 2.816 # in MeV/fm^3

    # Decode the comdination of Γ to take the Γ_i values
    Γi_vals = Γ_decode(Γ_selection_coded,Γ_choices)    

    # Getting the total range of pressure for polytropic behavior
    ρ_sat_MeV = conv_to_MeV(ρ_sat)
    ρ_poly_low = ρ_sat_MeV # lower bound: 1*ρ_sat
    
    # soft polytropic behavior
    if Γi_vals[0]<=2:
        ρ_poly_high = 15*ρ_sat_MeV # higher bound 15*ρ_sat
        log_ρ0 = np.log(ρ_poly_low)
        log_ρn = np.log(7.5*ρ_sat_MeV)
        ρ_seg_bounds = np.exp(np.linspace(log_ρ0,log_ρn,num_segments+1))
        ρ_seg_bounds[-1] = ρ_poly_high

    # intermediate polytropic behavior
    elif 2<Γi_vals[0]<=3:
        ρ_poly_high = 12*ρ_sat_MeV # higher bound 12*ρ_sat
        log_ρ0 = np.log(ρ_poly_low)
        log_ρn = np.log(7.5*ρ_sat_MeV)
        ρ_seg_bounds = np.exp(np.linspace(log_ρ0,log_ρn,num_segments+1))
        ρ_seg_bounds[-1] = ρ_poly_high    

    # stiff polytropic behavior    
    else: 
        ρ_poly_high = 9*ρ_sat_MeV # higher bound 9*ρ_sat
        log_ρ0 = np.log(ρ_poly_low)
        log_ρn = np.log(7.5*ρ_sat_MeV)
        ρ_seg_bounds = np.exp(np.linspace(log_ρ0,log_ρn,num_segments+1))
        ρ_seg_bounds[-1] = ρ_poly_high 

       
    # Obtain the rest parameters for the polytrope EOS
    param_results = poly_param_calc(P_sat,main_EOS(P_sat),ρ_seg_bounds,Γi_vals)
    Pi_vals = param_results[0]
    Ei_vals = param_results[1]
    Ki_vals = param_results[2]     

    # Pressure ranges
    P_in_center_1: np.ndarray = np.arange(1.8,5,0.1) # lower pressure range

    # maximum initial pressure less than 2500 MeV/fm^3
    if Pi_vals[-1]<3000: 
        P_in_center_2: np.ndarray = np.arange(5,Pi_vals[-1],1) # higher pressure range
        P_in_center_total: np.ndarray = np.concatenate((P_in_center_1,P_in_center_2),axis=None)

    # every other case    
    else:
        P_in_center_2: np.ndarray = np.arange(5,3001,1) # higher pressure range
        P_in_center_total: np.ndarray = np.concatenate((P_in_center_1,P_in_center_2),axis=None)

    # Radius step in the scan of the NS
    R_step = 0.001 # km

    # Total number of initial values
    total_values = len(P_in_center_total)

    filename = f'{total_EOS_name}_sol.csv'
    with open(filename,"w") as file:
        file.write("P_out, P_c, E_c, rho/rho_sat, dE/dP, M, R\n")


    for i in range(0,total_values):
        P_0 = P_in_center_total[i]

        # Send progress update to the queue (task_id, current number of used initial values, total number of initial values)
        progress_queue.put((task_id,Γ_selection_coded,i + 1, total_values,P_0))
        
        # Calling the tov_sol() function
        tov_sol(P_0,ρ_sat_MeV,P_sat,R_step,total_EOS_name,main_EOS,main_EOS_sym,ρ_seg_bounds,Pi_vals,Ei_vals,Ki_vals,Γi_vals)

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

def task_giver(Γ_selected_group,Γ_choices,num_segments,main_EOS_name,main_EOS,main_EOS_sym,progress_queue):
    n = len(Γ_selected_group)
        
    # Appending the tasks to be parallelized
    tasks = []
    for i in range(0,n):
        tasks.append(pool.apply_async(tov_sol_worker, args=(i,Γ_selected_group[i],Γ_choices,num_segments,main_EOS_name,main_EOS,main_EOS_sym,progress_queue))) 


        
    

if __name__ == "__main__":
    # Printing the solution process info
    print("\n>SOLVING THE TOV EQUATIONS SYSTEM FOR A NEUTRON STAR (POLYTROPIC EOS APPROACH)\n")
    print('===============================================================================')

    print('-------------------------------------------------------------------------------')
    # Appending the EOSs info to lists
    main_EOS_list = PrettyTable()
    main_EOS_names = []
    main_EOS_formulas = []
    main_EOS_formulas_sym = []
    for EOS_info in eos_lib_NS.eos_list_core:
        main_EOS_names.append(EOS_info[0])
        main_EOS_formulas.append(EOS_info[1])
        main_EOS_formulas_sym.append(EOS_info[2])
    main_EOS_offered = ["HLPS-2","HLPS-3"]    
    main_EOS_list.add_column("Offered main EOSs models",main_EOS_offered)

    # Showing the offered EOSs
    print(main_EOS_list)
    print('-------------------------------------------------------------------------------')

    # Asking the user to select one of the given main EOSs to be solved
    main_EOS_name = input(f"Give the main EOS name:  ")
    while main_EOS_offered.count(main_EOS_name)==0:
        main_EOS_name = input(f"EOS not found, give another name for main EOS:  ")
    main_EOS = main_EOS_formulas[main_EOS_names.index(main_EOS_name)] # numerical formula of the selected main EOS
    main_EOS_sym = main_EOS_formulas_sym[main_EOS_names.index(main_EOS_name)] # symbolic formula of the selected main EOS
    print('-------------------------------------------------------------------------------')

    # Initialize a list for the choices of the Γ parameter
    Γ_choices = [1,2,3,4]

    # Τurn the choices of Γ into codec
    m = len(Γ_choices)
    Γ_choices_codec = []
    for i in range(0,m):
        Γ_choices_codec.append(Γ_codec(Γ_choices[i],Γ_choices))
    print("Choices of Γ parameter:")
    print(Γ_choices)
    print("Coded choices of Γ parameter:")
    print(Γ_choices_codec)
    print('-------------------------------------------------------------------------------')
    
    # Ask the user to give the number of mass density segments
    segments_amount = int(input("Give the number of the mass density segments:  "))
    while segments_amount<=0:
        segments_amount = int(input("The number must be positive. Try again:  "))

    print('-------------------------------------------------------------------------------')
    print("Total Γ combinations:  %d"%(m**segments_amount))
    print('-------------------------------------------------------------------------------')    
    
    # Ask the user to give the number of Γ combinations per group
    number_per_group = int(input("Give the number of Γ combinations per group:  "))
    while number_per_group<=0:
        number_per_group = int(input("The number must be positive. Try again:  "))

    
    print('-------------------------------------------------------------------------------')
    # Getting and printing the groups of Γ combinations
    Γ_groups = Γ_groups_maker(Γ_choices_codec,segments_amount,number_per_group)
    l = len(Γ_groups)
    
    print('-------------------------------------------------------------------------------')
    group_choosing = int(input("Choose the group of Γ combinations to be solved:  "))
    while group_choosing<1 or group_choosing>l:
        group_choosing = int(input("Group not found. Choose another group to be solved:  "))
    print('-------------------------------------------------------------------------------') 
    Γ_group_select = Γ_groups[group_choosing-1]
    num_tasks = len(Γ_group_select)
        

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
        tasks = task_giver(Γ_group_select,Γ_choices,segments_amount,main_EOS_name,main_EOS,main_EOS_sym,progress_queue)

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