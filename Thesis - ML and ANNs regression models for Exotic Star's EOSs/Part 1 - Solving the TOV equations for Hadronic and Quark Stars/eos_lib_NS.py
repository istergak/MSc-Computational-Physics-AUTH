# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models 

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py1
# Name: eos_lib_NS.py

# Description: 
# -> Defining different equations of states (EoSs) for a Neutron Star
# -> Storaging these equations in lists

# Abbrevations:
# NS -> Neutron Star


# Importing useful modules
import sympy as smp
import numpy as np

# Numerical definition of the EoSs (CORE)
# The following 21 models are being defined:

# APR-1
# BGP
# BL-1, BL-2
# DH
# HHJ-1, HHJ-2
# HLPS-2, HLPS-3
# MDI-1, MDI-2, MDI-3, MDI-4, 
# NLD
# PS
# SCVBB
# Ska
# SkI4
# W
# WFF-1, WFF-2

def APR_1(p):
    return 0.000719964*pow(p,1.85898)+108.975*pow(p,0.340074)
def BGP(p):
    return 0.0112475*pow(p,1.59689)+102.302*pow(p,0.335526)
def BL_1(p):
    return 0.488686*pow(p,1.01457)+102.26*pow(p,0.355095)
def BL_2(p):
    return 1.34241*pow(p,0.910079)+100.756*pow(p,0.354129)
def DH(p):
    return 39.5021*pow(p,0.541485)+96.0528*pow(p,0.00401285)
def HHJ_1(p):
    return 1.78429*pow(p,0.93761)+106.93652*pow(p,0.31715)
def HHJ_2(p):
    return 1.18961*pow(p,0.96539)+108.40302*pow(p,0.31264)                    
def HLPS_2(p):
    return 161.553+172.858*(1-np.exp(-p/22.8644))+2777.75*(1-np.exp(-p/1909.97))
def HLPS_3(p):
    return 81.5682+131.811*(1-np.exp(-p/4.41577))+924.143*(1-np.exp(-p/523.736))
def MDI_1(p):
    return 4.1844*pow(p,0.81449) + 95.00135*pow(p,0.31736)
def MDI_2(p):
    return 5.97365*pow(p,0.77374) + 89.24*pow(p,0.30993)
def MDI_3(p):
    return 15.55*pow(p,0.666)+76.71*pow(p,0.247)
def MDI_4(p):
    return 25.99587*pow(p,0.61209)+65.62193*pow(p,0.15512)
def NLD(p):
    return 119.05736+304.80445*(1-np.exp(-p/48.61465))+33722.34448*(1-np.exp(-p/17499.47411))
def PS(p):
    return 1.69483+9805.95*(1-np.exp(-0.000193624*p))+212.072*(1-np.exp(-0.401508*p))
def SCVBB(p):
    return 0.371414*pow(p,1.08004)+109.258*pow(p,0.351019)    
def Ska(p):
    return 0.53928*pow(p,1.01394)+94.31452*pow(p,0.35135)
def SkI4(p):
    return 4.75668*pow(p,0.76537)+105.722*pow(p,0.2745)
def W(p):
    return 0.261822*pow(p,1.16851)+92.4893*pow(p,0.307728)
def WFF_1(p):
    return 0.00127717*pow(p,1.69617)+135.233*pow(p,0.331471)
def WFF_2(p):
    return 0.00244523*pow(p,1.62692)+122.076*pow(p,0.340401)



# Symbolic definition of the EoSs (CORE)
# The same 21 models are being defined:

# APR-1
# BGP
# BL-1, BL-2
# DH
# HHJ-1, HHJ-2
# HLPS-2, HLPS-3
# MDI-1, MDI-2, MDI-3, MDI-4, 
# NLD
# PS
# SCVBB
# Ska
# SkI4
# W
# WFF-1, WFF-2

def APR_1_sym(p):
    return 0.000719964*pow(p,1.85898)+108.975*pow(p,0.340074)
def BGP_sym(p):
    return 0.0112475*pow(p,1.59689)+102.302*pow(p,0.335526)
def BL_1_sym(p):
    return 0.488686*pow(p,1.01457)+102.26*pow(p,0.355095)
def BL_2_sym(p):
    return 1.34241*pow(p,0.910079)+100.756*pow(p,0.354129)
def DH_sym(p):
    return 39.5021*pow(p,0.541485)+96.0528*pow(p,0.00401285)
def HHJ_1_sym(p):
    return 1.78429*pow(p,0.93761)+106.93652*pow(p,0.31715)
def HHJ_2_sym(p):
    return 1.18961*pow(p,0.96539)+108.40302*pow(p,0.31264)                    
def HLPS_2_sym(p):
    return 161.553+172.858*(1-smp.exp(-p/22.8644))+2777.75*(1-smp.exp(-p/1909.97))
def HLPS_3_sym(p):
    return 81.5682+131.811*(1-smp.exp(-p/4.41577))+924.143*(1-smp.exp(-p/523.736))
def MDI_1_sym(p):
    return 4.1844*pow(p,0.81449) + 95.00135*pow(p,0.31736)
def MDI_2_sym(p):
    return 5.97365*pow(p,0.77374) + 89.24*pow(p,0.30993)
def MDI_3_sym(p):
    return 15.55*pow(p,0.666)+76.71*pow(p,0.247)
def MDI_4_sym(p):
    return 25.99587*pow(p,0.61209)+65.62193*pow(p,0.15512)
def NLD_sym(p):
    return 119.05736+304.80445*(1-smp.exp(-p/48.61465))+33722.34448*(1-smp.exp(-p/17499.47411))
def PS_sym(p):
    return 1.69483+9805.95*(1-smp.exp(-0.000193624*p))+212.072*(1-smp.exp(-0.401508*p))
def SCVBB_sym(p):
    return 0.371414*pow(p,1.08004)+109.258*pow(p,0.351019)    
def Ska_sym(p):
    return 0.53928*pow(p,1.01394)+94.31452*pow(p,0.35135)
def SkI4_sym(p):
    return 4.75668*pow(p,0.76537)+105.722*pow(p,0.2745)
def W_sym(p):
    return 0.261822*pow(p,1.16851)+92.4893*pow(p,0.307728)
def WFF_1_sym(p):
    return 0.00127717*pow(p,1.69617)+135.233*pow(p,0.331471)
def WFF_2_sym(p):
    return 0.00244523*pow(p,1.62692)+122.076*pow(p,0.340401)



# Definition of the EoSs (CRUST) and their slopes
# The following EoSs for the 4 layers of the NS
# OUTER-CRUST are being defined:

# eos_crust1 -> EoS for pressure P : 9.34375*10^-5 < P < 0.184 [MeV/fm^3] 
# (for the PS EoS the upper bound, i.e. the crust/core bound, is 0.696 [MeV/fm^3])

# eos_crust2 -> EoS for pressure P : 4.1725*10^-8 < P < 9.34375*10^-5 [MeV/fm^3]

# eos_crust3 -> EoS for pressure P : 1.44875*10^-11 < P < 4.1725*10^-8 [MeV/fm^3]

# eos_crust4 -> EoS for pressure P : P < 1.44875*10^-11 [MeV/fm^3]


def eos_crust1(p):
    return 0.00873 + 103.17338*(1-np.exp(-p/0.38527))+7.34979*(1-np.exp(-p/0.01211))
def eos_crust2(p):
    return 0.00015 + 0.00203*(1-np.exp(-p*344827.5))+0.10851*(1-np.exp(-p*7692.3076))
def eos_crust3(p):
    return 0.0000051*(1-np.exp(-p*0.2373*1e10))+0.00014*(1-np.exp(-p*0.4020*1e8))
def eos_crust4(p):
    c0 = 31.93753
    c1 = 10.82611*np.log10(p)
    c2 = 1.29312 * (np.log10(p)**2)
    c3 = 0.08014*(np.log10(p)**3)
    c4 = 0.00242*(np.log10(p)**4)
    c5 = 0.000028*(np.log10(p)**5)
    return 10**(c0 + c1 + c2 + c3 + c4 + c5)


def slope_eos_crust1(p):
    return 267.8274087168961 * np.exp(-p / 0.38527) + 606.8353790330587 * np.exp(-p / 0.01211)
def slope_eos_crust2(p):
    return 700.994825 * np.exp(-p * 344827.5) + 834.262074 * np.exp(-p * 7692.3076)
def slope_eos_crust3(p):
    return 1.20923e6 * np.exp(-p * 0.2373 * 1e10) + 5.628e6 * np.exp(-p * 0.4020 * 1e8)
def slope_eos_crust4(p):
    log10_p = np.log10(p)
    
    # Calculate the original exponent g(p) = c0(r) + c1(r) +c2(r) + c3(r) + c4(r) + c5(r)
    c0 = 31.93753
    c1 = 10.82611*log10_p
    c2 = 1.29312 * (log10_p**2)
    c3 = 0.08014*(log10_p**3)
    c4 = 0.00242*(log10_p**4)
    c5 = 0.000028*(log10_p**5)
    g_p = c0+c1+c2+c3+c4+c5
    
    
    # Calculate the derivative of g'(r) = c0'(r) + c1'(r) +c2'(r) + c3'(r) + c4'(r) + c5'(r)
    d1 = 10.82611
    d2 = 2*1.29312*log10_p
    d3 = 3*0.08014*log10_p**2
    d4 = 4*0.00242*log10_p**3
    d5 = 5*0.000028*log10_p**4
    dg_dp = (d1+d2+d2+d3+d4+d5)/(p*np.log(10))
    
    # Return the full derivative
    return 10**g_p*np.log(10)*dg_dp    


# Defining a list to store the EoSs of the CORE of the Neutron Star

eos_list_core = [["APR-1",APR_1,APR_1_sym],
            ["BGP",BGP,BGP_sym],
            ["BL-1",BL_1,BL_1_sym],
            ["BL-2",BL_2,BL_2_sym],
            ["DH",DH,DH_sym],
            ["HHJ-1",HHJ_1,HHJ_1_sym],
            ["HHJ-2",HHJ_2,HHJ_2_sym],
            ["HLPS-2",HLPS_2,HLPS_2_sym],
            ["HLPS-3",HLPS_3,HLPS_3_sym],
            ["MDI-1",MDI_1,MDI_1_sym],
            ["MDI-2",MDI_2,MDI_2_sym],
            ["MDI-3",MDI_3,MDI_3_sym],
            ["MDI-4",MDI_4,MDI_4_sym],
            ["NLD",NLD,NLD_sym],
            ["PS",PS,PS_sym],
            ["SCVBB",SCVBB,SCVBB_sym],
            ["Ska",Ska,Ska_sym],
            ["SkI4",SkI4,SkI4_sym],
            ["W",W,W_sym],
            ["WFF-1",WFF_1,WFF_1_sym],
            ["WFF-2",WFF_2,WFF_2_sym]
            ]


# Defining a list to store the EoSs of the CRUST of the Neutron Star

eos_list_crust = [["Crust-1",eos_crust1,slope_eos_crust1],
                  ["Crust-2",eos_crust2,slope_eos_crust2],
                  ["Crust-3",eos_crust3,slope_eos_crust3],
                  ["Crust-4",eos_crust4,slope_eos_crust4]]
