# MSc Computational Physics AUTh
# Academic Year: 2024-2025
# Master's Thesis

# Thesis Title:  
# Reconstruction of the EoSs of Exotic Stars using ML and ANNs regression models

# Implemented by: Ioannis Stergakis
# AEM: 4439

# Python Script: Py4
# Name: eos_lib_QS.py

# Description: 
# -> Defining different equation of states (EoSs) for a Quark Star
# -> Storaging these equations in lists

# Abbrevations:
# QS -> Quark Star


# Importing useful modules
import numpy as np
import sympy as smp

# Defining useful constants
m_s = 95 # MeV, mass of Strange quark
m_n = 939.565 # MeV, mass of Neutron
hbarc = 197.327 # MeV*fm, hbar*c constant
Beff_min = 57 # MeV*fm^-3, minimum value of the B_eff constant of the MIT bag model


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


# Numerical definition of the CFL EoSs models
# The following 23 models are defined:

# Name   \\ B_eff [MeV*fm^-3] \\ Δ [MeV]
# CFL-1  \\   60              \\ 50 
# CFL-2  \\   60              \\ 100 
# CFL-3  \\   60              \\ 150 
# CFL-4  \\   70              \\ 50
# CFL-5  \\   70              \\ 100 
# CFL-6  \\   70              \\ 150
# CFL-7  \\   80              \\ 50
# CFL-8  \\   80              \\ 100
# CFL-9  \\   80              \\ 150
# CFL-10 \\   90              \\ 50
# CFL-11 \\   90              \\ 100
# CFL-12 \\   90              \\ 150
# CFL-13 \\   100             \\ 100 
# CFL-14 \\   100             \\ 150 
# CFL-15 \\   110             \\ 100
# CFL-16 \\   110             \\ 150
# CFL-17 \\   120             \\ 100
# CFL-18 \\   120             \\ 150
# CFL-19 \\   130             \\ 150
# CFL-20 \\   140             \\ 150
# CFL-21 \\   150             \\ 150
# CFL-22 \\   160             \\ 150
# CFL-23 \\   170             \\ 150


def CFL_1(p):
    return EOS_CFL_stable(p,60,50)
def CFL_2(p):
    return EOS_CFL_stable(p,60,100)
def CFL_3(p):
    return EOS_CFL_stable(p,60,150)
def CFL_4(p):
    return EOS_CFL_stable(p,70,50)
def CFL_5(p):
    return EOS_CFL_stable(p,70,100)
def CFL_6(p):
    return EOS_CFL_stable(p,70,150)
def CFL_7(p):
    return EOS_CFL_stable(p,80,50)
def CFL_8(p):
    return EOS_CFL_stable(p,80,100)
def CFL_9(p):
    return EOS_CFL_stable(p,80,150)
def CFL_10(p):
    return EOS_CFL_stable(p,90,50)
def CFL_11(p):
    return EOS_CFL_stable(p,90,100)
def CFL_12(p):
    return EOS_CFL_stable(p,90,150)
def CFL_13(p):
    return EOS_CFL_stable(p,100,100)
def CFL_14(p):
    return EOS_CFL_stable(p,100,150)
def CFL_15(p):
    return EOS_CFL_stable(p,110,100)
def CFL_16(p):
    return EOS_CFL_stable(p,110,150)
def CFL_17(p):
    return EOS_CFL_stable(p,120,100)
def CFL_18(p):
    return EOS_CFL_stable(p,120,150)
def CFL_19(p):
    return EOS_CFL_stable(p,130,150)
def CFL_20(p):
    return EOS_CFL_stable(p,140,150)
def CFL_21(p):
    return EOS_CFL_stable(p,150,150)
def CFL_22(p):
    return EOS_CFL_stable(p,160,150)
def CFL_23(p):
    return EOS_CFL_stable(p,170,150)



# Symbolic definition of the CFL EoSs models
# The same 23 models are being defined:

# Name   \\ B_eff [MeV*fm^-3] \\ Δ [MeV]
# CFL-1  \\   60              \\ 50 
# CFL-2  \\   60              \\ 100 
# CFL-3  \\   60              \\ 150 
# CFL-4  \\   70              \\ 50
# CFL-5  \\   70              \\ 100 
# CFL-6  \\   70              \\ 150
# CFL-7  \\   80              \\ 50
# CFL-8  \\   80              \\ 100
# CFL-9  \\   80              \\ 150
# CFL-10 \\   90              \\ 50
# CFL-11 \\   90              \\ 100
# CFL-12 \\   90              \\ 150
# CFL-13 \\   100             \\ 100 
# CFL-14 \\   100             \\ 150 
# CFL-15 \\   110             \\ 100
# CFL-16 \\   110             \\ 150
# CFL-17 \\   120             \\ 100
# CFL-18 \\   120             \\ 150
# CFL-19 \\   130             \\ 150
# CFL-20 \\   140             \\ 150
# CFL-21 \\   150             \\ 150
# CFL-22 \\   160             \\ 150
# CFL-23 \\   170             \\ 150


def CFL_1_sym(p):
    return EOS_CFL_stable_sym(p,60,50)
def CFL_2_sym(p):
    return EOS_CFL_stable_sym(p,60,100)
def CFL_3_sym(p):
    return EOS_CFL_stable_sym(p,60,150)
def CFL_4_sym(p):
    return EOS_CFL_stable_sym(p,70,50)
def CFL_5_sym(p):
    return EOS_CFL_stable_sym(p,70,100)
def CFL_6_sym(p):
    return EOS_CFL_stable_sym(p,70,150)
def CFL_7_sym(p):
    return EOS_CFL_stable_sym(p,80,50)
def CFL_8_sym(p):
    return EOS_CFL_stable_sym(p,80,100)
def CFL_9_sym(p):
    return EOS_CFL_stable_sym(p,80,150)
def CFL_10_sym(p):
    return EOS_CFL_stable_sym(p,90,50)
def CFL_11_sym(p):
    return EOS_CFL_stable_sym(p,90,100)
def CFL_12_sym(p):
    return EOS_CFL_stable_sym(p,90,150)
def CFL_13_sym(p):
    return EOS_CFL_stable_sym(p,100,100)
def CFL_14_sym(p):
    return EOS_CFL_stable_sym(p,100,150)
def CFL_15_sym(p):
    return EOS_CFL_stable_sym(p,110,100)
def CFL_16_sym(p):
    return EOS_CFL_stable_sym(p,110,150)
def CFL_17_sym(p):
    return EOS_CFL_stable_sym(p,120,100)
def CFL_18_sym(p):
    return EOS_CFL_stable_sym(p,120,150)
def CFL_19_sym(p):
    return EOS_CFL_stable_sym(p,130,150)
def CFL_20_sym(p):
    return EOS_CFL_stable_sym(p,140,150)
def CFL_21_sym(p):
    return EOS_CFL_stable_sym(p,150,150)
def CFL_22_sym(p):
    return EOS_CFL_stable_sym(p,160,150)
def CFL_23_sym(p):
    return EOS_CFL_stable_sym(p,170,150)


# Defining a list to store the info of the CFL EoSs
eos_list_cfl = [["CFL-1",CFL_1,CFL_1_sym,60,50],
                ["CFL-2",CFL_2,CFL_2_sym,60,100],
                ["CFL-3",CFL_3,CFL_3_sym,60,150],
                ["CFL-4",CFL_4,CFL_4_sym,70,50],
                ["CFL-5",CFL_5,CFL_5_sym,70,100],
                ["CFL-6",CFL_6,CFL_6_sym,70,150],
                ["CFL-7",CFL_7,CFL_7_sym,80,50],
                ["CFL-8",CFL_8,CFL_8_sym,80,100],
                ["CFL-9",CFL_9,CFL_9_sym,80,150],
                ["CFL-10",CFL_10,CFL_10_sym,90,50],
                ["CFL-11",CFL_11,CFL_11_sym,90,100],
                ["CFL-12",CFL_12,CFL_12_sym,90,150],
                ["CFL-13",CFL_13,CFL_13_sym,100,100],
                ["CFL-14",CFL_14,CFL_14_sym,100,150],
                ["CFL-15",CFL_15,CFL_15_sym,110,100],
                ["CFL-16",CFL_16,CFL_16_sym,110,150],
                ["CFL-17",CFL_17,CFL_17_sym,120,100],
                ["CFL-18",CFL_18,CFL_18_sym,120,150],
                ["CFL-19",CFL_19,CFL_19_sym,130,150],
                ["CFL-20",CFL_20,CFL_20_sym,140,150],
                ["CFL-21",CFL_21,CFL_21_sym,150,150],
                ["CFL-22",CFL_22,CFL_22_sym,160,150],
                ["CFL-23",CFL_23,CFL_23_sym,170,150]
]
