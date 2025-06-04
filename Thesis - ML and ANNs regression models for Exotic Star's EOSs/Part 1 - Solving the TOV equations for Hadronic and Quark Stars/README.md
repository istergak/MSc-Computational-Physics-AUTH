# Part 1: Solving the TOV equations

This directory contains the modules (.py) and Jupyter notebooks (.ipynb) we developed in Python for the solution of TOV equations. We include solutions for different models of equations of states (EOSs) and both for Neutron and Quark stars. The solutions data are also included in seperate directories.

## Neutron Stars

[eos_lib_NS.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/eos_lib_NS.py): module where the main EOSs for the core and the crust EOSs of Neutron Stars are defined (numerically and symbolically) and being stored in lists.

[tov_solver_NS.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_NS.py): module to solve the TOV equations serially for a single core EOS of a Neutron Star included in **eos_lib_NS.py** module. The crust EOSs are always included.

[tov_solver_NS_par.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_NS_par.py): module to solve the TOV equations in parallel for a selected number of main models for the core EOS of a Neutron Star included in **eos_lib_NS.py** module. Each model is distributed to a single thread for solution. The crust EOSs are always included. 

[StudyPolyNS.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/StudyPolyNS.ipynb): Jupyter notebook where the general methodology of parametrizing an EOS, using piecewise polytropes is being studied. The pressure values of the HLPS-2 and HLPS-3 main EOSs at the nuclear saturation density are being determined.

[tov_solver_polyNS_par.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_polyNS_par.py): module to solve the TOV equations in parallel for a selected number of polytropic mock EOSs (combined with a main EOS model: either HLPS-2 or HLPS-3) as the core EOSs of a Neutron Star. Each mock EOS is distributed to a single thread for solution. The crust EOSs are always included.

[tov_solver_polyNS_par2.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_polyNS_par2.py): same as **tov_solver_polyNS_par.py** module, but corrections in the polytropic part of the mock EOSs are being made to avoid violation of causality. The corrections involve the replace of the polytropic part of the mock EOS that violates causality with a linear part, with fixed slope that does not violate causality.


## Quark Stars

[eos_lib_QS.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/eos_lib_QS.py): module where CFL EOSs of Quark Stars are defined (numerically and symbolically) and stored in lists

[tov_solver_cflQS.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_cflQS.py): module to solve the TOV equations serially for a single CFL EOS of a Quark Star included in **eos_lib_QS.py** module. No crust EOSs are included.

[tov_solver_cflQS_par.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_cflQS_par.py): module to solve the TOV equations in parallel for a selected number of CFL EOSs of a Quark Star. No crust EOSs are included. The user can determine the ranges of the $B_{eff}$ and $\Delta$ parameters and generate arbitrarily the preferred CFL models.

[tov_solver_mitQS_par.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars/tov_solver_mitQS_par.py): module to solve the TOV equations in parallel for a selected number of MIT bag EOSs of a Quark Star. No crust EOSs are included. The user can determine the range of the $B_{eff}$ parameter and generate arbitrarily the preferred MIT bag models.
