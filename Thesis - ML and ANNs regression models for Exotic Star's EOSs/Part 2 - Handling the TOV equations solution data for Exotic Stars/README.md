# Part 2 - Handling the data from the solutions of TOV equations

In this directory we present the codes we developed, in order to utilize the data obtained from the solutions of TOV equations, both for Quark and Neutron Stars. In general, we include validation of parameters, plotting and sampling of data.

**Notes**:

The data from the solution of TOV equations must be included in the same dierectory with the following modules and notebooks, for the algorithms to run properly.

**Links to solution data**:

->Main Neutron Star EOSs solution data: [here](https://drive.google.com/drive/folders/1tzKjCmlceXtXBja5AHdiDcJv3q4RlP-a)<br>
->Polytropic and linear Neutron Star EOSs solution data: [here](https://drive.google.com/drive/folders/1iJNcD9arRdhSIKv6iQoxUYLkpZilexu4)<br>
->MIT bag Quark Star EOSs solution data: [here](https://drive.google.com/drive/folders/1ggiVcc7ypTlooGC-VL1rPPhoCQL4Udgp)<br>
->CFL Quarks Star EOSs solution data: [here](https://drive.google.com/drive/folders/1v7T21M9TRBBpSC7YjSZnBx3M62FIM82s)

## Modules

[ExoticStarsDataHandling.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars/ExoticStarsDataHandling.py): module containing functions and classes for: **a)** validating the parameters of polytropic Neutron Stars EOSs ($\Gamma$ parameter) and CFL Quark Stars EOSs ($B_{eff}$ and $\Delta$ parameters), **b)** plotting $E_c-P_c$, $c^2_s-P_c$ and $M-R$ curves of the respective EOSs and **c)** sampling data for regression purposes.

[ExoticStarsDataHandling2.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars/ExoticStarsDataHandling2.py): a different version of **ExoticStarsDataHandling.py** module, containing major or minor differences in some classes.

[plot_curves_NS.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars/plot_curves_NS.py): module that plots the $E_c-P_c$ and $M-R$ curves of the respective main Neutron Star EOSs. Needs to be executed from a terminal.

## Jupyter Notebooks

[ExoticStarsResults_1.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars/ExoticStarsResults_1.ipynb): demonstrating the use of **ExoticStarsDataHandling.py** module for plotting curves

[ExoticStarsResults_2.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars/ExoticStarsResults_2.ipynb): demonstrating the use of **ExoticStarsDataHandling.py** module for sampling data for regression purposes
