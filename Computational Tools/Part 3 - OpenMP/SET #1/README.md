# Brief discription

This directory contains all the files for the **SET #1** of the course. The goal of this set is
to find and fix the errors in three C scripts that implement the OpenMP environment

**Contents**:
1. [table-add1-wrong.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-add1-wrong.c): the code must add 1 at every element of the table, but it does not work properly under the parallelization of the task

**Mistake**: The user forgot to declare that the *for* loop inside the parallel region is to be shared among the threads.

**Correction**: Add the **#pragma omp for** just above the *for* loop inside the parallel region, so that the algorithm will share automatically the *for* loop's load to the threads

2. [table-add1-fixed.c]
4.
