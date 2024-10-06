# Brief discription

This directory contains all the files for the **SET #1** of the course. The goal of this set is
to find and fix the errors in three C scripts that implement the OpenMP environment

**Contents**:
1. [table-add1-wrong.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-add1-wrong.c): the code must add 1 at every element of the table, but it does not work properly under the parallelization of the task

**Mistake**: The user forgot to declare that the *for* loop inside the parallel region is to be shared among the threads.

**Correction**: Add the **#pragma omp for** command just above the *for* loop inside the parallel region, so that the algorithm will share automatically the *for* loop's load to the threads

2. [table-add1-fixed.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-add1-fixed.c): same as 1. but with the correcion included, for the code to print the expected result
  
3. [table-implicit-notpar.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-implicit-notpar.c): the code must change the elements of the table by adding the value of the previous element to the value of the current element, using a *for* loop. Since the value of the first element of the table is 0, executing the loop serially, i.e. with a single thread, would result in the sequential addition of 0 to all elements of the table. However, when the code runs with more threads and executes the *for* loop parallely the result is not the expected: not all the elements are 0 at the end of the loop, while theee are elements in the table with the same value.

**Mistake**: There is a problem with the synchronization of the threads. Since all threads execute their respective simultaneously, the number 0 does not have time to be distributed to the subsequent elements of the table, rather has time to be distributed only to the elements of the table that are included within the iterative load of the first thread.

**Correction**: To fix the syncroziation problem we can use a schedule for the parallel *for* loop

4. [table-implicit-notpar-fix-guided.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-implicit-notpar-fix-guided.c): same as 3. but with the **guided** schedule included for the **for** loop. This way, the algorithm starts handing out large chunks of reps to the threads, which it keeps reducing. Moreover, when the load of a thread is done, this thread gets to run the load of another thread. This solution is effective, since we have a table with relatively short length: when the load of the first thread is done, a big amount of elements in the table will already be 0 and since this load is completed quickly (due to the small length of the table) the same thread continues its work, by changing the rest elements of the table (the ones they immediately following the already 0 elements).

5. [table-sum-wrong.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-sum-wrong.c): the code must calculate the sum of  all the elements in the table but it gives the wrong result

**Mistake**: the user has declared the sum variable as a shared variable in the data scope of the parallel region. In this way, all threads have access to the memory location of the sum variable, so during the calculation of its value in each iteration, the results from each thread are mixed up and lead to an incorrect final sum value.

**Correction**: The sum variable must be declared as reduction variable in the data scope of the parallel region. This way, every thread calculates a certain value for the sum (i.e. sums only the elements of the table that correspond to its load), without the interference of the results of the other threads (i.e. the sums of other elements). Thus, every element of the table is considered only once (as it should) in the sum. At the end of the loop, the algorithm adds all the separately calculated values of the sum to obtain the final value of the total sum, i.e. the sum of all the elements of the table

6. [table-sum-fixed.c](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Computational%20Tools/Part%203%20-%20OpenMP/SET%20%231/table-sum-fixed.c): same as 5. but with the correction in the sum variable included
