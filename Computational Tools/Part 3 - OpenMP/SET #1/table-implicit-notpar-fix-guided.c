#include <stdio.h>
#include <omp.h>

#define N 12   

int main(void) {

int i,
    nThreads,
    myid,
    A[N];

 for(i=0;i<N;i++) A[i] = i;


#pragma omp parallel shared(A,nThreads) private(i) default(none)
{
 int myid = omp_get_thread_num(); // get ID number of thread
 if(myid==0) nThreads = omp_get_num_threads(); // if the master thread is used get the 

 #pragma omp for schedule(guided) // added the guided schedule for the loop 
 for(i=1;i<N;i++) A[i] =A[i-1];
}

printf("Total number of threads used was: %d\n",nThreads);

printf("The result of parallel guided execution of the algorithm is:\n");

for(i=0;i<N;i++) printf("A[%d] = %d\n",i,A[i]);

return 0;

}
