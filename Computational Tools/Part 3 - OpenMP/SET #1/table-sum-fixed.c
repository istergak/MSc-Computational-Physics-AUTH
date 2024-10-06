#include <stdio.h>
#include <omp.h>

#define N 10

int main(void) {

int i,
    sum,
    a[N];

 for(i=0;i<N;i++) a[i] = i;

 sum = 0;

// declare the sum variable as a reduction (and not as a shared) variable in the parallel region data scope
#pragma omp parallel for shared(a) private(i) reduction(+:sum)\
                     default(none)
 for(i=0;i<N;i++) sum += a[i];

 printf("sum = %d\n",sum);

return 0;

}
