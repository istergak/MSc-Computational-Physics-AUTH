#include <stdio.h> // basic C header
#include <stdlib.h> // basic C header
#include <math.h> // Math header for math functions, like pow()
#include <time.h> // Time header for time measurement
#include <omp.h> // OpenMP header

// MSc Commputational Physics AUTH
// Subject: Computational Tools
// Course: OpenMP
// Semester 1
// Academic Year: 2023-2024

// Implemented by: Ioannis Stergakis
// AEM: 4439

// C script for Monte Carlo simulations using OpenMP

// Compilation and execution commands

// Windows terminal
// Compilation: gcc pi_estimation_MC.c -o pi_estimation_MC -fopenmp
// Execution: pi_estimation_MC {points} {threads}

// Linux terminal
// Compilation: g++ pi_estimation_MC.c -o pi_estimation_MC.exe -fopenmp
// Execution: ./pi_estimation_MC {points} {threads}

// where the {points} and {threads} must be replaced with numbers for the total points
// and the threads used for the Monte Carlo simulations

// Random number generator function prototype
double randf(double,double);


// Main function
int main(int argc, char *argv[]){
    
    // Setting the total number of random points (x,y) to be generated in 2D-plane using external info
    long long int total_points = 1e9; // by default (no external info) set to 10^9 
    if((long long int) atof(argv[1])!=0){// external info has been given
      total_points = (long long int) atof(argv[1]); // Use LL suffix for long long int
    }
    
    // Setting the number of threads to be used for OpenMP parallezation as well as the maximum number of threads
    int max_threads = omp_get_max_threads();
    int num_threads = omp_get_max_threads(); // by default (no external info) set to maximum number of threads
    if((long long int) atof(argv[2])!=0){// external info has been given
      num_threads = (int) atof(argv[2]);
    }

    // Set the number of threads
    omp_set_num_threads(num_threads);

    // Printing the SetUp Info
    printf("\n>>MC SIMULATION INFO (C script):\n");
    printf("--------------------------------------\n");
    printf("Max Physical Threads: %d\n",max_threads);
    printf("Used Threads: %d\n",num_threads);
    printf("Total points: %lld\n",total_points);
    printf("--------------------------------------\n");
     
    // Initializing some useful variables for the Monte Carlo Simulation
    double x_point, y_point; // definition of (x,y) coordinates of random points
    double radius_sq; // definition of the squared radius r^2 variable: r^2 = x^2 + y^2
    double pi_estimation; // definition of π's estimation variable
    double pi_error; // definition of error in π's estimation

    long long int in_circle_points = 0; // initialization of counter of points that lie inside the circle: x^2+y^2=1

    // Initializing random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Definition of time measurement variables
    clock_t start_time,end_time;
    double cpu_time;

    // Starting execution time measurement
    start_time = clock();
    
    // Implementing the Monte Carlo Simulation inside a parallel region of OpenMP
    #pragma omp parallel shared(pi_estimation,pi_error,total_points)\
                        private(x_point,y_point,radius_sq)\
                        reduction(+:in_circle_points)\
                        default(none)
    {
      // Parallel Iterative Process
      #pragma omp for
      for(long long int i = 0; i < total_points; i++){

        // Random point generation
        x_point = randf(-1.0,1.0); // random x coordinate in the interval [-1,1]
        y_point = randf(-1.0,1.0); // random y coordinate in the interval [-1,1]
        radius_sq = x_point*x_point + y_point*y_point; // random radius

        // Check if the random point is inside the circle (i.e if r^2=x^2+y^2<=1) 
        if(radius_sq<=1) in_circle_points++; 
        }

    }
    
    // Estimating π after the end of the iterative simulation
    pi_estimation = 4*(double) in_circle_points/(double) total_points; // estimation
    pi_error = (fabs(M_PI - pi_estimation) / M_PI) * 100; // true percent relative error

    // Ending execution time measurement
    end_time = clock();

    // Calculating the CPU execution time (in seconds)
    cpu_time = ((double)(end_time-start_time))/CLOCKS_PER_SEC;

    // Printing the results
    printf("Pi Estimation: %.10lf\n",pi_estimation);
    printf("True Percent Relative Error: %.6f%%\n",pi_error);
    printf("Execution Time: %.10f seconds\n",cpu_time);
    printf("--------------------------------------\n"); 

    return 0;
}

// Random number generator function definition
double randf(double n_min, double n_max){
	
	return n_min + (double) (n_max-n_min)*(rand())/( RAND_MAX);
}

