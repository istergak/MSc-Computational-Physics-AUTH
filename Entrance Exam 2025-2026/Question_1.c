#include <stdio.h>

// MSc Computational Physics AUTH
// Mandatory Entrance Exam
// Academic Year: 2025-2026

// Question 1/5:

// Create a programm, where two real numbers are appended as values to the variables a and b.
// The programm then should calculate and print the value of the expression (a+b)^2.

int main(){
    // Defining two variables a,b and appending real values to them
    float a=2.45, b=7.32;

    // Printing the values of a and b variables (OPTIONAL)
    printf("\nThe value of a is: %.2f\n",a);
    printf("The value of b is: %.2f\n",b);
     

    // Defining the result of the requested expression (a+b)^2 as a float variable
    float res;

    // Calculating the value of the result
    res = (a+b)*(a+b);

    // Printing the result with 4 decimal points
    printf("The result (a+b)^2 is: %.4f\n", res);

    // Adding a scanf command to prevent .exe from closing (NOT NEEDED if the .exe runs via a terminal)
    // scanf("%f",res);


    return 0;
}