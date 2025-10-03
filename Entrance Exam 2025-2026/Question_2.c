#include <stdio.h>

// MSc Computational Physics AUTH
// Mandatory Entrance Exam
// Academic Year: 2025-2026

// Question 2/5:

// Create a programm, where two numbers are appended as values to the variables a and b.
// The programm then should switch the values: a should take the value of b and b the value of a.

int main(){
    // Defining two variables a,b and appending integer values to them
    int a=1, b=20;

    // Printing the values of a and b before the switch
    printf("\nBEFORE SWITCH:\n");
    printf("Value of a: %d\n",a);
    printf("Value of b: %d\n\n",b);

    // Defining a help variable for switch and appending the value of a to it
    float temp=a;

    // SWITCH OF VALUES
    // Appending the value of b to a
    a=b;

    // Appending the value of temp variable to b
    b=temp;

    // Printing the values of a and b before the switch
    printf("AFTER SWITCH:\n");
    printf("Value of a: %d\n",a);
    printf("Value of b: %d\n\n",b);

    // Adding a scanf command to prevent .exe from closing (NOT NEEDED if the .exe runs via a terminal)
    // scanf("%f",res);


    return 0;
}