#include <stdio.h>

// MSc Computational Physics AUTH
// Mandatory Entrance Exam
// Academic Year: 2025-2026

// Question 3/5:

// Create a programm, which reads from the keybord a positive integer number. The programm should then
// calculate and print the value of the factorial of this number (n!=1*2*3*...*n)

int main(){
    // Defining an integer variable n
    int n;

    // Asking the user to give a value to variable n
    printf("\nGive a positive integer number n: ");
    scanf("%d",&n);

    // Checking if the given value is a positive integer
    while(n<=0){
        printf("Invalid input. Please give a positive integer number n: ");
        scanf("%d",&n);
    }

    // Defining a help variable for the factorial and initializing it with value 1
    int n_fact=1; 


    // Calculating the factorial of number n
    for(int i=1;i<n+1;i++){
        n_fact = n_fact*i;
    }

    // Printing the calculated factorial of number n
    printf("The factorial %d! is: %d\n",n,n_fact);

    // Adding a scanf command to prevent .exe from closing (NOT NEEDED if the .exe runs via a terminal)
    // scanf("%f",res);


    return 0;
}