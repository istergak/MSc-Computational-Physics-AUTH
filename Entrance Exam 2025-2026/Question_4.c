#include <stdio.h>

// MSc Computational Physics AUTH
// Mandatory Entrance Exam
// Academic Year: 2025-2026

// Question 4/5:

// Create a programm, which reads from the keybord a positive integer number in the interval [2,1000] 
// and prints if the number is a prime number 

int main(){
    // Defining an integer variable n
    int n;

    // Defining a help variable for the modulo
    int mod; 

    // Asking the user to give a value to variable n
    printf("\nGive a positive integer number n in the interval [2,1000]: ");
    scanf("%d",&n);

    // Checking if the given value is a positive integer in the interval [2,1000]
    while(n<2){
        printf("Invalid input. Please give a positive integer number n in the interval [2,1000]: ");
        scanf("%d",&n);
    }

    // Determining if the given number n is a prime number
    // CASE 1: n=2, so n IS a prime number
    if(n==2){
        printf("Number %d IS a prime number\n",n);
    }
    else{
        for(int i=2;i<=n;i++){
            // Getting the modulo of the (integer) division n/i
            mod=n%i; 

            // CASE 2: n>2, i<n and mod=0 -> n is divisble by at least one number besides itself and 1 ===> n IS NOT a prime number
            if(i<n && mod==0){
                printf("Found division by %d besides %d and 1\n",i,n);
                printf("Number %d IS NOT a prime number\n",n);
                break;
            }

            // CASE 3: n>2 and i=n -> i has reached n ===> n has been found indivisible besides itself and 1 ===> n IS a prime number
            if(i==n){
                printf("Number %d IS a prime number\n",n);
            }
        }
    }

    // Adding a scanf command to prevent .exe from closing (NOT NEEDED if the .exe runs via a terminal)
    // scanf("%f",res);


    return 0;
}