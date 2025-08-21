#include <stdio.h>
#include <pthread.h>

// These are the shared variables. Both threads will access them.
int num1 = 5;
int num2 = 7;
int sum_result;
int product_result;

// Function for the first thread (calculates the sum)
void *calculate_sum(void *arg) {
    sum_result = num1 + num2;
    printf("Thread 'Sum': Calculated %d + %d = %d\n", num1, num2, sum_result);
    pthread_exit(NULL); // Terminate the thread
}

// Function for the second thread (calculates the product)
void *calculate_product(void *arg) {
    product_result = num1 * num2;
    printf("Thread 'Product': Calculated %d * %d = %d\n", num1, num2, product_result);
    pthread_exit(NULL); // Terminate the thread
}

int main() {
    pthread_t thread1, thread2; // Variables to hold thread IDs

    printf("Main: Starting threads...\n");

    // Create the first thread that runs calculate_sum
    if (pthread_create(&thread1, NULL, calculate_sum, NULL)) {
        fprintf(stderr, "Error creating sum thread\n");
        return 1;
    }

    // Create the second thread that runs calculate_product
    if (pthread_create(&thread2, NULL, calculate_product, NULL)) {
        fprintf(stderr, "Error creating product thread\n");
        return 1;
    }

    printf("Main: Threads started. Waiting for them to finish...\n");

    // Wait for both threads to complete before main continues.
    // This is crucial to avoid the program ending prematurely.
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    printf("Main: Both threads are done. The results are Sum: %d, Product: %d\n", sum_result, product_result);
    printf("Main: Now exiting.\n");

    return 0;
}