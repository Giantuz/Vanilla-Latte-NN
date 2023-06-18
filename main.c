#include <stdio.h>

#include "network_structure.h"
#include "MLP_init.h"
#include "print_debug.h"

#define INPUT_SIZE 4
#define HIDDEN_LAYERS 4
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2

int main(int argc, char *argv[]){
    multiLayerPerceptron myNN;

    mlp_init(&myNN, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS, HIDDEN_SIZE);
    
    printWMatrix(&myNN);
    printBVectors(&myNN);
    
    mlp_free(&myNN);

}