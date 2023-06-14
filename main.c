#include <stdio.h>

#include "MLP_init.h"

#define INPUT_SIZE 3
#define HIDDEN_LAYERS 5
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 10

int main(int argc, char *argv[]){
    multiLayerPerceptron myNN;

    mlp_init(&myNN, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS, HIDDEN_SIZE);
    
    printf("Network parameters:\nInput layer: [%d]\nOutput layer: [%d]\nHidden Layers: [%dx%d]\n",
    myNN.inLayerSize,myNN.outLayerSize,myNN.nHiddenLayers,myNN.hiddenLayerSize);

    printf("Input Matrix:\n");
    for(int i=0;i<myNN.hiddenLayerSize;i++){
        for(int j=0;j<myNN.inLayerSize;j++){
            printf("%f ",myNN.weigthMatrix[0][i][j]);
        }
        printf("\n");
    }

    for(int i=1;i<=myNN.nHiddenLayers;i++){
        printf("Hidden Layer #%d Matrix:\n",i);
        for(int j=0;j<myNN.hiddenLayerSize;j++){
            for(int k=0;k<myNN.hiddenLayerSize;k++){
                printf("%f ",myNN.weigthMatrix[i][j][k]);
            }
            printf("\n");
        }
    }
    

    printf("Output Matrix:\n");
    for(int i=0;i<myNN.outLayerSize;i++){
        for(int j=0;j<myNN.hiddenLayerSize;j++){
            printf("%f ",myNN.weigthMatrix[myNN.totalLayersSize-1][i][j]);
        }
        printf("\n");
    }

    mlp_free(&myNN);

}