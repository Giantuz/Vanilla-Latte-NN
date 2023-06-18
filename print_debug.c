#include <stdio.h>
#include "print_debug.h"
/* helper function that prints to stdout the network architecture*/
void printNetworkParams(multiLayerPerceptron *mlp){
    printf("Network parameters:\nInput layer: [%d]\nOutput layer: [%d]\nHidden Layers: [%dx%d]\n",
    mlp->inLayerSize,mlp->outLayerSize,mlp->nHiddenLayers,mlp->hiddenLayerSize);
}

/* helper function that prints to stdout W matrixes for each layer*/
void printWMatrix(multiLayerPerceptron *mlp){

    printf("Input Matrix:\n");
    for(int i=0;i<mlp->hiddenLayerSize;i++){
        for(int j=0;j<mlp->inLayerSize;j++){
            printf("%f ",mlp->weigthMatrix[0][i][j]);
        }
        printf("\n");
    }
    printf("\n");

    for(int i=1;i<mlp->nHiddenLayers;i++){
        printf("Hidden Layer #%d Matrix:\n",i);
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            for(int k=0;k<mlp->hiddenLayerSize;k++){
                printf("%f ",mlp->weigthMatrix[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    printf("Output Matrix:\n");
    for(int i=0;i<mlp->outLayerSize;i++){
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            printf("%f ",mlp->weigthMatrix[mlp->nTotalLayers-2][i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return;
}


/* helper function that prints to stdout the biases vector for each hidden layer*/
void printBVectors(multiLayerPerceptron *mlp){
    for(int i=0;i<mlp->nHiddenLayers;i++){
        printf("Hidden Layer #%d Biases:\n",i+1);
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            printf("%f\n",mlp->biasMatrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return;
}


