#include <stdio.h>
#include <stdlib.h>

#include "MLP_init.h"

/* MLP network architecture initializer */
void mlp_init(multiLayerPerceptron *mlp, int inLayerSize, int outLayerSize, int nHiddenLayers, int hiddenLayerSize){
    /* define MLP network size. */
    mlp->inLayerSize = inLayerSize;
    mlp->outLayerSize = outLayerSize;
    mlp->nHiddenLayers = nHiddenLayers;
    mlp->hiddenLayerSize = hiddenLayerSize;
    mlp->totalLayersSize = mlp->nHiddenLayers+2;

    /* dynamically allocate wheight matrix. */
    mlp->weigthMatrix = (float ***) calloc(mlp->totalLayersSize, sizeof(float **));
    if (mlp->weigthMatrix == NULL){
        fprintf(stderr, "MLP wheight matrix initialization error.\n");
        return;
    }

    /* input layer mem allocation */
    /* of size [hiddenLayerSize x inLayersSize]*/
    mlp->weigthMatrix[0] = (float **) calloc(mlp->hiddenLayerSize, sizeof(float *));
    if (mlp->weigthMatrix[0] == NULL){
        fprintf(stderr, "MLP input layer initialization error.\n");
        return;
    }
    for(int i=0;i<mlp->hiddenLayerSize;i++){
        mlp->weigthMatrix[0][i] = (float *) calloc(mlp->inLayerSize, sizeof(float));
        if (mlp->weigthMatrix[0][i] == NULL){
            fprintf(stderr, "MLP input layer matrix row [%d] initialization error.\n",i);
            return;
        }
    }

    /* hidden layers mem allocation. */
    for(int i=1;i<=mlp->nHiddenLayers;i++){
        mlp->weigthMatrix[i] = (float **) calloc(mlp->hiddenLayerSize, sizeof(float *));
        if (mlp->weigthMatrix[i] == NULL){
            fprintf(stderr, "MLP hidden layer [%d] initialization error.\n",i);
            return;
        } 
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            mlp->weigthMatrix[i][j] = (float *) calloc(mlp->hiddenLayerSize, sizeof(float));
            if (mlp->weigthMatrix[i][j] == NULL){
                fprintf(stderr, "MLP hidden layer [%d], row [%d] initialization error.\n",i,j);
                return;
            }
        }
    }

    /* output layer mem allocation*/
    mlp->weigthMatrix[mlp->totalLayersSize-1] = (float **) calloc(mlp->outLayerSize, sizeof(float *));
    if (mlp->weigthMatrix[mlp->totalLayersSize-1] == NULL){
        fprintf(stderr, "MLP output layer initialization error.\n");
        return;
    }
    for(int i=0;i<mlp->outLayerSize;i++){
        mlp->weigthMatrix[mlp->totalLayersSize-1][i] = (float *) calloc(mlp->hiddenLayerSize, sizeof(float));
        if (mlp->weigthMatrix[mlp->totalLayersSize-1][i] == NULL){
            fprintf(stderr, "MLP output layer matrix row [%d] initialization error.\n",i);
            return;
        }
    }

    printf("MLP initialization complete.\n");
    return;

}

/* MLP network memory deallocation*/
void mlp_free(multiLayerPerceptron *mlp){

    /*free input matrix*/
    for(int j=0;j<mlp->hiddenLayerSize;j++){
        free(mlp->weigthMatrix[0][j]);
    }

    /*free hidden layers matrixes*/    
    for(int i=1;i<=mlp->nHiddenLayers;i++){
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            free(mlp->weigthMatrix[i][j]);
        }
    }

    /*free output layer matrix*/
    for(int j=0;j<mlp->outLayerSize;j++){
        free(mlp->weigthMatrix[mlp->totalLayersSize-1][j]);
    }

    for(int i=0;i<mlp->totalLayersSize;i++){
        free(mlp->weigthMatrix[i]);
    }

    free(mlp->weigthMatrix);

}
