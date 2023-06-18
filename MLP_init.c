#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "MLP_init.h"

/*Marsaglia polar method*/
float generateGaussian(float mean, float stdDev) {
    static float spare;
    static int hasSpare = 0;

    if (hasSpare==1) {
        hasSpare = 0;
        return spare * stdDev + mean;
    } else {
        float u, v, s;
        do {
            u = (rand() / ((float)RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((float)RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        hasSpare = 1;
        return mean + stdDev * u * s;
    }
}


void W_init(multiLayerPerceptron *mlp){
    /*currently supports only Xavier initialization*/
    float mean=0,stdDev;

    /*Input matrix weight init*/
    stdDev = sqrtf((float) 1 / mlp->inLayerSize);

    for(int j=0;j<mlp->hiddenLayerSize;j++){
        for(int k=0;k<mlp->inLayerSize;k++){
            mlp->weigthMatrix[0][j][k] = generateGaussian(mean,stdDev);
        }
    }

    /*Hidden layers matrix weight init*/
    stdDev = sqrtf((float) 1 / mlp->hiddenLayerSize);

    for(int i=1;i<mlp->nHiddenLayers;i++){
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            for(int k=0;k<mlp->inLayerSize;k++){
                mlp->weigthMatrix[i][j][k] = generateGaussian(mean,stdDev);
            }
        }        
    }

    /*Output matrix weight init*/
    /*stdDev = sqrtf((float) 1 / mlp->hiddenLayerSize);*/

    for(int j=0;j<mlp->outLayerSize;j++){
        for(int k=0;k<mlp->hiddenLayerSize;k++){
            mlp->weigthMatrix[mlp->nTotalLayers-2][j][k] = generateGaussian(mean,stdDev);
        }
    }

    return;
}


void W_alloc(multiLayerPerceptron *mlp){

    /* dynamically allocate wheight matrix. */
    /* of size [nTotalLayers-1]*/
    mlp->weigthMatrix = (float ***) calloc(mlp->nTotalLayers-1, sizeof(float **));
    if (mlp->weigthMatrix == NULL){
        fprintf(stderr, "MLP wheight matrix initialization error.\n");
        return;
    }

    /* input layer matrix allocation */
    /* of size [hiddenLayerSize x inLayersSize]*/
    mlp->weigthMatrix[0] = (float **) calloc(mlp->hiddenLayerSize, sizeof(float *));
    if (mlp->weigthMatrix[0] == NULL){
        fprintf(stderr, "MLP input layer initialization error.\n");
        return;
    }
    for(int j=0;j<mlp->hiddenLayerSize;j++){
        mlp->weigthMatrix[0][j] = (float *) calloc(mlp->inLayerSize, sizeof(float));
        if (mlp->weigthMatrix[0][j] == NULL){
            fprintf(stderr, "MLP input layer matrix row [%d] initialization error.\n",j);
            return;
        }
    }

    /* hidden layers matrix allocation */
    /* of size [hiddenLayerSize x hiddenLayerSize]*/
    for(int i=1;i<mlp->nHiddenLayers;i++){
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

    /* output layer matrix allocation */
    /* of size [outLayerSize x hiddenLayerSize]*/
    mlp->weigthMatrix[mlp->nTotalLayers-2] = (float **) calloc(mlp->outLayerSize, sizeof(float *));
    if (mlp->weigthMatrix[mlp->nTotalLayers-2] == NULL){
        fprintf(stderr, "MLP output layer initialization error.\n");
        return;
    }
    for(int j=0;j<mlp->outLayerSize;j++){
        mlp->weigthMatrix[mlp->nTotalLayers-2][j] = (float *) calloc(mlp->hiddenLayerSize, sizeof(float));
        if (mlp->weigthMatrix[mlp->nTotalLayers-2][j] == NULL){
            fprintf(stderr, "MLP output layer matrix row [%d] initialization error.\n",j);
            return;
        }
    }

    return;

}


void B_alloc(multiLayerPerceptron *mlp){

    /* dynamically allocate bias matrix. */
    /* of size [hiddenLayerSize x nHiddenLayers]*/
    mlp->biasMatrix = (float **) calloc(mlp->nHiddenLayers, sizeof(float *));
    if (mlp->biasMatrix == NULL){
        fprintf(stderr, "MLP biases initialization error.\n");
        return;
    }

    /* hidden layers biases vectors allocation */
    /* of size [nHiddenLayers x 1]*/
    for(int i=0;i<mlp->nHiddenLayers;i++){
        mlp->biasMatrix[i] = (float *) calloc(mlp->hiddenLayerSize, sizeof(float));
        if (mlp->biasMatrix[i] == NULL){
            fprintf(stderr, "MLP hidden layer [%d] bias initialization error.\n",i);
            return;
        } 
    }

    return;

}


/* MLP network architecture initializer */
void mlp_init(multiLayerPerceptron *mlp, int inLayerSize, int outLayerSize, int nHiddenLayers, int hiddenLayerSize){

    /* define MLP network size. */
    mlp->inLayerSize = inLayerSize;
    mlp->outLayerSize = outLayerSize;
    mlp->nHiddenLayers = nHiddenLayers;
    mlp->hiddenLayerSize = hiddenLayerSize;
    mlp->nTotalLayers = mlp->nHiddenLayers+2;

    /* initialize W matrixes. */
    W_alloc(mlp);
    W_init(mlp);
    /* initialize Biases. */
    B_alloc(mlp);

    printf("MLP initialization complete.\n");
    return;

}


void W_free(multiLayerPerceptron *mlp){
    /*free input matrix*/
    for(int j=0;j<mlp->hiddenLayerSize;j++){
        free(mlp->weigthMatrix[0][j]);
    }
    /*free hidden layers matrixes*/  
    for(int i=1;i<mlp->nHiddenLayers;i++){
        for(int j=0;j<mlp->hiddenLayerSize;j++){
            free(mlp->weigthMatrix[i][j]);
        }
    }

    /*free output layer matrix*/
    for(int j=0;j<mlp->outLayerSize;j++){
        free(mlp->weigthMatrix[mlp->nTotalLayers-2][j]);
    }

    for(int i=0;i<mlp->nTotalLayers-1;i++){
        free(mlp->weigthMatrix[i]);
    }
    free(mlp->weigthMatrix);
    return;
}


void B_free(multiLayerPerceptron *mlp){
    for(int i=0;i<mlp->nHiddenLayers;i++){
        free(mlp->biasMatrix[i]);
    }
    free(mlp->biasMatrix);
}


/* MLP network memory deallocation*/
void mlp_free(multiLayerPerceptron *mlp){

    /* free W matrixes*/
    W_free(mlp);

    /* free B vectors*/
    B_free(mlp);
}
