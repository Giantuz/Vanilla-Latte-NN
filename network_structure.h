#ifndef NETWORK_STRUCTURE
#define NETWORK_STRUCTURE

/*Struct containing the multi layer perceptron architecture*/
typedef struct MLP{

    int inLayerSize;
    int outLayerSize;
    int nHiddenLayers;
    int hiddenLayerSize;
    int nTotalLayers;

    float ***weigthMatrix;
    float **biasMatrix;
    
} multiLayerPerceptron;

#endif