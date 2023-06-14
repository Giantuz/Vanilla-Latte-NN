#ifndef MLP_INIT
#define MLP_INIT

/*Struct containing the multi layer perceptron architecture*/

typedef struct MLP{

    int inLayerSize;
    int outLayerSize;
    int nHiddenLayers;
    int hiddenLayerSize;
    int totalLayersSize;

    float ***weigthMatrix;
    float ***biasMatrix;
    
} multiLayerPerceptron;

void mlp_init(multiLayerPerceptron *mlp, int inLayerSize, int outLayerSize, int nHiddenLayers, int hiddenLayerSize);
void mlp_free(multiLayerPerceptron *mlp);

#endif