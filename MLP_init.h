#ifndef MLP_INIT
#define MLP_INIT

#include  "network_structure.h"

float generateGaussian(float mean, float stdDev);

void W_init(multiLayerPerceptron *mlp);
void B_init(multiLayerPerceptron *mlp);
void mlp_init(multiLayerPerceptron *mlp, int inLayerSize, int outLayerSize, int nHiddenLayers, int hiddenLayerSize);

void W_free(multiLayerPerceptron *mlp);
void B_free(multiLayerPerceptron *mlp);
void mlp_free(multiLayerPerceptron *mlp);

#endif