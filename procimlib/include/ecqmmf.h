#ifndef ECQMMF_H
#define ECQMMF_H
void iterateNode(int site, int numClases, double **v, double **p, int **neighbors, int* numNeighbors, int numVertices, double lambda, double mu, double *&N, double *&D);
void updateMeanVectors(double **means, double **data, double **p, int numVertices, int dataDimmension, int numClases);
void computeGaussianNegLogLikelihood(double **data, double **models, int numVertices, int numClases, int dataDimmension, double **v);
#endif
