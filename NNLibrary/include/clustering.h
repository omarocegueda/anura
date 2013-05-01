#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <utility>

void getLabelsFromProbs(double *p, int n, int k, int *lab);

std::pair<double, double> iterateFuzzyKMeans(double *data, int n, int dim, double *means, double *P, int k, int m);
std::pair<double, double> iterateAngularFuzzyKMeans(double *data, double *alpha, int n, int dim, double *means, double *P, int k, int m);
void fuzzyKMeans(double *data, int n, int dim, int K, double exponent, int maxIter, double *&means, double *&probs, bool initKMPP=true);
int angularFuzzyKMeans(double *data, double *alpha, int n, int dim, int K, double exponent, int maxIter, double *&means, double *&probs, bool initKMPP=true);
double combineDirections(double alpha0, double *d0, double alpha1, double *d1, int dim, double *u);

double selectFromDistribution(double *D, int n);
#endif
