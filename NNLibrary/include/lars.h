#ifndef LARS_H
#define LARS_H
int lars(double *X, int n, int m, double *y, double *beta, double lambda);
int nnlars_old(double *inputX, int n, int m, double *inputy, double *beta, double lambda, double *error, bool standardize);
int nnlars(double *X, int n, int m, double *y, double *beta, double lambda, double *error=0, bool standardize=true);
int nnlars_addaptiveScale(double *inputX, int n, int m, double *inputy, double *beta, double mu, double lambda, double *error);
#endif
