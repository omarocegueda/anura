#ifndef BFGS_H
#define BFGS_H
#include "linearAlgebra.h"
double ternarySearch(ScalarFunction objective, double *x0, double *p, int n, double a, double b, double tol, double *data, int nData);
double BFGS_Sherman_Morrison(ScalarFunction objective, VectorFunction gradient, double *x0, double *B0, int n, double tol, int maxIter, void *data);
double BFGS_Nocedal(ScalarFunction objective, VectorFunction gradient, double *x0, double *B0, int n, double tol, int maxIter, void *data);
#endif
