#ifndef ECKQMMF_H
#define ECKQMMF_H
//----- EC-KQMMF for dense kernel matrices---
void computeKernelIntegralFactors(double *P, int nrows, int ncols, int K, double *kernel, double *s, double *sk1, double *sk2);
void iterateP_kernel(double *P, int nrows, int ncols, int K, double lambda, double mu, double *kernel, double *s, double *sk1, double *sk2);
void computeSparseKernel(double sigma, double *img, int nrows, int ncols, int numBands, double *kernel);
void computeDenseKernel(double sigma, double *img, int nrows, int ncols, int numBands, double *kernel);

//----- EC-KQMMF for sparse kernel matrices---
double iterateP_kernel(double *P, double *pbuff, SparseMatrix &kernel, int K, double lambda, double mu, double *s, double *sk1, double *sk2);
void computeKernelIntegralFactors(double *P, SparseMatrix &kernel, int K, double *s, double *sk1, double *sk2);
#endif
