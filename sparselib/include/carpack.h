#ifndef CARPACK_H
#define CARPACK_H
#include "SparseMatrix.h"
#ifdef USE_ARPACK
int arpack_symetric_evd(double *M, int n, int k, double *evec, double *eval);
int arpack_symetric_evd(SparseMatrix &M, int n, int k, double *evec, double *eval);
#endif
#endif
