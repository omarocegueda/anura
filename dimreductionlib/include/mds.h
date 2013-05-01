#ifndef MDS_H
#define MDS_H
int multidimensionalScaling(double *distMat, int n, int &k, double *&x);

#ifdef USE_ARPACK
int multidimensionalScaling_arpack(double *distMat, int n, int &k, double *&x);
#endif

#endif
