#ifndef NNSLS_PGS_H
#define NNSLS_PGS_H

/**
	solves: min ||c*Phi*alpha-s|| + lambda*(||alpha||_2)^2 iteratively using Gauss-Seidel
*/
int nnsls_pgs2(double *Phi, int n, int m, double *s, double *alpha, double lambda, double *error=0);

#endif
