/*Non-negative Sparse Least Squares solver
	Author: Omar Ocegueda
*/

#ifndef NNSLS_H
#define NNSLS_H

/**
	solves: min ||Ex-f|| + lambda*(||x||_1)^2
*/
int nnsls(double *E, int n, int m, double *f, double *x, double lambda, double *error=0);

#endif
