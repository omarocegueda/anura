/*Least Squares solver
	Author: Omar Ocegueda
*/

#ifndef LS_H
#define LS_H
//expects E to be given by columns (first n elements E[0], ..., E[n-1] form the first column)
//>>>>>>>>>>>NOTE:'out' but be assigned at least as much memory as 'b', for efficiency reasons<<<<<<<<<<
void solveLeastSquares(double *E, long n, long m, double *b, double *out);

//	solves: min ||Ex-b||^2, considering only the mm columns of E listed in I
//	expects E to be given by rows (first n elements E[0], ..., E[m-1] form the first row)
//>>>>>>>>>>>NOTE:'out' but be assigned at least as much memory as 'b', for efficiency reasons<<<<<<<<<<
void solveSubsetLeastSquares(double *E, long n, long m, double *b, int *I, long mm, double *EE, double *out);

#endif
