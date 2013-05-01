/*Least Squares solver
	Author: Omar Ocegueda
*/

#include "nnls.h"
#include "linearalgebra.h"
#include <string.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern "C"{
	#include "f2c.h"
	#include "clapack.h"
}
//expects E to be given by columns (first n elements E[0], ..., E[n-1] form the first column)
//>>>>>>>>>>>NOTE:'out' but be assigned at least as much memory as 'b', for efficiency reasons<<<<<<<<<<
void solveLeastSquares(double *E, long n, long m, double *b, double *out){
	integer lda=n;
	integer ldb=n;
	integer info;
	integer nrhs=1;
	double rcond =-1;
	integer rank;
	integer lwork = -1;
	double *work=NULL;
	double wkopt;
	integer *jpvt=new integer[m];	
	memset(jpvt, 0, sizeof(integer)*m);
	memcpy(out, b, sizeof(double)*n);
	dgelsy_(&n, &m, &nrhs, E, &lda, out, &ldb, jpvt, &rcond, &rank, &wkopt, &lwork, &info );
	work = new double[(int)wkopt];
	lwork=(integer)wkopt;
	// Solve the equations A*X = B
	dgelsy_(&n, &m, &nrhs, E, &lda, out, &ldb, jpvt, &rcond, &rank, work, &lwork, &info );
	delete[] work;
	delete[] jpvt;
}


//	solves: min ||Ex-b||^2, considering only the mm columns of E listed in I
//	expects E to be given by rows (first m elements E[0], ..., E[m-1] form the first row)
void solveSubsetLeastSquares(double *E, long n, long m, double *b, int *I, long mm, double *EE, double *out){
	if(mm==m){
		for(int i=0;i<n;++i){
			for(int j=0;j<m;++j){
				EE[j*n+i]=E[i*m+j];
			}
		}
	}else{
		for(int i=0;i<n;++i){
			for(int j=0;j<mm;++j){
				EE[j*n+i]=E[i*m+I[j]];
			}
		}
	}
	solveLeastSquares(EE, n, mm, b, out);
}
