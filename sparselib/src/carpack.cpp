#ifdef USE_ARPACK
#include "carpack.h"
#include "linearalgebra.h"
#include <iostream>
using namespace std;
//--------interfaces to arpack (fortran) routines---------
extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, 
			int *ipntr, double *workd, double *workl, int *lworkl, int *info);

extern "C" void dseupd_(int *rvec, char *howmany, int *select, double *d,
			double *z, int *ldz, double *sigma, char *bmat, 
			int *n, char *which, int *nev, double *tol, 
			double *resid, int *ncv, double *v, int *ldv, 
			int *iparam, int *ipntr, double *workd, double *workl, 
			int *lworkl, int *info);
//-----------------------------------------------		

int arpack_symetric_evd(double *M, int n, int k, double *evec, double *eval){
	int ido = 0; 
	char bmat[2] = "I"; /* stabdard symmetric eigenvalue problem*/
	char which[3] = "LA"; /* Largest eigenvalue*/
	//har which[3] = "LM"; /* Largest eigenvalue norm*/
	//char which[3] = "SA"; /* Smallest eigenvalue*/
	//char which[3] = "SM"; /* Smallest eigenvalue norm */
	
	
	double tol = 0.0; /* Machine precision*/
	double *resid=new double[n];
	int ncv = 4*k; 
	if(ncv>n){
		ncv = n;
	}
	int ldv=n;
	double *v=new double[ldv*ncv];
	int *iparam=new int[11];
	iparam[0] = 1;   // Specifies the shift strategy (1->exact)
	iparam[2] = 3*n; // Maximum number of iterations
	iparam[6] = 1;   /* Sets the mode of dsaupd.
		      1 is exact shifting,
		      2 is user-supplied shifts,
		      3 is shift-invert mode,
		      4 is buckling mode,
		      5 is Cayley mode. */

	int *ipntr=new int[11];
	double *workd=new double[3*n];
	double *workl=new double[ncv*(ncv+8)];
	int lworkl = ncv*(ncv+8); /* Length of the workl array */
	int info = 0; /* Passes convergence information out of the iteration
					routine. */
	int rvec = 1; /* Specifies that eigenvectors should be calculated */
	int *select=new int[ncv];
	double *d=new double[2*ncv];

	do{
		dsaupd_(&ido, bmat, &n, which, &k, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		if ((ido==1)||(ido==-1)){
			multMatrixVector(M,workd+ipntr[0]-1, n,n, workd+ipntr[1]-1);
		}
	}while((ido==1)||(ido==-1));

	if (info<0) {
         cout << "Error [dsaupd], info = " << info << "\n";
	}else{
		double sigma;
		int ierr;
		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat, &n, which, &k, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);

		if(ierr!=0){
			cout << "Error [dseupd], info = " << ierr << "\n";	
		}else if(info==1){
			cout << "Maximum number of iterations reached.\n\n";
		}else if(info==3){
			cout << "No shifts could be applied during implicit\n";
			cout << "Arnoldi update, try increasing NCV.\n\n";
		}
		memcpy(eval, d, sizeof(double)*k);
		memcpy(evec, v, sizeof(double)*k*n);
		
	}
	return 0;
}

int arpack_symetric_evd(SparseMatrix &M, int n, int k, double *evec, double *eval){
	int ido = 0; 
	char bmat[2] = "I"; /* stabdard symmetric eigenvalue problem*/
	char which[3] = "LA"; /* Largest eigenvalue*/
	//har which[3] = "LM"; /* Largest eigenvalue norm*/
	//char which[3] = "SA"; /* Smallest eigenvalue*/
	//char which[3] = "SM"; /* Smallest eigenvalue norm */
	double tol = 1e-6; /* Machine precision*/
	double *resid=new double[n];
	int ncv = 4*k; 
	if(ncv>n){
		ncv = n;
	}
	int ldv=n;
	double *v=new double[ldv*ncv];
	int *iparam=new int[11];
	iparam[0] = 1;   // Specifies the shift strategy (1->exact)
	iparam[2] = 100*n; // Maximum number of iterations
	iparam[6] = 1;   /* Sets the mode of dsaupd.
		      1 is exact shifting,
		      2 is user-supplied shifts,
		      3 is shift-invert mode,
		      4 is buckling mode,
		      5 is Cayley mode. */

	int *ipntr=new int[11];
	double *workd=new double[3*n];
	double *workl=new double[ncv*(ncv+8)];
	int lworkl = ncv*(ncv+8); /* Length of the workl array */
	int info = 0; /* Passes convergence information out of the iteration
					routine. */
	int rvec = 1; /* Specifies that eigenvectors should be calculated */
	int *select=new int[ncv];
	double *d=new double[2*ncv];

	do{
		dsaupd_(&ido, bmat, &n, which, &k, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		if ((ido==1)||(ido==-1)){
			M.multVecRight(workd+ipntr[0]-1, workd+ipntr[1]-1);
		}
	}while((ido==1)||(ido==-1));

	if (info<0) {
         cout << "Error [dsaupd], info = " << info << "\n";
	}else{
		double sigma;
		int ierr;
		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat, &n, which, &k, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);

		if(ierr!=0){
			cout << "Error [dseupd], info = " << ierr << "\n";	
		}else if(info==1){
			cout << "Maximum number of iterations reached.\n\n";
		}else if(info==3){
			cout << "No shifts could be applied during implicit\n";
			cout << "Arnoldi update, try increasing NCV.\n\n";
		}
		memcpy(eval, d, sizeof(double)*k);
		memcpy(evec, v, sizeof(double)*k*n);
		
	}
	return 0;
}

#endif
