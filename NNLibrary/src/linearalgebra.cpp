/*some useful references:
	http://www.nag.com/numeric/fl/nagdoc_fl23/xhtml/F08/f08waf.xml
	http://www.netlib.org/lapack/lug/node35.html
	http://www.netlib.org/lapack/lug/node36.html#tabdrivegeig
*/

#include "linearalgebra.h"
#include <math.h>
#include <iostream>
using namespace std;
extern "C"{
	#include "f2c.h"
	#include "clapack.h"
}

double evaluateQuadraticForm(double *Q, double *x, int n){
	double sum=0;
	int pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j, ++pos){
			sum+=Q[pos]*x[i]*x[j];
		}
	}
	return sum;
}



void computeInverse(double *A, long N){
    long *IPIV = new integer[N+1];
    long LWORK = N*N;
    double *WORK = new double[LWORK];
    long INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
}

/*
	A is given in column major order
*/
int computePseudoInverseCMO(double *A, long nrows, long ncols, double *Ainv){
	int md=MIN(nrows, ncols);
	double *U=new double[nrows*nrows];
	double *s=new double[md];
	double *Vt=new double[ncols*ncols];
	/* compute optimal workspace size*/
	long lwork = -1;
	double wkopt;
	long info;
	dgesvd_( "All", "All", &nrows, &ncols, A, &nrows, s, U, &nrows, Vt, &ncols, &wkopt, &lwork, &info);
	lwork = (int)wkopt;
	double *work = new double[lwork*sizeof(double)];
	/* Compute SVD */
	dgesvd_( "All", "All", &nrows, &ncols, A, &nrows, s, U, &nrows, Vt, &ncols, work, &lwork, &info );
	if(info>0){
		//return -1;
	}
	/* Free workspace */
	delete[] work;
	delete[] U;
	delete[] s;
	delete[] Vt;
	return 0;
}

/*
	A is given in row major order
*/
int computePseudoInverseRMO(double *Ain, long nrows, long ncols, double *Ainv){
	bool transposed=false;
	double *Acpy=new double[nrows*ncols];
	if(ncols<nrows){
		transpose<double>(Ain, nrows, ncols, Acpy);
		int tmp=nrows;
		nrows=ncols;
		ncols=tmp;
		transposed=true;
	}else{
		memcpy(Acpy, Ain, sizeof(double)*nrows*ncols);
	}
	double *U=new double[ncols*ncols];
	double *s=new double[nrows];
	double *Vt=new double[nrows*nrows];
	/* compute optimal workspace size*/
	long lwork = -1;
	double wkopt;
	long info;
	dgesvd_( "All", "All", &ncols, &nrows, Acpy, &ncols, s, U, &ncols, Vt, &nrows, &wkopt, &lwork, &info);//SVD of transpose(A)
	lwork = (int)wkopt;
	double *work = new double[lwork];
	/* Compute SVD */
	memset(s, 0, sizeof(double)*nrows);
	dgesvd_( "All", "All", &ncols, &nrows, Acpy, &ncols, s, U, &ncols, Vt, &nrows, work, &lwork, &info);//SVD of transpose(A)
	if(info==0){
		//U is ncols*ncols, in CMO
		//s contains the md sibgular values
		//Vt is nrows*nrows, in CMO
		//transpose(A) = USV^T
		//inverse(A^T)=VS^{+}U^{T}, where V is Vt in RMO, U is also in RMO
		for(int i=0;i<nrows;++i){
			if(fabs(s[i])>1e-10){
				s[i]=1.0/s[i];
			}else{
				s[i]=0;
			}
		}
		for(int j=0;j<nrows;++j){
			for(int i=0;i<nrows;++i){
				Vt[i*nrows+j]*=s[j];
			}
		}
		multMatrixMatrix(Vt, U, nrows, nrows, ncols, Ainv);//g-inverse of transpose(A) in RMO
		if(!transposed){
			transpose<double>(Ainv, nrows, ncols, Ainv);
		}
	}
	delete[] work;
	delete[] U;
	delete[] s;
	delete[] Vt;
	delete[] Acpy;
	return -1*(info!=0);
}

void testPseudoInverseRMO(double *A, int nrows, int ncols){
	double *Ainv=new double[ncols*nrows];
	double *tmp0=new double[nrows*nrows];
	double *tmp1=new double[ncols*ncols];
	double *tmp2=new double[ncols*nrows];
	computePseudoInverseRMO(A, nrows, ncols, Ainv);

	//----test A*Ainv*A==A---
	multMatrixMatrix(A, Ainv, nrows, ncols, nrows, tmp0);
	multMatrixMatrix(tmp0, A, nrows, nrows, ncols, tmp2);
	double error=0;
	for(int i=nrows*ncols-1;i>=0;--i){
		error+=SQR(tmp2[i]-A[i]);
	}
	error=sqrt(error);
	cerr<<error<<endl;
	//----test Ainv*A*Ainv==Ainv---
	multMatrixMatrix(Ainv, A, ncols, nrows, ncols, tmp1);
	multMatrixMatrix(tmp1, Ainv, ncols, ncols, nrows, tmp2);
	error=0;
	for(int i=nrows*ncols-1;i>=0;--i){
		error+=SQR(tmp2[i]-Ainv[i]);
	}
	error=sqrt(error);
	cerr<<error<<endl;

	delete[] Ainv;
	delete[] tmp0;
	delete[] tmp1;
	delete[] tmp2;
}

//returns in M the eigenvectors (each row is one eigenvector)
int symmetricEigenDecomposition(double *M, double *eigenvalues, int N){
	//double *Mcopy=new double[N*N];
	//memcpy(Mcopy, M, sizeof(double)*N*N);
	integer n=N, lda = N, info, lwork;
	double wkopt;
	double* work;
	/* Local arrays */
	lwork = -1;
	dsyev_( "Vectors", "Upper", &n, M, &lda, eigenvalues, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
	work = new double[lwork];
	/* Solve eigenproblem */
	dsyev_( "Vectors", "Upper", &n, M, &lda, eigenvalues, work, &lwork, &info );
	/* Check for convergence */
	if( info > 0 ) {
		cerr<<"Eigendecomposition failed."<<endl;
		return info;
	}
	//-----test----
	/*double *x=new double[N];
	double *y0=new double[N];
	double *y1=new double[N];
	for(int i=0;i<N;++i){
		x[i]=M[i];
	}
	multMatrixVector<double>(Mcopy, x, N, N, y0);
	multVectorScalar(x,eigenvalues[0], N, y1);
	delete[] Mcopy;
	delete[] x;
	delete[] y0;
	delete[] y1;*/
	//-------------
	
	delete[] work;
	return 0;
}


/*
Constraints: A and B are symmetric, B is positive definite
returns the eigenvectors in A and eigenvalues in W
reference:http://www.nag.com/numeric/fl/nagdoc_fl23/xhtml/F08/f08saf.xml
*/
void clapack_generalized_symmetric_eigenvalue_decomposition(double *A, double *B, long int n, double *&W){
	/*
	itype=1
		Az=lambda*Bz.
	itype=2
		ABz=lambda*z.
	itype=3
		BAz=lambda*z.
	*/
	integer itype=1;
	/*
	jobz='N'
		compute eigenvalues only
	jobz='V'
		compute eigenvalues and eigenvectors
	*/
	char jobz='V';
	/*
	uplo='U'
		the upper triangles of A and B are stored
	uplo='L'
		the lower triangles of A and B are stored
	*/
	char uplo='U';
	/*
	N=n
	*/
	integer N=n;

	integer workSize=3*n-1;
	if(workSize<1){
		workSize=1;
	}
	double *work=new double[workSize];
	integer info;
	dsygv_(&itype, &jobz, &uplo, &N, A, &N/*LDA*/, B, &N/*LDB*/, W, work, &workSize, &info);
	delete[] work;
}

/*
	Reference:http://www.nag.com/numeric/fl/nagdoc_fl23/xhtml/F08/f08waf.xml
	Eigenvalues:
		output: alphar = real part of the eigenvalues
		output: alphai = imaginary part of the eigenvalues
		output: beta   = the generalized eigenvalues are: lambda[k]=(alphar[k]+i*alphai[k])/beta[k]
	Eigenvectors:
		output: leftEVectors = left eigenvectors
		output: rightEVectors = right eigenvectors
*/
void clapack_generalized_eigenvalue_decomposition(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors){
	/*
	jobvl='N'
		do not compute the left generalized eigenvectors
	jobvl='V'
		compute the left generalized eigenvectors
	*/
	char jobvl='V';
	/*
	jobvr='N'
		do not compute the right generalized eigenvectors
	jobvr='V'
		compute the right generalized eigenvectors
	*/
	char jobvr='V';
	/*
	N=n
	*/
	integer N=n;

	integer lwork=-1;
	
	
	integer info=0;
	//---transpose matrices---
	double *BB=new double[n*n];
	double *AA=new double[n*n];
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			BB[i*n+j]=B[j*n+i];
			AA[i*n+j]=A[j*n+i];

		}
	}
	//------------------------
	
	double WORKDUMMY=0;
	dggev_(&jobvl, &jobvr, &N, AA, &N, BB, &N, 
		alphar, alphai, beta, 
		leftEVectors,  &N, 
		rightEVectors, &N, 
		&WORKDUMMY, &lwork, &info);
	if(info!=0){
		info=0;
	}
	lwork=int(WORKDUMMY+32);
	double *work=new double[lwork];
	

	dggev_(&jobvl, &jobvr, &N, AA, &N/*LDA*/, BB, &N/*LDB*/, 
		alphar, alphai, beta, 
		leftEVectors,  &N/*LD_VL*/, 
		rightEVectors, &N/*LD_VR*/, 
		work, &lwork, &info);
	if(info!=0){
		info=0;
	}
	delete[] AA;
	delete[] BB;
	delete[] work;
}


void clapack_GEP_direct(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors){

	double *Ainv=new double[n*n];
	/*computeInverse(A, Ainv, n);
	multMatrixMatrix<double>(Ainv, B, n, A);*/
	saveMatrix<double>(A,n,n,"A.txt");
	/*
	jobvl='N'
		do not compute the left generalized eigenvectors
	jobvl='V'
		compute the left generalized eigenvectors
	*/
	char jobvl='V';
	/*
	jobvr='N'
		do not compute the right generalized eigenvectors
	jobvr='V'
		compute the right generalized eigenvectors
	*/
	char jobvr='V';
	/*
	N=n
	*/
	integer N=n;

	integer lwork=-1;
	integer info=0;
	double WORKDUMMY=0;
	double *Aorig=new double[n*n];
	memcpy(Aorig, A, sizeof(double)*n*n);
	
	dgeev_(&jobvl,&jobvr, &N, A, &N, alphar, alphai, leftEVectors, &N, rightEVectors, &N,&WORKDUMMY, &lwork, &info);
	lwork=int(WORKDUMMY+32);
	double *work=new double[lwork];
	dgeev_(&jobvl,&jobvr, &N, A, &N, alphar, alphai, leftEVectors, &N, rightEVectors, &N,work, &lwork, &info);
	for(int i=0;i<n;++i){
		beta[i]=1;
	}

	//test
	double *v=new double[n];
	double *vMult=new double[n];
	for(int k=0;k<n;++k)if(fabs(alphai[k])<EPSILON){
		for(int i=0;i<n;++i){
			v[i]=rightEVectors[k*n+i];
		}
		multMatrixVector<double>(Aorig, v, n, n,vMult);
		multVectorScalar<double>(v,alphar[k],n, v);
		double error=sqrDistance<double>(v, vMult, n);
		cout<<error<<endl;
		
	}
	
}



void test_clapack_generalized_eigenvalue_decomposition(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors){
	//test only the eigenvectors with real, finite eigenvalues
	double *left=new double[n];
	double *right=new double[n];
	double *v=new double[n];

	double valid=0;
	for(int k=0;k<n;++k)if(fabs(beta[k])>EPSILON){//finite eigenvalue
		if(fabs(alphai[k])<EPSILON){//real eigenvalue
			double lambda=alphar[k]/beta[k];
			//test right eigenvectors:
			for(int i=0;i<n;++i){
				v[i]=rightEVectors[(k*n)+i];
			}
			multMatrixVector<double>(A, v, n, n, left);
			multMatrixVector<double>(B, v, n, n, right);
			multVectorScalar<double>(left, beta[k], n, left);
			multVectorScalar<double>(right, alphar[k], n, right);
			double sqrRightError=sqrDistance(left, right, n);
			
			//test left eigenvectors:
			for(int i=0;i<n;++i){
				v[i]=leftEVectors[(k*n)+i];
			}
			multVectorMatrix<double>(v, A, n, n, left);
			multVectorMatrix<double>(v, B, n, n, right);
			multVectorScalar<double>(left, beta[k], n, left);
			multVectorScalar<double>(right, alphar[k], n, right);
			double sqrLeftError=sqrDistance(left, right, n);
			//cout<<valid<<":\tRight:\t"<<sqrRightError<<"\tLeft:\t"<<sqrLeftError<<endl;
			++valid;
		}
	}
	delete[] left;
	delete[] right;
	delete[] v;

}

void forceNonnegativeTensor(double *coefs, double *eigenVectors, double *eigenValues){
	eigenVectors[0]=coefs[0]; eigenVectors[1]=coefs[3]; eigenVectors[2]=coefs[4];
	eigenVectors[3]=coefs[3]; eigenVectors[4]=coefs[1]; eigenVectors[5]=coefs[5];
	eigenVectors[6]=coefs[4]; eigenVectors[7]=coefs[5]; eigenVectors[8]=coefs[2];

	symmetricEigenDecomposition(eigenVectors,eigenValues,3);
	eigenValues[0]=MAX(eigenValues[0], 0);
	eigenValues[1]=MAX(eigenValues[1], 0);
	eigenValues[2]=MAX(eigenValues[2], 0);
	double TT_D[9]={
		eigenVectors[0]*eigenValues[0], eigenVectors[3]*eigenValues[1], eigenVectors[6]*eigenValues[2],
		eigenVectors[1]*eigenValues[0], eigenVectors[4]*eigenValues[1], eigenVectors[7]*eigenValues[2],
		eigenVectors[2]*eigenValues[0], eigenVectors[5]*eigenValues[1], eigenVectors[8]*eigenValues[2],
	};
	multMatrixMatrix(TT_D,eigenVectors, 3, TT_D);
	coefs[0]=TT_D[0];
	coefs[1]=TT_D[4];
	coefs[2]=TT_D[8];
	coefs[3]=TT_D[1];
	coefs[4]=TT_D[2];
	coefs[5]=TT_D[5];
}


void computeSVD(double *_M, long n, long m, double *U, double *S, double *VT){
	double *M=new double[m*n];
	double *UT=new double[n*n];
	double *V=new double[n*n];
	transpose<double>(_M, n, m, M);
	
	long lda = n, ldu = n, ldvt = m, info, lwork;
	double wkopt;
	double* work;
	lwork = -1;
	dgesvd_( "All", "All", &n, &m, M, &lda, S, UT, &ldu, V, &ldvt, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
	work = (double*)malloc( lwork*sizeof(double) );
	/* Compute SVD */
	dgesvd_( "All", "All", &n, &m, M, &lda, S, UT, &ldu, V, &ldvt, work, &lwork, &info );
	transpose<double>(UT, m,n,U);
	transpose<double>(V, m,m, VT);
	delete[] M;
	delete[] UT;
	delete[] V;

}

void testSVD(void){
	const long n=6;
	const long m=5;
	double at[m*n] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
	};
	double a[n*m];
	transpose<double>(at, m, n, a);

	double u[n*m];
	double vt[m*m];
	double s[m];
	double b[n*m];
	computeSVD(a, n, m, u, s, vt);
	//----test decomp----
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			u[i*m+j]*=s[j];
		}
	}
	multMatrixMatrix<double>(u,vt,n,m,m,b);
	double error=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			error+=fabs(a[i*m+j]-b[i*m+j]);
		}
	}
	cerr<<error<<endl;

}
