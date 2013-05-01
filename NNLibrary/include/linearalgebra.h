#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H
#include <stdio.h>
#include "macros.h"

void testSVD(void);

void computeSVD(double *_M, long n, long m, double *U, double *S, double *VT);
typedef void (*VectorFunction)(double *x, int n, double *y, int m, void *data);
typedef double (*ScalarFunction)(double *x, int n, void *data);
int solveLinearSystem(double *A, double *b, double *x, int n);

template<class T> void transpose(T *A, int n, int m, T* AT){
	if(A!=AT){
		for(int i=0;i<n;++i){
			for(int j=0;j<m;++j){
				AT[j*n+i]=A[i*m+j];
			}
		}
	}else{
		double *Atmp=new double[n*m];
		memcpy(Atmp, A, sizeof(double)*n*m);
		for(int i=0;i<n;++i){
			for(int j=0;j<m;++j){
				AT[j*n+i]=Atmp[i*m+j];
			}
		}
		delete[] Atmp;
	}
	
}

template<class T> double sqrDistance(T *x, T*y, int n){
	double s=0;
	for(int i=0;i<n;++i){
		s+=SQR(x[i]-y[i]);
	}
	return s;
}

template<class T> void multVectorScalar(T *v, T t, int n, T *y){
	for(int i=0;i<n;++i){
		y[i]=t*v[i];
	}
}

template<class T> void sumVectorVector(double *a, double *b, int len, double *dest){
	for(int i=0;i<len;++i){
		dest[i]=a[i]+b[i];
	}
}

template<class T> void linCombVector(double *a, double factor, double *b, int len, double *dest){
	for(int i=0;i<len;++i){
		dest[i]=a[i]*factor+b[i];
	}
}

template<class T> void linCombVector(double *a, double factorA, double *b, double factorB, int len, double *dest){
	for(int i=0;i<len;++i){
		dest[i]=a[i]*factorA+b[i]*factorB;
	}
}

template<class T>void setIdentity(T *I, int n){
	memset(I,0, sizeof(T)*n*n);
	for(int i=n*n-1;i>=0;i-=(n+1)){
		I[i]=1;
	}
}

template<class T> void multVectorMatrix(T *x, T *A, int n, int m, T *y){
	double *tmp;
	if(y==x){
		tmp=new T[m];
	}else {
		tmp=y;
	}

	for(int i=0;i<m;++i){
		double s=0;
		for(int j=0;j<n;++j){
			s+=x[j]*A[j*m+i];
		}
		tmp[i]=s;
	}

	if(y==x){
		memcpy(y, tmp, m*sizeof(T));
		delete[] tmp;
	}
}


template<class T> void multMatrixMatrix(T *A, T *B, int n, T *C){
	double *tmp;
	if(C==A){
		tmp=new T[n*n];
	}else if(C==B){
		tmp=new T[n*n];
	}else{
		tmp=C;
	}

	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			double s=0;
			for(int k=0;k<n;++k){
				s+=A[i*n+k]*B[k*n+j];
			}
			tmp[i*n+j]=s;
		}
	}


	if(C==A){
		memcpy(A, tmp, n*n*sizeof(T));
		delete[] tmp;
	}else if(C==B){
		memcpy(B, tmp, n*n*sizeof(T));
		delete[] tmp;
	}
}

template<class T> void multMatrixMatrix(T *A, T *B, int n, int m, int k, T *C){
	double *tmp;
	if(C==A){
		tmp=new T[n*n];
	}else if(C==B){
		tmp=new T[n*n];
	}else{
		tmp=C;
	}

	for(int i=0;i<n;++i){
		for(int j=0;j<k;++j){
			double s=0;
			for(int ii=0;ii<m;++ii){
				s+=A[i*m+ii]*B[ii*k+j];
			}
			tmp[i*k+j]=s;
		}
	}


	if(C==A){
		memcpy(A, tmp, n*k*sizeof(T));
		delete[] tmp;
	}else if(C==B){
		memcpy(B, tmp, n*k*sizeof(T));
		delete[] tmp;
	}
}

template<class T> void multMatrixVector(T *A, T *x, int n, int m, T *y){
	double *tmp;
	if(y==x){
		tmp=new T[n];
	}else {
		tmp=y;
	}

	for(int i=0;i<n;++i){
		double s=0;
		for(int j=0;j<m;++j){
			s+=A[i*m+j]*x[j];
		}
		tmp[i]=s;
	}

	if(y==x){
		memcpy(y, tmp, n*sizeof(T));
		delete[] tmp;
	}
}


template<class T> void saveMatrix(T *M, int n, int m, const char *fname){
	FILE *F=fopen(fname, "w");
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			fprintf(F, "%lf\t", M[i*m+j]);
		}
		fprintf(F, "\n");
	}
	fclose(F);
}

template<class T>T vectorNorm(T *x, int n){
	T sum=0;
	for(int i=0;i<n;++i){
		sum+=(x[i]*x[i]);
	}
	return (T)sqrt(sum);
}

template<class T>T sqrNorm(T *x, int n){
	T sum=0;
	for(int i=0;i<n;++i){
		sum+=(x[i]*x[i]);
	}
	return sum;
}

template<class T>void subtractVectors(T *x, T *y, int n, T *dest){
	for(int i=0;i<n;++i){
		dest[i]=x[i]-y[i];
	}
}

template<class T> void normalize(T *v, int n){
    double norm=0;
    for(int i=0;i<n;++i){
        norm+=SQR(v[i]);
    }
    norm=sqrt(norm);
	if(norm<EPSILON){
		return;
	}else{
		for(int i=0;i<n;++i){
			v[i]=(T)(v[i]/norm);
		}
	}
}

template<class T> void normalizeDinamicRange(T *v, int n){
    double maxVal=v[0];
	double minVal=v[0];
    for(int i=1;i<n;++i){
        maxVal=MAX(maxVal, v[i]);
		minVal=MIN(minVal, v[i]);
    }
	double diff=maxVal-minVal;
	if(diff>1e-9){
		for(int i=0;i<n;++i){
			v[i]=(v[i]-minVal)/diff;
		}
	}else{
		memset(v, 0, sizeof(T)*n);
	}
}

double evaluateQuadraticForm(double *Q, double *x, int n);

void computeInverse(double *A, long N);
int computePseudoInverseCMO(double *A, long nrows, long ncols, double *Ainv);
int computePseudoInverseRMO(double *Ain, long nrows, long ncols, double *Ainv);
void testPseudoInverseRMO(double *A, int nrows, int ncols);

int symmetricEigenDecomposition(double *M, double *eigenvalues, int n);

void forceNonnegativeTensor(double *coefs, double *eigenVectors, double *eigenValues);

void clapack_generalized_symmetric_eigenvalue_decomposition(double *A, double *B, long int n, double *&W);

void clapack_generalized_eigenvalue_decomposition(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors);
void test_clapack_generalized_eigenvalue_decomposition(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors);
void clapack_GEP_direct(double *A, double *B, long int n, 
												  double *alphar, double *alphai, double *beta,
												  double *leftEVectors, double *rightEVectors);
template<class T> void multVecScalar(T *A, int size, T x){
	for(int i=0;i<size;++i){
		A[i]*=x;
	}
}

template<class T> void multVecScalar(T *A, int size, T x, T *destination){
	for(int i=0;i<size;++i){
		destination[i]=A[i]*x;
	}
}

/**
	Multiply the given (vectorized) nxn matrix A by the vector b and puts the result in dest. The element (i,j) of a 
	vectorize nxn matrix A is A[i*n+j].
	\param	a	the (vectorized) matrix
	\param	b	the input vector
	\param	dest	the array that will contain the product Ab
	\param n	the dimension of the matrix (its nxn) and the vector
*/
template<class T> void multvec(T *A, T *b, T *dest, int n){
	T *cp;
	if(b==dest){
		cp=new T[n];
		memcpy(cp, b, sizeof(T)*n);
	}else{
		cp=b;
	}
	
	
	memset(dest, 0, sizeof(T)*n);
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			dest[i]+=A[i*n+j]*cp[j];
		}
	}
	if(b==dest){
		delete[] cp;
	}
	
}
#endif
