#include "mds.h"
#include "linearalgebra.h"
#include <vector>
#include <algorithm>
#include "carpack.h"
#include <iostream>
using namespace std;
int multidimensionalScaling(double *distMat, int n, int &k, double *&x){
	double *P=new double[n*n];
	double *D=new double[n*n];
	memset(P,0,sizeof(double)*n*n);
	int pos=0;
	for(int i=0;i<n;++i){
		P[i*(n+1)]=1;
		for(int j=0;j<n;++j,++pos){
			P[pos]-=1.0/n;
			D[pos]=-0.5*distMat[pos]*distMat[pos];
		}
	}
	double *B=new double[n*n];
	multMatrixMatrix(P,D,n,B);
	multMatrixMatrix(B,P,n,B);
	//make symmetric
	for(int i=0;i<n;++i){
		for(int j=i;j<n;++j){
			B[i*n+j]=0.5*(B[i*n+j]+B[j*n+i]);
		}
	}
	double *E=new double[n];
	symmetricEigenDecomposition(B, E, n);
	vector<pair<double, int> > v;
	for(int i=0;i<n;++i){
		if(E[i]>1e-9){//keep only positive eigenvalues
			v.push_back(make_pair(E[i],i));
		}
	}
	if(v.empty()){
		k=1;
		x=new double[n];
		memset(x,0,sizeof(double)*n);
	}else{
		sort(v.rbegin(), v.rend());//sort ascending
		if(k>v.size()){
			k=v.size();
		}
		x=new double[n*k];
		int pos=0;
		for(int i=0;i<n;++i){
			for(int j=0;j<k;++j, ++pos){
				int sel=v[j].second;
				x[pos]=B[sel*n+i]*E[sel];
			}
		}
	}
	delete[] P;
	delete[] D;
	delete[] B;
	delete[] E;
	return 0;
}

#ifdef USE_ARPACK
int multidimensionalScaling_arpack(double *distMat, int n, int &k, double *&x){
	double *P=new double[n*n];
	double *D=new double[n*n];
	memset(P,0,sizeof(double)*n*n);
	int pos=0;
	for(int i=0;i<n;++i){
		P[i*(n+1)]=1;
		for(int j=0;j<n;++j,++pos){
			P[pos]-=1.0/n;
			D[pos]=-0.5*distMat[pos]*distMat[pos];
		}
	}
	double *B=new double[n*n];
	multMatrixMatrix(P,D,n,B);
	multMatrixMatrix(B,P,n,B);
	//make symmetric
	for(int i=0;i<n;++i){
		for(int j=i;j<n;++j){
			B[i*n+j]=0.5*(B[i*n+j]+B[j*n+i]);
		}
	}
	double *eval=new double[n];
	double *evec=new double[n*k];
	
	arpack_symetric_evd(B, n, k, evec, eval);
	for(int i=0;i<k;++i){
		if(eval[i]<0){
			cerr<<"[MDS - arpack] Warning: using negative eigenvalues."<<endl;
			break;
		}
	}

	x=new double[n*k];
	pos=0;
	for(int j=0;j<n;++j){
		for(int i=0;i<k;++i, ++pos){
			x[pos]=evec[i*n+j]*eval[i];
		}
	}
	delete[] P;
	delete[] D;
	delete[] B;
	delete[] eval;
	delete[] evec;
	return 0;
}
#endif
