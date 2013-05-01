#include "clustering.h"
#include <string.h>
#include <utility>
#include <math.h>
#include <set>
#include "macros.h"
#include <stdlib.h>
#include <iostream>
#include "linearalgebra.h"
#include "geometryutils.h"
using namespace std;

void getLabelsFromProbs(double *p, int n, int k, int *lab){
	for(int i=0;i<n;++i){
		double maxVal=p[i*k];
		lab[i]=0;
		for(int j=1;j<k;++j){
			if(maxVal<p[i*k+j]){
				maxVal=p[i*k+j];
				lab[i]=j;
			}
		}
	}
}

double dist(double *x, double *y, int n){
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=SQR(x[i]-y[i]);
	}
	return sum;
}

//JOOG: D is the cummulative distribution, the size of the array is n+1
double selectFromDistribution(double *D, int n){
	if(D[n]<=1/double(RAND_MAX)){//degenerate distribution
		return rand()%n;
	}
	double r=D[n]*(rand()%RAND_MAX+1)/double(RAND_MAX);//remember: D[i]=sum{d[j], j<i}, where d is the density function: D is of size n+1
	//binary search
	int a=0,b=n;
	while(a<b-1){
		int m=(a+b+1)/2;
		if(r<D[m]){
			b=m;
		}else{
			a=m;
		}
	}
	return b-1;
}

double weightedAngularDistance(double alpha0, double *d0, double alpha1, double *d1, int dim){
	double *P=new double[dim*dim];
	double *p=P;
	alpha0*=alpha0;
	alpha1*=alpha1;
	for(int i=0;i<dim;++i){
		for(int j=0;j<dim;++j, ++p){
			*p=alpha0*d0[i]*d0[j]+alpha1*d1[i]*d1[j];
		}
	}
	double *lambda=new double[dim];
	double *u=new double[dim];
	symmetricEigenDecomposition(P,lambda, dim);
	memcpy(u, &P[(dim-1)*dim], dim*sizeof(double));
	double theta0=getAbsAngle(d0,u,dim);
	double theta1=getAbsAngle(d1,u,dim);
	delete[] P;
	delete[] u;
	delete[] lambda;
	return MIN(theta0, theta1);
}


double combineDirections(double alpha0, double *d0, double alpha1, double *d1, int dim, double *u){
	double *P=new double[dim*dim];
	double *p=P;
	alpha0*=alpha0;
	alpha1*=alpha1;
	for(int i=0;i<dim;++i){
		for(int j=0;j<dim;++j, ++p){
			*p=alpha0*d0[i]*d0[j]+alpha1*d1[i]*d1[j];
		}
	}
	double *lambda=new double[dim];
	symmetricEigenDecomposition(P,lambda, dim);
	memcpy(u, &P[(dim-1)*dim], dim*sizeof(double));
	double theta0=getAbsAngle(d0,u,dim);
	double theta1=getAbsAngle(d1,u,dim);
	delete[] P;
	delete[] lambda;
	return MIN(theta0, theta1);
}


//initialize means using KMeans++ algorithm
void initializeAngular(double *data, double *alpha, int n, int dim, double *&means, int k){
	means=new double[k*dim];
	double *alphaMeans=new double[k];
	double *D=new double[n+1];
	int sel=0;
	memcpy(&means[0],&data[sel*dim],sizeof(double)*dim);
	alphaMeans[0]=alpha[sel];
	for(int c=1;c<k;++c){
		//compute the angular distance from each point to the nearest centroid
		D[0]=0;
		double maxDist=0;
		int bestIndex=-1;
		for(int i=0;i<n;++i){
			double minDist=-1;
			for(int j=0;j<c;++j){
				double d=getAbsAngleDegrees(&means[j*dim], &data[i*dim], dim);
				//double d=weightedAngularDistance(alphaMeans[j], &means[j*dim], alpha[i], &data[i*dim],dim);
				if((minDist<0) || (d<minDist)){
					minDist=d;
				}
			}
			if(maxDist<minDist){
				maxDist=minDist;
				bestIndex=i;
			}
			D[i+1]=D[i]+minDist;//compute the cummilative distribution
		}
		//sel=selectFromDistribution(D, n);
		sel=bestIndex;
		memcpy(&means[c*dim],&data[sel*dim],sizeof(double)*dim);
		alphaMeans[c]=alpha[sel];
	}
	delete[] D;
	delete[] alphaMeans;
}

//initialize means using KMeans++ algorithm
void initialize(double *data, int n, int dim, double *&means, int k){
	means=new double[k*dim];
	double *D=new double[n+1];
	//int sel=rand()%n;
	int sel=0;
	memcpy(&means[0],&data[sel*dim],sizeof(double)*dim);
	for(int c=1;c<k;++c){
		//compute the squared distance from each point to the nearest centroid
		D[0]=0;
		for(int i=0;i<n;++i){
			double minDist=-1;
			for(int j=0;j<c;++j){
				double d=dist(&means[j*dim], &data[i*dim], dim);
				if((minDist<0) || (d<minDist)){
					minDist=d;
				}
			}
			D[i+1]=D[i]+minDist;//compute the cummilative distribution
		}
		sel=selectFromDistribution(D, n);
		memcpy(&means[c*dim],&data[sel*dim],sizeof(double)*dim);
	}
}



void computeMostCorrelatedVector(double *X, double *alpha, int n, int m, int *I, int nn, double *u){
	double *XX=new double[m*m];
	for(int i=0;i<m;++i){
		for(int j=0;j<m;++j){
			double &s=XX[i*m+j];
			s=0;
			for(int k=0;k<nn;++k){
				s+=alpha[I[k]]*alpha[I[k]]*X[I[k]*m+i]*X[I[k]*m+j];
			}
		}
	}
	double *lambda=new double[m];
	symmetricEigenDecomposition(XX,lambda, m);
	memcpy(u, &XX[(m-1)*m], m*sizeof(double));
	delete[] XX;
	delete[] lambda;
	
	normalize<double>(u,m);
}

void computeMostCorrelatedVector(double *X, double *alpha, double *beta, int n, int m, int numClasses, int classIndex, double mu, double *u){
	double *XX=new double[m*m];
	for(int i=0;i<m;++i){
		for(int j=0;j<m;++j){
			double &s=XX[i*m+j];
			s=0;
			double *xi=&X[i];
			double *xj=&X[j];
			double *betaj=&beta[classIndex];
			for(int k=0;k<n;++k, xi+=m, xj+=m, betaj+=numClasses){
				double factor=pow(*betaj, mu);
				s+=factor*alpha[k]*alpha[k]**xi**xj;
			}
		}
	}
	double *lambda=new double[m];
	symmetricEigenDecomposition(XX,lambda, m);
	memcpy(u, &XX[(m-1)*m], m*sizeof(double));
	delete[] XX;
	delete[] lambda;
	normalize<double>(u,m);
}

double recomputeAngularMeans(double *data, double *alpha, int n, int dim, double *means, double *P, int k, double m){
	if(m<=0){
		return -1;
	}
	if(fabs(m-1)<1e-9){
		double *prev=new double[k*dim];
		memcpy(prev, means, sizeof(double)*k*dim);
		memset(means, 0, sizeof(double)*k*dim);
		int *counts=new int[k];
		memset(counts, 0, sizeof(int)*k);
		int *I=new int[k*n];
		for(int i=0;i<n;++i){
			double *p=&P[i*k];
			int sel=0;
			for(int j=1;j<k;++j){
				if(p[sel]<p[j]){
					sel=j;
				}
			}
			I[sel*n+counts[sel]]=i;
			counts[sel]++;
		}
		for(int i=0;i<k;++i)if(counts[i]>0){
			computeMostCorrelatedVector(data, alpha, n, dim, &I[i*n], counts[i], &means[i*dim]);
		}
		double err=0;
		for(int i=0;i<k;++i){
			double angle=getAbsAngleDegrees(&means[i*dim], &prev[i*dim],dim);
			err+=angle;
		}
		err/=k;
		delete[] prev;
		delete[] counts;
		delete[] I;
		return err;
	}else{
		cerr<<"Warning: executing angular-fuzzy-KMeans"<<endl;
		double *prev=new double[k*dim];
		memcpy(prev, means, sizeof(double)*k*dim);
		memset(means, 0, sizeof(double)*k*dim);
		for(int i=0;i<k;++i){
			computeMostCorrelatedVector(data, alpha, P, n, dim, k, i, m, &means[i*dim]);
		}
		double err=0;
		for(int i=0;i<k;++i){
			double angle=getAbsAngleDegrees(&means[i*dim], &prev[i*dim],dim);
			err+=angle;
		}
		err/=k;
		delete[] prev;
		return err;
	}
}

double recomputeMeans(double *data, int n, int dim, double *means, double *P, int k, int m){
	if(m<=0){
		return -1;
	}
	if(m==1){
		double *prev=new double[k*dim];
		memcpy(prev, means, sizeof(double)*k*dim);
		memset(means, 0, sizeof(double)*k*dim);
		int *counts=new int[k];
		memset(counts, 0, sizeof(int)*k);
		for(int i=0;i<n;++i){
			double *p=&P[i*k];
			int sel=0;
			for(int j=1;j<k;++j){
				if(p[sel]<p[j]){
					sel=j;
				}
			}
			counts[sel]++;
			double *mean=&means[sel*dim];
			double *x=&data[i*dim];
			for(int j=0;j<dim;++j){
				mean[j]+=x[j];
			}
		}
		for(int j=0;j<k;++j){
			double *mean=&means[j*dim];
			for(int i=0;i<dim;++i)if(counts[j]>0){
				mean[i]/=counts[j];
			}
		}
		double variation=0;
		for(int i=0;i<dim*k;++i){
			variation+=fabs(prev[i]-means[i]);
		}
		delete[] prev;
		return variation/(k*dim);
	}
	double *prev=new double[dim];
	double variation=0;
	for(int j=0;j<k;++j){
		double *mu=&means[j*dim];
		memcpy(prev, mu, sizeof(double)*dim);
		memset(mu, 0, sizeof(double)*dim);
		double sum=0;
		for(int i=0;i<n;++i){
			double *d=&data[i*dim];
			double pj=pow(P[i*k+j], m);
			sum+=pj;
			for(int r=0;r<dim;++r){
				mu[r]+=d[r]*pj;
			}
		}
		for(int r=0;r<dim;++r){
			mu[r]/=sum;
			variation+=fabs(mu[r]-prev[r]);
		}
	}
	delete[] prev;
	return variation/(k*dim);
}


double recomputeAngularProbs(double *data, int n, int dim, double *means, double *P, int k, double m){
	if(m<=0){
		return -1;
	}
	if(fabs(m-1)<1e-9){
			double *prev=new double[n*k];
		memcpy(prev, P, sizeof(double)*n*k);
		for(int i=0;i<n;++i){
			int sel=0;
			double *mean=&means[sel*dim];
			double *x=&data[i*dim];
			double *p=&P[i*k];
			double selDist=getAbsAngleDegrees(x, mean, dim);
			for(int j=1;j<k;++j){
				mean=&means[j*dim];
				double opc=getAbsAngleDegrees(x, mean, dim);
				if(opc<selDist){
					sel=j;
					selDist=opc;
				}
			}
			memset(p,0,sizeof(double)*k);
			p[sel]=1;
		}
		double variation=0;
		for(int i=0;i<n*k;++i){
			variation+=fabs(P[i]-prev[i]);
		}
		delete[] prev;
		return variation/(n*k);
	}
	cerr<<"Warning: executing angular-fuzzy-KMeans"<<endl;
	double *prev=new double[k];
	double variation=0;
	for(int i=0;i<n;++i){
		double *p=&P[i*k];
		memcpy(prev, p, sizeof(double)*k);
		double *d=&data[i*dim];
		double sum=0;
		for(int j=0;j<k;++j){
			double *mu=&means[j*dim];
			double dij=dotProduct(d, mu, dim);
			dij*=dij;
			p[j]=1.0/(1e-6+dij);
			p[j]=pow(p[j], 1.0/(m-1.0));
			sum+=p[j];
		}
		for(int j=0;j<k;++j){
			p[j]/=sum;
			variation+=fabs(p[j]-prev[j]);
		}
	}
	delete[] prev;
	return variation/(n*k);
}

double recomputeProbs(double *data, int n, int dim, double *means, double *P, int k, int m){
	if(m<=0){
		return -1;
	}
	if(m==1){
		double *prev=new double[n*k];
		memcpy(prev, P, sizeof(double)*n*k);
		for(int i=0;i<n;++i){
			int sel=0;
			double *mean=&means[sel*dim];
			double *x=&data[i*dim];
			double *p=&P[i*k];
			double selDist=dist(x, mean, dim);
			for(int j=1;j<k;++j){
				mean=&means[j*dim];
				double opc=dist(x, mean, dim);
				if(opc<selDist){
					sel=j;
					selDist=opc;
				}
			}
			memset(p,0,sizeof(double)*k);
			p[sel]=1;
		}
		double variation=0;
		for(int i=0;i<n*k;++i){
			variation+=fabs(P[i]-prev[i]);
		}
		delete[] prev;
		return variation/(n*k);
	}
	double *prev=new double[k];
	double variation=0;
	for(int i=0;i<n;++i){
		double *p=&P[i*k];
		memcpy(prev, p, sizeof(double)*k);
		double *d=&data[i*dim];
		double sum=0;
		for(int j=0;j<k;++j){
			double *mu=&means[j*dim];
			p[j]=1.0/(1e-6+dist(d, mu, dim));
			p[j]=pow(p[j], 1.0/(m-1.0));
			sum+=p[j];
		}
		for(int j=0;j<k;++j){
			p[j]/=sum;
			variation+=fabs(p[j]-prev[j]);
		}
	}
	delete[] prev;
	return variation/(n*k);
}

/*una iteracion de Fuzzy K-Means: n puntos en dimension d, clasificados en k 
categorias usando m como exponente de los pesos*/
pair<double, double> iterateAngularFuzzyKMeans(double *data, double *alpha, int n, int dim, double *means, double *P, int k, double m){
	pair<double, double> retVal;
	retVal.first=recomputeAngularMeans(data, alpha, n, dim, means, P, k, m);
	retVal.second=recomputeAngularProbs(data, n, dim, means, P, k, m);
	return retVal;
}


/*una iteracion de Fuzzy K-Means: n puntos en dimension d, clasificados en k 
categorias usando m como exponente de los pesos*/
pair<double, double> iterateFuzzyKMeans(double *data, int n, int dim, double *means, double *P, int k, int m){
	pair<double, double> retVal;
	retVal.first=recomputeMeans(data, n, dim, means, P, k, m);
	retVal.second=recomputeProbs(data, n, dim, means, P, k, m);
	return retVal;
}

double angularFuzzyKMeansEnergy(double *data, double *alpha, int n, int dim, int K, double exponent, double *&means, double *&probs){
	double energy=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<K;++j){
			double factorBeta=pow(probs[i*K+j],exponent);
			double factorProd=alpha[i]*dotProduct(&data[dim*i],&means[dim*j], dim);
			factorProd*=factorProd;
			energy+=factorBeta*factorProd;
		}
	}
	return energy;
}

int angularFuzzyKMeans(double *data, double *alpha, int n, int dim, int K, double exponent, int maxIter, double *&means, double *&probs, bool initKMPP){	
	if(initKMPP){//initialize with K-means ++
		initializeAngular(data, alpha, n, dim, means, K);
	}else{//initialize sequential
		means=new double[K*dim];
		for(int i=0;i<K;++i){
			memcpy(&means[i*dim], &data[i*dim], sizeof(double)*dim);
		}
	}
	probs=new double[n*K];
	memset(probs,0,sizeof(double)*n*K);
	//initialize probabilities
	recomputeAngularProbs(data, n, dim, means, probs, K, exponent);
	//iterate until convergence
	double toleranceProbs=1e-4;
	double toleranceMeans=1e-3;
	//double energy=angularFuzzyKMeansEnergy(data, alpha, n, dim, K, exponent, means, probs);
	//cerr<<"0:\t"<<energy<<endl;
	for(int iter=1;iter<=maxIter;++iter){
		pair<double, double> variation=iterateAngularFuzzyKMeans(data, alpha, n, dim, means, probs, K, exponent);
		//energy=angularFuzzyKMeansEnergy(data, alpha, n, dim, K, exponent, means, probs);
		//cerr<<iter<<"\t"<<energy<<endl;
		//cerr<<iter<<":"<<variation.first<<"\t"<<variation.second<<endl;
		if(variation.first<toleranceMeans){
			break;
		}
		if(variation.second<toleranceProbs){
			break;
		}
	}
	set<int> clusters;
	for(int i=0;i<n;++i){
		int sel=0;
		for(int j=1;j<K;++j){
			if(probs[i*K+j]>probs[i*K+sel]){
				sel=j;
			}
		}
		clusters.insert(sel);
	}
	return clusters.size();
}

void fuzzyKMeans(double *data, int n, int dim, int K, double exponent, int maxIter, double *&means, double *&probs, bool initKMPP){	
	if(initKMPP){//initialize with K-means ++
		initialize(data, n, dim, means, K);
	}else{//initialize sequential
		means=new double[K*dim];
		for(int i=0;i<K;++i){
			memcpy(&means[i*dim], &data[i*dim], sizeof(double)*dim);
		}
	}
	probs=new double[n*K];
	memset(probs,0,sizeof(double)*n*K);
	//initialize probabilities
	recomputeProbs(data, n, dim, means, probs, K, exponent);
	//iterate until convergence
	double toleranceProbs=1e-4;
	double toleranceMeans=1e-3;
	for(int iter=1;iter<=maxIter;++iter){
		pair<double, double> variation=iterateFuzzyKMeans(data, n, dim, means, probs, K, exponent);
		//cerr<<iter<<":"<<variation.first<<"\t"<<variation.second<<endl;
		if(variation.first<toleranceMeans){
			break;
		}
		if(variation.second<toleranceProbs){
			break;
		}
	}
}
	