#ifndef STATISTICSUTILS_H
#define STATISTICSUTILS_H
#include <set>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "linearalgebra.h"
#include "macros.h"
double generateWeibull(double uniform_sample/*uniform [0,1]sample*/, double alpha/*centering parameter*/, double beta/*scale parameter*/, double gamma/*shape parameter*/);
double generateRician(double nu=0, double sigma=1);

//returns the largest index i such that v[i]<=thr. Assumes v is sorted in non-decreasing order
int binary_search_verifRate(std::vector<double> &v, double thr);

int select(int nth, std::set<int>&forbidden);
void randomSampleReplacement(int samples, int n, int *selected);
void randomSampleNoReplacement(int samples, int n, int *selected);
void randomPermutation(int n, int *perm);
double evaluateBiasedQuadraticForm(double *Q, double *v, double *bias, int n);
double computeLR(double **data, int n, int m, int *groupIndex, int numGroups);
void computeBetweenStatistics(double **x, int numSamples, int numVertices, int numFeatures, double *&mean, double *&S);
void computeNeutralLogLikelihood(double **x, int *y, int numSamples, int numVertices, int numFeatures, double *&llNeutral);
void computeCovariance(double **x, int n, int m, double *s);
void computeMean(double **data, int n, int m, double *mean);
void computeCentroids(double **data, int n, int m, int *labels, int numClasses, double **means, int *groupSize);
bool ReadMask(std::string filename, std::vector<std::vector<unsigned char> > &mask);
void computeROCData(double *D, int *labels, int n, FILE *F);
void computeROCData(std::vector<std::pair<double, int> > &scores, int n, int numSubjects, FILE *F);
void computeROCThreshold(std::vector<std::pair<double, int> > &scores, int n, int numSubjects, double maxFAR, double &threshold, double &verificationRate);
double computeVerificationRate(double *D, int n, std::vector<std::vector<unsigned char> > &mask, double falseAcceptanceRate);
void computeVerificationScores(double *D, std::vector<int> &ids, std::vector<std::pair<double, int> > &scores, int &numSubjects);
void printROCData(std::vector<double > &matchScores, std::vector<double> &nonMatchScores, FILE *F);
void printROCData(std::vector<double > *matchScores, std::vector<double> *nonMatchScores, int numCurves, FILE *F);
void ENormalization(double *GG, double *PG, int ng, int np, double *&embedded, int &dim);
int numCategories(int *v, int n);
int unique(int *v, int n, int *&_unique);
int computeDistance(double **data, int n, int m, double *D);
double uniform(double a=0, double b=1);
double gaussianSample(double mean=0, double sigma=1);

template<class T>T getMean(std::vector<T> &v){
	double sum=0;
	for(unsigned i=0;i<v.size();++i){
		sum+=v[i];
	}
	return (T)(sum/v.size());
}

template<class T>T getStdev(std::vector<T> &v){
	double sum=0;
	double sqsum=0;
	for(unsigned i=0;i<v.size();++i){
		sum+=v[i];
		sqsum+=SQR(v[i]);
	}
	sum/=v.size();
	sqsum/=v.size();
	return (T)sqrt(double(sqsum - sum*sum));
}

/**
given a set of labeled objects (class T) and the number of objects per label, sort the lists of objects and
labels so that the objects of the classes with cardinality less than k are at the end of the the arrays
*/
template<class T> int dropSmallGroups(std::vector<T*> &samples, int *labels, int *groupSize, int k ){
	int n=samples.size();
	int i=0;
	int j=n-1;
	while(i<j){
		if(k<=groupSize[labels[i]]){
			++i;
		}else if(groupSize[labels[j]]<k){
			--j;
		}else{
			T *temp=samples[i];
			samples[i]=samples[j];
			samples[j]=temp;
			int itemp=labels[i];
			labels[i]=labels[j];
			labels[j]=itemp;
		}
	}
	if(k<=groupSize[labels[i]]){//valid
		return i+1;
	}
	return i;
}

template <class T> T median(T *v, int n){
	sort(v, v+n);
	return v[n/2];
}

template<class T> void getBounds(T *v, int n, T &minVal, T &maxVal){
	if(n<=0){
		return;
	}
	minVal=v[0];
	maxVal=v[0];
	for(int i=1;i<n;++i){
		if(v[i]<minVal){
			minVal=v[i];
		}
		if(maxVal<v[i]){
			maxVal=v[i];
		}
	}
}

template<class T> void computeBounds(T **v, int n, int m, T *minVal, T *maxVal){
	if((n<=0) || (m<=0)){
		return;
	}
	for(int j=0;j<m;++j){
		minVal[j]=v[0][j];
		maxVal[j]=v[0][j];
		for(int i=1;i<n;++i){
			if(v[i][j]<minVal[j]){
				minVal[j]=v[i][j];
			}
			if(maxVal[j]<v[i][j]){
				maxVal[j]=v[i][j];
			}
		}
	}
}

template<class  T> void normlaizeProblem(T **v, int n, int m, T *minVal, T *maxVal){
	for(int j=0;j<m;++j){
		double diff=maxVal[j]-minVal[j];
		if(diff<EPSILON){
			continue;
		}
		for(int i=0;i<n;++i){
			v[i][j]=(v[i][j]-minVal[j])/diff;
		}
	}
}

bool ReadMask(std::string filename, std::vector<std::vector<unsigned char> > &mask);

int convertToConsecutiveIds(std::vector<int> &ids);
int convertToConsecutiveIds(int *ids, int n);
void permuteRelativePosition(int *labels, int n, int *labPerm);
template <class T> void applyPermutation(T *data, int *perm, int n, T* destination){
	for(int i=0;i<n;++i){
		destination[i]=data[perm[i]];
	}
}




#endif
