#include "classification.h"
#include <stdio.h>
#include <macros.h>
#include <iostream>
#include <math.h>
#include <set>
#include <algorithm>
#include "statisticsutils.h"
#include "linearalgebra.h"
#include "utilities.h"
using namespace std;
#define SWAP(a,b, temp) {temp=a;a=b;b=temp;}
//returns the number of samples selected
int randomCategorySample(double **data, int *labels, int n, double proportion){
	set<int> S;
	for(int i=0;i<n;++i){
		S.insert(labels[i]);
	}
	int numClasses=S.size();
	//============================
	int sampleSize=proportion*numClasses;
	int *selected=new int[sampleSize];
	randomSampleNoReplacement(sampleSize, numClasses, selected);
	S.clear();
	for(int i=0;i<sampleSize;++i){
		S.insert(selected[i]);
	}
	int i=0;
	int j=n-1;
	while(i<j){
		if(S.find(labels[i])!=S.end()){//i present
			++i;
		}else if(S.find(labels[j])==S.end()){//j not present
			--j;
		}else{
			int itemp;
			double *ftemp;
			SWAP(data[i], data[j], ftemp);
			SWAP(labels[i], labels[j], itemp);
			++i;
			--j;
		}
	}
	delete[] selected;
	if(S.find(labels[i])!=S.end()){//i present
		return i+1;
	}
	return i;
}



double sqrDistance(double *a, double *b, int n){
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=SQR(a[i]-b[i]);
	}
	return sum;
}

void computeDistanceMatrix(double **v, int n, int m, double *D){
	for(int i=0;i<n;++i){
		D[i*(n+1)]=0;
		for(int j=i+1;j<n;++j){
			D[i*n+j]=sqrDistance(v[i], v[j], m);
			D[j*n+i]=D[i*n+j];
		}
	}
}

void accumulateDistance(double **v, int n, int k, double *D){
	for(int i=0;i<n;++i){
		for(int j=i+1;j<n;++j){
			D[i*n+j]+=SQR(v[i][k]-v[j][k]);
			D[j*n+i]+=SQR(v[i][k]-v[j][k]);
		}
	}
}
//==========================High dimensional LDA proposed by Hua Yu and Jie Yang========================
int computeHighDimensionalLDA(double *_data, int *labels, int n, int m, int numClasses, double **&w, double *&lambda, bool maxLikelihood){
	double **data=new double*[n];
	for(int i=0;i<n;++i){
		data[i]=&_data[i*m];
	}
	int retVal=computeHighDimensionalLDA(data, labels, n, m, numClasses, w, lambda, maxLikelihood);
	delete[] data;
	return retVal;
}

int computeHighDimensionalLDA(double **data, int *labels, int n, int m, int numClasses, double **&w, double *&lambda, bool maxLikelihood){//n<<m
		bool verbose=false;
		if(verbose){
			cerr<<"(n,m,k)=("<<n<<", "<<m<<", "<<numClasses<<")"<<endl;
		}
	    
	//====compute the first k eigenvectors of the between covariance matrix====
		//********allocate memory**********
		double *bmean=new double[m];
		double *bs=NULL;
		double **wmean=new double*[numClasses];
		if(verbose){
			cerr<<"Allocating memory...";
		}
		for(int i=0;i<numClasses;++i){
			wmean[i]=new double[m];
		}
		int *groupSize=new int[numClasses];
		
		double *wEigenValues=new double[n];
		if(verbose){
			cerr<<"done."<<endl;
		}
		//******compute the global mean and the centroids. Center the data around its own centroid*****
		if(verbose){
			cerr<<"B-Mean..."<<data[0][0];
		}
		computeMean(data, n, m, bmean);
		if(verbose){
			cerr<<"done."<<endl;
			cerr<<"W-Means...";
		}
		computeCentroids(data, n, m, labels, numClasses, wmean, groupSize);
		if(verbose){
			cerr<<"done."<<endl;
			cerr<<"Btween covariance matrix...";
		}
		double **bEigenVectors=NULL;
		int rank=0;
		int nullSpaceDimension=0;
		double *bEigenValues=NULL;

		if(maxLikelihood){
			bs=new double[n*n];
			bEigenValues=new double[n];
			//*****compute the sample covariance matrix*********
			for(int i=0;i<n;++i){
				for(int j=i;j<n;++j){
					double &sum=bs[i*n+j];
					sum=0;
					for(int k=0;k<m;++k){
						sum+=(data[i][k]-bmean[k])*(data[j][k]-bmean[k]);
					}
					bs[j*n+i]=sum;
				}
			}
			symmetricEigenDecomposition(bs,bEigenValues,n);
			//----recover the eigenvectors (Turk and Pentland)----
			nullSpaceDimension=0;
			for(int i=0;i<n;++i){
				if(bEigenValues[i]<1e-6){//having problems with EPSILON=1e-7 =S
					++nullSpaceDimension;
				}
			}
			if(nullSpaceDimension==0){
				nullSpaceDimension=1;
			}
			if(verbose){
				cerr<<"done."<<endl;
			}
			rank=n-nullSpaceDimension;
			bEigenVectors=new double*[rank];
			for(int i=0;i<rank;++i){
				bEigenVectors[i]=new double[m];
				memset(bEigenVectors[i], 0, sizeof(double)*m);
				for(int j=0;j<m;++j){
					for(int k=0;k<n;++k){
						bEigenVectors[i][j]+=(data[k][j]-bmean[j])*bs[(i+nullSpaceDimension)*n+k];
					}
				}
				normalize<double>(bEigenVectors[i],m);
			}
		}else{
			bs=new double[numClasses*numClasses];
			bEigenValues=new double[numClasses];
			//*****compute the between scatter matrix*********
			for(int i=0;i<numClasses;++i){
				for(int j=i;j<numClasses;++j){
					double &sum=bs[i*numClasses+j];
					sum=0;
					for(int k=0;k<m;++k){
						sum+=(wmean[i][k]-bmean[k])*(wmean[j][k]-bmean[k]);
					}
					sum*=sqrt(double(groupSize[i]))*sqrt(double(groupSize[j]));
					bs[j*numClasses+i]=sum;
				}
			}
			if(verbose){
				for(int i=0;i<numClasses;++i){
					for(int j=0;j<numClasses;++j){
						cerr<<bs[i*numClasses+j]<<"\t";
					}
					cerr<<endl;
				}
				cerr<<"done."<<endl;
				cerr<<"Eigenvalue decomposition...";
			}
			symmetricEigenDecomposition(bs,bEigenValues,numClasses);
			if(verbose){
				cerr<<"done."<<endl;
				cerr<<"Recovering...";	
			}
			//----recover the eigenvectors (Turk and Pentland)----
			nullSpaceDimension=0;
			for(int i=0;i<numClasses;++i){
				if(fabs(bEigenValues[i])<1e-6){//having problems with EPSILON=1e-7 =S
					++nullSpaceDimension;
				}
			}
			if(nullSpaceDimension==0){
				nullSpaceDimension=1;
			}
			if(verbose){
				cerr<<"done."<<endl;
			}
			rank=numClasses-nullSpaceDimension;
			bEigenVectors=new double*[rank];
			for(int i=0;i<rank;++i){
				bEigenVectors[i]=new double[m];
				memset(bEigenVectors[i], 0, sizeof(double)*m);
				for(int j=0;j<m;++j){
					for(int k=0;k<numClasses;++k){
						bEigenVectors[i][j]+=sqrt(double(groupSize[k]))*(wmean[k][j]-bmean[j])*bs[(i+nullSpaceDimension)*numClasses+k];
					}
				}
				normalize<double>(bEigenVectors[i],m);
			}
		}
		
		
		//multiply the eigenvectors (Y) by sqrt of the inverse of the eigenvalue diagonal (D) to obtain Z (H.Yu and J.Yang "High dimensional LDA")
		for(int j=0;j<rank;++j){
			double multiplier=1.0/sqrt(bEigenValues[j+nullSpaceDimension]);
			for(int i=0;i<m;++i){
				bEigenVectors[j][i]*=multiplier;
			}
		}
		
		//compute P=Z^T Phi_w
		double **P=new double*[rank];
		for(int i=0;i<rank;++i){
			P[i]=new double[n];
			memset(P[i], 0, sizeof(double)*n);
			for(int j=0;j<n;++j){
				int sel=labels[j];
				for(int k=0;k<m;++k){
					P[i][j]+=bEigenVectors[i][k]*(data[j][k]-wmean[sel][k]);//notice that the data is already centered around the appropriate centroid
				}
			}
		}
		//compute ws=P P^T
		double *ws=new double[rank*rank];
		for(int i=0;i<rank;++i){
			for(int j=i;j<rank;++j){
				double &sum=ws[i*rank+j];
				sum=0;
				for(int k=0;k<n;++k){
					sum+=P[i][k]*P[j][k];
				}
				ws[j*rank+i]=sum;
			}
		}
		if(verbose){
			cerr<<"Eigenvalue decomposition...";
		}
		symmetricEigenDecomposition(ws,wEigenValues,rank);
		if(verbose){
			cerr<<"done."<<endl;
		}
		for(int i=0;i<rank;++i){
			double norm=0;
			for(int j=0;j<rank;++j){
				norm+=ws[i*rank+j]*ws[i*rank+j];
			}
			norm=sqrt(norm);
			for(int j=0;j<rank;++j){
				ws[i*rank+j]/=norm;
			}
		}
		if(verbose){
			cerr<<"Sphering...";
		}
		
		//====compute the LDA matrix====
		double **wEigenVectors=new double*[rank];
		for(int i=0;i<rank;++i){
			wEigenVectors[i]=new double[m];
			memset(wEigenVectors[i], 0, sizeof(double)*m);
			for(int j=0;j<m;++j){
				double &sum=wEigenVectors[i][j];
				double *weights=&ws[i*rank];
				for(int k=0;k<rank;++k){
					sum+=weights[k]*bEigenVectors[k][j];
				}
			}
			if(wEigenValues[i]>EPSILON){
				double multiplier=1.0/sqrt(wEigenValues[i]);
				for(int j=0;j<m;++j){
					wEigenVectors[i][j]*=multiplier;
				}
			}
		}
		if(verbose){
			cerr<<"done."<<endl;
		}
		
		//********Test******
		/*double *bsCheck=new double[m*m];
		memset(bsCheck, 0, sizeof(double)*m*m);
		for(int i=0;i<m;++i){
			for(int j=0;j<m;++j){
				double &target=bsCheck[i*m+j];
				for(int k=0;k<numClasses;++k){
					target+=groupSize[k]*(wmean[k][i]-bmean[i])*(wmean[k][j]-bmean[j]);
				}
				
			}
		}
		double *prod=new double[rank*m];
		memset(prod, 0, sizeof(double)*rank*m);
		for(int i=0;i<rank;++i){
			for(int j=0;j<m;++j){
				double &target=prod[i*m+j];
				for(int k=0;k<m;++k){
					target+=wEigenVectors[i][k]*bsCheck[k*m+j];
				}
			}
		}
		double *diag=new double[rank*rank];
		memset(diag, 0, sizeof(double)*rank*rank);
		for(int i=0;i<rank;++i){
			for(int j=0;j<rank;++j){
				double &target=diag[i*rank+j];
				for(int k=0;k<m;++k){
					target+=prod[i*m+k]*wEigenVectors[j][k];
				}
			}
		}
		printf("\n");
		for(int i=0;i<rank;++i){
			for(int j=0;j<rank;++j){
				printf("%0.5lf\t", diag[i*rank+j]);
			}
			printf("\n");
		}
		delete[] bsCheck;
		delete[] prod;
		delete[] diag;*/
		//******************

		w=wEigenVectors;
		lambda=wEigenValues;
		//-------------print results----------------
		/*int numVertices=m/3;
		double *norms=new double[numVertices];
		for(int i=0;i<rank;++i){
			ostringstream os;
			os<<"LDA_"<<i<<".bin";
			//for(int j=0;j<numVertices;++j){
			//	norms[j]=sqrt(SQR(wEigenVectors[i][3*j])+SQR(wEigenVectors[i][3*j+1])+SQR(wEigenVectors[i][3*j+2]));
			//}
			for(int j=0;j<numVertices;++j){
				double s=sqrt(3.0)*sqrt(SQR(wEigenVectors[i][3*j])+SQR(wEigenVectors[i][3*j+1])+SQR(wEigenVectors[i][3*j+2]));
				norms[j]=(wEigenVectors[i][3*j+2])/s;
				norms[j]=255*(norms[j]+1)/2.0;
					
			}
			//normalizeLLR(norms, numVertices, 0, 255);
			saveLLRColorMap(norms, numVertices, os.str().c_str());
		}
		delete[] norms;*/
		//-----------------------------------------------------------------------------
		
		//---clean up---
		for(int i=0;i<numClasses;++i){
			delete[] wmean[i];
		}
		for(int i=0;i<rank;++i){
			delete[] bEigenVectors[i];
			delete[] P[i];
		}
		delete[] bmean;
		delete[] bs;
		delete[] bEigenValues;
		delete[] bEigenVectors;
		
		delete[] wmean;
		delete[] ws;
		delete[] P;
		delete[] groupSize;
		return rank;
}







/*void projectData(double **data, int n, int m, double **w, int numDirections, double **projected){
	for(int i=0;i<n;++i){
		for(int k=0;k<numDirections;++k){
			double projection=0;
			for(int j=0;j<m;++j){
				projection+=data[i][j]*w[k][j];
			}
			projected[i][k]=projection;
		}
	}
}

void projectData(double *data, int m, double **w, int numDirections, double *projected){
	for(int k=0;k<numDirections;++k){
		double projection=0;
		for(int j=0;j<m;++j){
			projection+=data[j]*w[k][j];
		}
		projected[k]=projection;
	}
}*/

void computeProjectedCentroids(double **projectedData, int *labels, int n, int numDirections, int numClasses, double **centroids){
	for(int i=0;i<numClasses;++i){
		memset(centroids[i], 0, sizeof(double)*numDirections);
	}
	int *sampleCount=new int[numClasses];
	memset(sampleCount, 0, sizeof(int)*numClasses);
	for(int i=0;i<n;++i){
		int lab=labels[i];
		sampleCount[lab]++;
		for(int j=0;j<numDirections;++j){
			centroids[lab][j]+=projectedData[i][j];
		}
	}
	for(int i=0;i<numClasses;++i){
		for(int j=0;j<numDirections;++j){
			centroids[i][j]/=double(sampleCount[i]);
		}
	}
	delete[] sampleCount;
}

//warning: this prediction algorithm is the simplest one: assign to each sample, its nearest class centroid
void predictHDLDA(double **projectedData, int n, double **centroids, int numClasses,int numDirections, int *predicted, double *priors){
	for(int i=0;i<n;++i){
		int sel=-1;
		double minDist=-1;
		for(int k=0;k<numClasses;++k){
			double Edist=0;
			for(int j=0;j<numDirections;++j){
				Edist+=SQR(projectedData[i][j]-centroids[k][j]);
			}
			if(priors!=NULL){
				Edist-=priors[k];
			}
			if((sel<0) || (Edist<minDist)){
				sel=k;
				minDist=Edist;
			}
		}
		predicted[i]=sel;
	}
}

//warning: this prediction algorithm is the simplest one: assign to each sample, its nearest class centroid
void predictHDLDA(double *projectedData, double **centroids, int numClasses,int numDirections, int &predicted){
	int sel=-1;
	double minDist=-1;
	for(int k=0;k<numClasses;++k){
		double Edist=0;
		for(int j=0;j<numDirections;++j){
			Edist+=SQR(projectedData[j]-centroids[k][j]);
		}
		if((sel<0) || (Edist<minDist)){
			sel=k;
			minDist=Edist;
		}
	}
	predicted=sel;
}

int computeHDLDA_CV(double **_data, int *_labels, int n, int m, int numClasses, bool shuffle,
					 double **&w, double *&lambda, int trainingSize, int *predictions, 
					 double &inSampleError, double &outSampleError, bool evalWithROC){
	
	int testingSize=n-trainingSize;
	double **data=new double*[n];//preserve the ordering in data
	int *labels=new int[n];
	int *trainingSet=new int[trainingSize];
	if(shuffle){
		randomSampleNoReplacement(trainingSize, n, trainingSet);
	}else{
		for(int i=0;i<trainingSize;++i){
			trainingSet[i]=i;
		}
	}
	for(int i=0;i<trainingSize;++i){
		data[i]=_data[trainingSet[i]];
		labels[i]=_labels[trainingSet[i]];
	}
	
	
	int numDirections=computeHighDimensionalLDA(data, labels, trainingSize, m, numClasses, w, lambda);
	double *priors=new double[numClasses];
	memset(priors, 0, sizeof(double)*numClasses);
	for(int i=0;i<trainingSize;++i){
		priors[labels[i]]+=1;
	}
	for(int i=0;i<numClasses;++i){
		priors[i]/=double(trainingSize);
	}
	double **projectedData=new double*[n];
	double **projectedPermutedData=new double*[n];
	
	for(int i=0;i<n;++i){
		projectedData[i]=new double[numDirections];
	}

	for(int i=0;i<trainingSize;++i){
		projectedPermutedData[i]=projectedData[trainingSet[i]];
	}
	projectData(_data, n, m, w, numDirections, projectedData);
	
	double **centroids=new double*[numClasses];
	for(int i=0;i<numClasses;++i){
		centroids[i]=new double[numDirections];
	}
	computeProjectedCentroids(projectedPermutedData, labels, trainingSize, numDirections, numClasses, centroids);

	double *variances=new double[numClasses];
	memset(variances, 0, sizeof(double)*numClasses);
	for(int i=0;i<trainingSize;++i){
		variances[labels[i]]+=SQR(projectedPermutedData[i][0]-centroids[labels[i]][0]);
	}
	for(int i=0;i<numClasses;++i){
		variances[i]/=(priors[i]*trainingSize);
		priors[i]=2*variances[i]*log(sqrt(2*M_PI*variances[i])*priors[i]);
	}

	double **invCov=NULL;
	double *determinants=NULL;
	double *distanceMatrix=NULL;
		if(evalWithROC){
			distanceMatrix=new double[n*n];
			computeDistanceMatrix(projectedData,n,numDirections,distanceMatrix);
		}else{
			predictHDLDA(projectedData, n, centroids, numClasses, numDirections, predictions, priors);
		}


	if(evalWithROC){
		outSampleError=computeVerificationRate(distanceMatrix, _labels, n, 0.001);
		inSampleError=outSampleError;
	}else{
		inSampleError=0;
		outSampleError=0;
		for(int i=0;i<trainingSize;++i){
			inSampleError+=(predictions[trainingSet[i]]!=labels[i]);
		}
		for(int i=0;i<n;++i){
			outSampleError+=(predictions[i]!=_labels[i]);
		}
		outSampleError-=inSampleError;
		inSampleError/=double(trainingSize);
		if(testingSize>0){
			outSampleError/=double(testingSize);
		}else{
			outSampleError=0;
		}
	}
	
	for(int i=0;i<n;++i){
		delete[] projectedData[i];
	}
	for(int i=0;i<numClasses;++i){
		delete[] centroids[i];
	}
	delete[] centroids;
	delete[] projectedData;
	delete[] projectedPermutedData;
	delete[] data;
	delete[] labels;
	delete[] priors;
	if(invCov!=NULL){
		for(int i=0;i<numClasses;++i){
			delete[] invCov[i];
		}
		delete[] invCov;
	}
	if(determinants!=NULL){
		delete[] determinants;
	}
	if(distanceMatrix!=NULL){
		delete[] distanceMatrix;
	}
	return numDirections;
}

//======================================================================================================

void generateLDASummary(const char *fileName, double **data, int *labels, int n, int m, int numClasses, 
						double **w, double *lambda, int numDirections, vector<string> &labelDescription, vector<string> &names){
	
	double **transformedData=new double*[n];
	double **classCentroids=new double*[numClasses];
	for(int i=0;i<numClasses;++i){
		classCentroids[i]=new double[numDirections];
		memset(classCentroids[i], 0, sizeof(double)*numDirections);
	}
	int *groupSize=new int[numClasses];
	memset(groupSize, 0, numClasses*sizeof(int));
	for(int i=0;i<n;++i){
		transformedData[i]=new double[numDirections];
	}
	for(int i=0;i<n;++i){
		int category=labels[i];
		groupSize[category]++;
		for(int k=0;k<numDirections;++k){
			double projection=0;
			for(int j=0;j<m;++j){
				projection+=data[i][j]*w[k][j];
			}
			transformedData[i][k]=projection;
			classCentroids[category][k]+=transformedData[i][k];
		}
	}
	for(int i=0;i<numClasses;++i){
		for(int j=0;j<numDirections;++j){
			classCentroids[i][j]/=double(groupSize[i]);
		}
	}
	//evaluate the nearest-centroid classifier
	double **confusionMatrix=new double*[numClasses];
	for(int j=0;j<numClasses;++j){
		confusionMatrix[j]=new double[numClasses];
		memset(confusionMatrix[j], 0, sizeof(double)*numClasses);
	}

	int totalErrors=0;
	vector<pair<int, int> > failures;
	for(int i=0;i<n;++i){
		int best=-1;
		double bestDistance=-1;
		for(int k=0;k<numClasses;++k){
			double distance=0;
			for(int j=0;j<numDirections;++j){
				distance+=SQR(transformedData[i][j] - classCentroids[k][j]);
			}
			if((best<0) || (distance<bestDistance)){
				best=k;
				bestDistance=distance;
			}
		}
		if(best!=labels[i]){
			failures.push_back(make_pair(i, best));
		}
		confusionMatrix[labels[i]][best]+=1;
		if(labels[i]!=best){
			totalErrors++;
		}
	}
	double totalVariance=0;
	for(int i=0;i<numDirections;++i){
		totalVariance+=SQR(lambda[i]);
	}
	//===print results===
	FILE *F=fopen(fileName, "w");
	fprintf(F,"NUM_SAMPLES\t%d\n", n);
	fprintf(F,"DATA_DIMENSION\t%d\n", m);
	fprintf(F,"NUM_CATEGORIES\t%d\n", numClasses);
	fprintf(F,"NUM_DIRECTIONS\t%d\n", numDirections);
	fprintf(F,"TOTAL_VARIANCE\t%0.8E\n", totalVariance);
	fprintf(F,"TOTAL_ERRORS\t%d\n", totalErrors);
	fprintf(F,"CONFUSION_MATRIX\n");
	fprintf(F,"Left: ground-truth. Top: predicted\t\n");
	fprintf(F,"Groups: ");
	if(!labelDescription.empty()){
		for(unsigned i=0;i<labelDescription.size();++i){
			fprintf(F, "%s%s", (i>0)?"\t":"", labelDescription[i].c_str());
		}
	}else{
		fprintf(F," not available.");
	}
	fprintf(F,"\n");
	
	for(int i=0;i<numClasses;++i){
		for(int j=0;j<numClasses;++j){
			confusionMatrix[i][j]*=100.0/double(groupSize[i]);
			fprintf(F,"%0.2lf\t",confusionMatrix[i][j]);
		}
		fprintf(F,"\n");
	}
	fprintf(F,"Failures:\nSubject\tGroundTruth\tPredicted\n");
	for(unsigned i=0;i<failures.size();++i){
		if(labelDescription.empty()){
			fprintf(F,"%d:%s\t%d\t%d\n",failures[i].first, names.empty()?"":names[failures[i].first].c_str(), labels[failures[i].first], failures[i].second);
		}else{
			fprintf(F,"%d:%s\t%s\t%s\n",failures[i].first, names.empty()?"":names[failures[i].first].c_str(), labelDescription[labels[failures[i].first]].c_str() , labelDescription[failures[i].second].c_str());
		}

	}
	fclose(F);
	//===================

	for(int j=0;j<numClasses;++j){
		delete[] confusionMatrix[j];
	}
	delete[] confusionMatrix;
	for(int i=0;i<n;++i){
		delete[] transformedData[i];
	}
	delete[] transformedData;
	delete[] groupSize;
	
}
void generateClassROCCurves(const char *fileName, double **data, int *labels, int n, int m, int numClasses, double **testData, int *testLabels, int nTest,
						double **w, double *lambda, int numDirections, std::vector<std::string> &labelDescription, vector<vector<double> > &verificationThresholds){
	if((testData==NULL) || (testLabels==NULL) || (nTest==0)){
		cerr<<"Warning: test data not provided. Evaluating performance on the train data."<<endl;
		testData=data;
		testLabels=labels;
		nTest=n;
	}
	double **transformedData=new double*[n];
	double **transformedTestData=new double*[nTest];
	double **classCentroids=new double*[numClasses];
	for(int i=0;i<numClasses;++i){
		classCentroids[i]=new double[numDirections];
		memset(classCentroids[i], 0, sizeof(double)*numDirections);
	}
	int *groupSize=new int[numClasses];
	memset(groupSize, 0, numClasses*sizeof(int));
	for(int i=0;i<n;++i){
		transformedData[i]=new double[numDirections];
	}

	for(int i=0;i<nTest;++i){
		transformedTestData[i]=new double[numDirections];
	}

	for(int i=0;i<n;++i){
		int category=labels[i];
		groupSize[category]++;
		for(int k=0;k<numDirections;++k){
			double projection=0;
			for(int j=0;j<m;++j){
				projection+=data[i][j]*w[k][j];
			}
			transformedData[i][k]=projection;
			classCentroids[category][k]+=transformedData[i][k];
		}
	}
	for(int i=0;i<nTest;++i){
		for(int k=0;k<numDirections;++k){
			double projection=0;
			for(int j=0;j<m;++j){
				projection+=testData[i][j]*w[k][j];
			}
			transformedTestData[i][k]=projection;
		}
	}

	for(int i=0;i<numClasses;++i){
		for(int j=0;j<numDirections;++j){
			classCentroids[i][j]/=double(groupSize[i]);
		}
	}

	double *distMatrix=new double[nTest*numClasses];
	vector<double> *matchScores=new vector<double>[numClasses];
	vector<double> *nonMatchScores=new vector<double>[numClasses];
	double identificationAccuracy=0;
	for(int i=0;i<nTest;++i){
		int bestClass=-1;
		double minDist;
		double distSum=0;
		for(int k=0;k<numClasses;++k){
			double distance=0;
			for(int j=0;j<numDirections;++j){
				distance+=SQR(transformedTestData[i][j] - classCentroids[k][j]);
			}
			distMatrix[i*numClasses+k]=distance;
			distSum+=distance;
		}

		for(int k=0;k<numClasses;++k){
			double distance=distMatrix[i*numClasses+k];//distSum;
			if(labels[i]==k){
				matchScores[k].push_back(distance);
			}else{
				nonMatchScores[k].push_back(distance);
			}
			if((bestClass<0) || distance<minDist){
				minDist=distance;
				bestClass=k;
			}
		}
		if(bestClass==testLabels[i]){
			identificationAccuracy+=1;
		}
	}
	identificationAccuracy/=nTest;
	for(int i=0;i<numClasses;++i){
		sort(matchScores[i].begin(), matchScores[i].end());
		sort(nonMatchScores[i].begin(), nonMatchScores[i].end());
	}
	//===print results===
	FILE *F=fopen(fileName, "w");
	fprintf(F,"Groups: ");
	if(!labelDescription.empty()){
		for(unsigned i=0;i<labelDescription.size();++i){
			fprintf(F, "%s%s", (i>0)?"\t":"", labelDescription[i].c_str());
		}
		fprintf(F,"\n");
	}else{
		fprintf(F," not available.\n");
	}
	fprintf(F,"Identification accuracy: %0.2lf%%\n", 100*identificationAccuracy);
	fprintf(F,"Thresholds (0.001, 0.01 and 0.1 FAR): \n");
	verificationThresholds.clear();
	for(int i=0;i<numClasses;++i){
		vector<double> vt;
		vt.push_back(nonMatchScores[i][(nonMatchScores[i].size()/1000)]);
		vt.push_back(nonMatchScores[i][(nonMatchScores[i].size()/100)]);
		vt.push_back(nonMatchScores[i][(nonMatchScores[i].size()/10)]);
		fprintf(F,"Class %d:\t%0.8E\t%0.8E\t%0.8E\n", i, vt[0], vt[1], vt[2]);
		verificationThresholds.push_back(vt);
	}
	printROCData(matchScores, nonMatchScores, numClasses, F);
	fprintf(F,"\n");
	fclose(F);
	//===================

	delete[] matchScores;
	delete[] nonMatchScores;
	delete[] distMatrix;

	for(int i=0;i<n;++i){
		delete[] transformedData[i];
	}
	delete[] transformedData;

	for(int i=0;i<nTest;++i){
		delete[] transformedTestData[i];
	}
	delete[] transformedTestData;

	
	delete[] groupSize;
	
}

void HDLDA_toyExample_CV(const char *fileName, int n, int m, int numClasses, double trainingProportion){
	int *labels=new int[n];
	double **data=new double*[n];

	FILE *F=fopen(fileName, "r");
	for(int i=0;i<n;++i){
		data[i]=new double[m];
		for(int j=0;j<m;++j){
			fscanf(F,"%lf", &data[i][j]);
		}
		fscanf(F,"%d", &labels[i]);
	}
	fclose(F);
	bool labelGeneralization=true;
	bool shuffle=true;

	int validationSize;
	if(labelGeneralization){
		validationSize=n-randomCategorySample(data, labels, n, trainingProportion);
	}else{
		validationSize=n*(1-trainingProportion);
		if(shuffle){
			for(int i=0;i<n-validationSize;++i){
				double *ftemp;
				int itemp;
				int sel=i+rand()%(n-i);
				SWAP(data[i], data[sel], ftemp);
				SWAP(labels[i], labels[sel], itemp);
			}
		}
	}
	int *newLabels=new int[n];
	memcpy(newLabels, labels, sizeof(int)*n);
	n-=validationSize;
	convertToConsecutiveIds(newLabels, n);
	convertToConsecutiveIds(newLabels+n, validationSize);
	double **w=NULL;
	double *lambda=NULL;
	double inSampleError=0;
	double outSampleError=0;
	int *predictions=new int[n+validationSize];
	int rank;
	const int numRuns=10;
	double *EProd=new double[n];
	double *EIn=new double[n];
	double *EOut=new double[n];
	double *EInSqr=new double[n];
	double *EOutSqr=new double[n];
	double *rho=new double[n+validationSize];
	memset(EProd, 0, sizeof(double)*n);
	memset(EIn, 0, sizeof(double)*n);
	memset(EOut, 0, sizeof(double)*n);
	memset(EInSqr, 0, sizeof(double)*n);
	memset(EOutSqr, 0, sizeof(double)*n);
	memset(rho, 0, sizeof(double)*n);
	int trainingSize=n*trainingProportion;
	int testingSize=n-trainingSize;
	double promInSample=0;
	double promOutSample=0;
	int kMax;
	cout<<"Total: "<<n+validationSize<<endl;
	cout<<"Training: "<<int(n*trainingProportion)<<endl;
	cout<<"Testing: "<<n-int(n*trainingProportion)<<endl;
	cout<<"Validation: "<<validationSize<<endl;
	cout<<"ValidationSamples: ";
	set<int>validationSamples;
	for(int i=n;i<n+validationSize;++i){
		validationSamples.insert(labels[i]);
	}
	for(set<int>::iterator it=validationSamples.begin(); it!=validationSamples.end();++it){
		cout<<*it<<" ";
	}
	cout<<endl;
	

	double validationError=-1;
	for(int r=0;r<numRuns;++r){
		if(labelGeneralization){
			rank=HDLDA_labelGeneralization(data, newLabels, n, m, trainingProportion*numClasses, trainingProportion, w, lambda, inSampleError, outSampleError, kMax, predictions, false);
			promInSample+=inSampleError;
			promOutSample+=outSampleError;
			set<int> trainingClasses;
			set<int> testingClasses;
			for(int i=0;i<n;++i){
				if(predictions[i]==0){
					trainingClasses.insert(labels[i]);
				}else{
					testingClasses.insert(labels[i]);
				}//the rest is in the validation set
			}
			for(set<int>::iterator it=trainingClasses.begin(); it!=trainingClasses.end();++it){
				cout<<*it<<" ";
			}
			for(set<int>::iterator it=testingClasses.begin(); it!=testingClasses.end();++it){
				cout<<*it<<" ";
			}
			validationError=1-validateLDA(w, rank, data+n, newLabels+n, validationSize, m);

			for(int i=0;i<n;++i){
				EProd[i]+=(1-predictions[i])*validationError;//correlation of "being used for training" and the error
				EIn[i]+=(1-predictions[i]);
				EOut[i]+=validationError;
				EInSqr[i]+=(1-predictions[i]);
				EOutSqr[i]+=validationError*validationError;
			}
			cout<<inSampleError<<" "<<outSampleError<<" "<<validationError<<endl;
		}else{
			rank=computeHDLDA_CV(data, labels, n, m, numClasses, true, w, lambda, trainingSize, predictions, inSampleError, outSampleError,false);
			promInSample+=inSampleError;
			promOutSample+=outSampleError;
			double totalErrors=(inSampleError*trainingSize+outSampleError*testingSize);
			for(int i=0;i<n;++i){
				double ein=(predictions[i]!=labels[i]);
				double outSampleEstimation=(totalErrors-ein)/double(n-1);//leave-one-out
				EProd[i]+=ein*(outSampleEstimation);
				EIn[i]+=ein;
				EOut[i]+=outSampleEstimation;
				EInSqr[i]+=ein;
				EOutSqr[i]+=outSampleEstimation*outSampleEstimation;
			}
		}
		for(int i=0;i<rank;++i){
			delete[] w[i];
		}
		delete[] w;
		delete[] lambda;
		cerr<<"Run "<<r<<":"<<inSampleError<<"\t"<<outSampleError<<"\t"<<validationError<<endl;
	}

	promInSample/=numRuns;
	promOutSample/=numRuns;
	cout<<promInSample<<"\t"<<promOutSample<<endl;
	int rhoNegatives=0;
	for(int i=0;i<n;++i){
		EProd[i]/=double(numRuns);
		EIn[i]/=double(numRuns);
		EOut[i]/=double(numRuns);
		EInSqr[i]/=double(numRuns);
		EOutSqr[i]/=double(numRuns);
		double den=sqrt((EInSqr[i] - EIn[i]*EIn[i])*(EOutSqr[i] - EOut[i]*EOut[i]));
		rho[i]=(EProd[i]-(EIn[i]*EOut[i]));
		if(den>EPSILON){
			rho[i]/=den;
		}
		if(rho[i]<0){
			++rhoNegatives;
			cerr<<i<<" ["<<labels[i]<<"]:\t"<<rho[i]<<endl;
		}
	}
	
	//====discard negatives and add validation set====
	int nn=n+validationSize;
	for(int i=n;i<nn;++i){
		rho[i]=-1;
	}

	set<int> prunedClasses;
	double **prunedData=new double*[nn];
	int *prunedLabels=new int[nn];
	int finalTrainingSize=0;
	for(int i=0;i<nn;++i){
		if(rho[i]<-0.1){
			continue;
		}
		prunedData[finalTrainingSize]=data[i];
		prunedLabels[finalTrainingSize]=labels[i];
		++finalTrainingSize;
		prunedClasses.insert(labels[i]);
	}
	int finalTestingSize=0;
	
	for(int i=n;i<nn;++i){//add only the validation set
		prunedData[finalTrainingSize+finalTestingSize]=data[i];
		prunedLabels[finalTrainingSize+finalTestingSize]=labels[i];
		++finalTestingSize;
		prunedClasses.insert(labels[i]);
	}
	convertToConsecutiveIds(prunedLabels, finalTrainingSize+finalTestingSize);
	int numPrunedClasses=prunedClasses.size();

	if(labelGeneralization){
		rank=HDLDA_labelGeneralization(prunedData, prunedLabels, finalTrainingSize+finalTestingSize, m, numPrunedClasses, trainingProportion, w, lambda, inSampleError, outSampleError, kMax, predictions, false);
	}else{
		computeHDLDA_CV(prunedData, prunedLabels, finalTrainingSize+finalTestingSize, m, numClasses, false, w, lambda, finalTrainingSize, predictions, inSampleError, outSampleError);
	}
	
	cerr<<"Prunned training set result:"<<endl;
	cerr<<inSampleError<<"\t"<<outSampleError<<endl;

	if(labelGeneralization){
		rank=HDLDA_labelGeneralization(data, labels, nn, m, numClasses, trainingProportion, w, lambda, inSampleError, outSampleError, kMax, predictions, false);
	}else{
		computeHDLDA_CV(data, labels, nn, m, numClasses, false, w, lambda, nn-validationSize, predictions, inSampleError, outSampleError);
	}
	
	cerr<<"Complete training set result:"<<endl;
	cerr<<inSampleError<<"\t"<<outSampleError<<endl;
	delete[] prunedData;
	delete[] prunedLabels;
	//=========================
	for(int i=0;i<rank;++i){
		delete[] w[i];
	}
	for(int i=0;i<n;++i){
		delete[] data[i];
	}
	delete[] data;
	delete[] labels;
	delete[] w;
	delete[] lambda;
	delete[] predictions;
	delete[] newLabels;

	delete[] EProd;
	delete[] EIn;
	delete[] EOut;
	delete[] EInSqr;
	delete[] EOutSqr;
	delete[] rho;
	
}

void HDLDA_toyExample(const char *fileName, int n, int m, int numClasses, bool subSample, bool libSVMFormat){
	int *labels=new int[n];
	double **data=new double*[n];

	FILE *F=fopen(fileName, "r");
	for(int i=0;i<n;++i){
		data[i]=new double[m];
		for(int j=0;j<m;++j){
			fscanf(F,"%lf", &data[i][j]);
		}
		fscanf(F,"%d", &labels[i]);
	}
	fclose(F);
	double **w=NULL;
	double *lambda=NULL;
	//=================
	int rank;
	if(subSample){//this is only for identification
		double **training=new double*[n];
		int *trainingLabels=new int[n];
		int trainingSize=0;
		for(int i=0;i<n;++i){//take the first 5 and the last 5 subjects
			if((labels[i]<15) || (numClasses-15<=labels[i])){
				training[trainingSize]=data[i];
				trainingLabels[trainingSize]=labels[i];
				if(trainingLabels[trainingSize]>=15){
					trainingLabels[trainingSize]-=70;
				}
				++trainingSize;
			}
		}
		rank=computeHighDimensionalLDA(training, trainingLabels, trainingSize, m, 30, w, lambda);
	}else{
		rank=computeHighDimensionalLDA(data, labels, n, m, numClasses, w, lambda);
	}
	//take projections
	F=fopen("discriminantDirections.txt","w");
	for(int i=0;i<m;++i){
		for(int j=0;j<rank;++j){
			fprintf(F, "%0.8E\t", w[j][i]);
		}
		fprintf(F, "\n");
	}
	fclose(F);
	
	
	F=fopen("LDAProjections.txt", "w");
	for(int i=0;i<n;++i){
		if(libSVMFormat){
			fprintf(F,"%d\t",labels[i]);
		}
		for(int j=0;j<rank;++j){
			double projection=0;
			for(int k=0;k<m;++k){
				projection+=w[j][k]*data[i][k];
			}
			if(libSVMFormat){
				fprintf(F,"%d:%0.8E\t",j, projection);
			}else{
				fprintf(F,"%0.8E\t",projection);
			}
			
		}
		if(libSVMFormat){
			fprintf(F,"\n");
		}else{
			fprintf(F,"%d\n",labels[i]);
		}
		
	}
	fclose(F);
	vector<string> labelDescription;//if the data was given from a text file, the label description is not available
	vector<string> names;//...we don't have the file names either...
	generateLDASummary("LDA_Summary.txt", data, labels, n, m, numClasses, w, lambda, rank, labelDescription, names);
	for(int i=0;i<rank;++i){
		delete[] w[i];
	}
	for(int i=0;i<n;++i){
		delete[] data[i];
	}
	delete[] data;
	delete[] labels;
	delete[] w;
	delete[] lambda;
}



double computeVerificationRate_AllPairs(double *D, int *labels, int n, double falseAR){
	vector<double> matches;
	vector<double> nonMatches;
	for(int i=0;i<n;++i){
		for(int j=i+1;j<n;++j){
			if(labels[i]==labels[j]){
				matches.push_back(D[i*n+j]);
			}else{
				nonMatches.push_back(D[i*n+j]);
			}
		}
	}
	sort(matches.begin(), matches.end());
	sort(nonMatches.begin(), nonMatches.end());
	int numMatches=matches.size();
	int numNonMatches=nonMatches.size();
	double thr=nonMatches[numNonMatches/1000];
	int numVerif=0;
	while((numVerif<numMatches) && (matches[numVerif]<thr)){
		++numVerif;
	}
	return double(numVerif)/double(numMatches);
}

double computeVerificationRate(double *D, int *labels, int n, double falseAR){
	set<int> subjects;
	for(int i=0;i<n;++i){
		subjects.insert(labels[i]);
	}
	int numSubjects=subjects.size();
	//mark gallery
	int *seen=new int[numSubjects];
	int *galleryIndex=new int[numSubjects];
	memset(galleryIndex, -1, sizeof(int)*numSubjects);
	memset(seen, 0, sizeof(int)*numSubjects);
	for(int i=0;i<n;++i){
		if(seen[labels[i]]==0){
			galleryIndex[labels[i]]=i;
			seen[labels[i]]=1;
		}
	}
	//============================================
	vector<pair<double, int> > scores;
	int index=0;
	for(int i=0;i<n;++i){
		if(galleryIndex[labels[i]]!=i){//it means "i is probe"
			for(int claim=0;claim<numSubjects;++claim){
				int pairPosition=i*n+galleryIndex[claim];
				scores.push_back(make_pair(D[pairPosition], labels[i]==claim));
			}
		}
	}
	sort(scores.begin(), scores.end());
	int totalProbe=n-numSubjects;
	int falseAcceptanceCount=0;
	int verificationCount=0;
	double verificationAtGivenFAR=1;
	bool checked=false;
	for(unsigned i=0;i<scores.size();++i){
		if(scores[i].second==1){
			++verificationCount;	
			double falseAcceptanceRate=double(falseAcceptanceCount)/double(scores.size());
			double verificationRate=double(verificationCount)/double(totalProbe);
			if((falseAR<=falseAcceptanceRate) && (!checked)){
				verificationAtGivenFAR=verificationRate;
				checked=true;
			}
			/*if(F==NULL){
				cout<<falseAcceptanceRate<<"\t"<<verificationRate<<endl;
			}else{
				fprintf(F, "%lf\t%lf\n", falseAcceptanceRate, verificationRate);
			}*/
		}else{
			++falseAcceptanceCount;
		}
	}
	return verificationAtGivenFAR;
}






void HDLDA_labelGeneralization(const char *fileName, double trainingProportion, 
							   double &inSampleError, double &outSampleError){
	double **data;
	int n;
	int m;
	int numClasses;
	set<int> sLabels;
	cerr<<"Loading input data...";
	readPlainMatrixFile<double>(fileName, data, n, m);
	cerr<<"done."<<endl;

	int *labels=new int[n];
	for(int i=0;i<n;++i){
		labels[i]=(int)data[i][m-1];
		sLabels.insert(labels[i]);
	}
	--m;
	numClasses=sLabels.size();

	double **w=NULL;
	double *lambda=NULL;
	int kMax;
	int numDirections=HDLDA_labelGeneralization(data, labels, n, m, numClasses, trainingProportion, w, lambda, inSampleError, outSampleError, kMax, NULL);

	FILE *F=fopen("LDA_bestDirections.txt","w");
	for(int i=0;i<m;++i){
		for(int j=0;j<=kMax;++j){
			fprintf(F,"%0.8E\t",w[j][i]);
		}
		fprintf(F,"\n");
	}
	fclose(F);
	cerr<<"done."<<endl;

	for(int i=0;i<numDirections;++i){
		delete[] w[i];
	}
	delete[] w;
	delete[] lambda;
	for(int i=0;i<n;++i){
		delete[] data[i];
	}
	delete[] data;
	delete[] labels;

}

double validateLDA(double **w, int rank, double **data, int *labels, int n, int m){
	double *D=new double[n*n];
	double **projectedData=new double*[n];
	for(int i=0;i<n;++i){
		projectedData[i]=new double[rank];
	}
	projectData(data,n,m,w,rank,projectedData);
	computeDistanceMatrix(projectedData, n, rank, D);
	double verificationRate=computeVerificationRate(D,labels, n, 0.001);
	for(int i=0;i<n;++i){
		delete[] projectedData[i];
	}
	delete[] projectedData;
	delete[] D;
	return verificationRate;
}


/**
	Returns in kMax the minimum number of directions that maximize the generalization error
	returns the total number of discriminant directions
*/
int HDLDA_labelGeneralization(double **data, int *labels, int n, int m, int numClasses, double trainingProportion, 
						  double **&w, double *&lambda, double &inSampleError, double &outSampleError, int &kMax, int *sampleUsage, bool verbose){

	int numTrainingClasses=numClasses*trainingProportion;
	int numTestingClasses=numClasses-numTrainingClasses;
	int *trainingClasses=new int[numTrainingClasses];
	if(verbose){
		cerr<<"Preparing training and testing data...";
	}
	
	randomSampleNoReplacement(numTrainingClasses,numClasses,trainingClasses);
	cerr<<"Training ids:"<<endl;
	for(int i=0;i<numTrainingClasses;++i){
		cerr<<trainingClasses[i]<<"\t";
	}
	cerr<<endl;
	map<int, int> labelsTrainingMap;
	for(int i=0;i<numTrainingClasses;++i){
		int currentSize=labelsTrainingMap.size();
		labelsTrainingMap[trainingClasses[i]]=currentSize;
	}
	//===get the testing classes;
	map<int, int> labelsTestingMap;
	numTestingClasses=0;
	for(int i=0;i<numClasses;++i){
		if(labelsTrainingMap.find(i)==labelsTrainingMap.end()){
			int currentSize=labelsTestingMap.size();
			labelsTestingMap[i]=currentSize;
		}
	}
	//======================================================
	int trainingSize=0;
	for(int i=0;i<n;++i){
		if(labelsTrainingMap.find(labels[i])!=labelsTrainingMap.end()){
			++trainingSize;
		}
	}
	double **dataTraining=new double*[trainingSize];
	int *labelsTraining=new int[trainingSize];
	int current=0;
	for(int i=0;i<n;++i){
		if(labelsTrainingMap.find(labels[i])!=labelsTrainingMap.end()){
			dataTraining[current]=data[i];
			if(sampleUsage!=NULL){
				sampleUsage[i]=0;//training
			}
			labelsTraining[current]=labelsTrainingMap[labels[i]];
			++current;
		}
	}
	int testingSize=n-trainingSize;
	double **dataTesting=new double*[testingSize];
	int *labelsTesting=new int[testingSize];
	current=0;
	for(int i=0;i<n;++i){
		if(labelsTestingMap.find(labels[i])!=labelsTestingMap.end()){
			dataTesting[current]=data[i];
			if(sampleUsage!=NULL){
				sampleUsage[i]=1;//testing
			}
			labelsTesting[current]=labelsTestingMap[labels[i]];
			++current;
		}
	}
	
	if(verbose){
		cerr<<"done."<<endl;
		cerr<<"Computing High-Dimensional LDA...";
	}
	
	int numDirections=computeHighDimensionalLDA(dataTraining, labelsTraining, trainingSize, m, numTrainingClasses, w, lambda);

	if(verbose){
		cerr<<"done."<<endl;
		cerr<<"Projecting data...";
	}
	
	double **projectedTraining=new double*[trainingSize];
	for(int i=0;i<trainingSize;++i){
		projectedTraining[i]=new double[numDirections];
	}
	double **projectedTesting=new double*[testingSize];
	for(int i=0;i<testingSize;++i){
		projectedTesting[i]=new double[numDirections];
	}
	projectData(dataTraining,trainingSize,m,w,numDirections,projectedTraining);
	projectData(dataTesting,testingSize,m,w,numDirections,projectedTesting);

	if(verbose){
		cerr<<"done."<<endl;
		cerr<<"Evaluating...";
	}
	//============evaluate==========
	
	double *inDistance=new double[trainingSize*trainingSize];
	double *outDistance=new double[testingSize*testingSize];
	/*computeDistanceMatrix(projectedTraining, trainingSize, numDirections, inDistance);
	computeDistanceMatrix(projectedTesting, testingSize, numDirections, outDistance);*/
	memset(inDistance, 0, sizeof(double)*trainingSize*trainingSize);
	memset(outDistance, 0, sizeof(double)*testingSize*testingSize);
	kMax=-1;
	double minOutError=-1;
	double minInError=-1;
	for(int k=0;k<numDirections;++k){
		accumulateDistance(projectedTraining, trainingSize, k, inDistance);
		accumulateDistance(projectedTesting, testingSize, k, outDistance);
		inSampleError=1-computeVerificationRate(inDistance,labelsTraining, trainingSize, 0.001);
		outSampleError=1-computeVerificationRate(outDistance,labelsTesting, testingSize, 0.001);
		
		if(verbose){
			cerr<<"In-sample error["<<k+1<<"]:\t"<<inSampleError<<endl;
			cerr<<"Out-of-sample error["<<k+1<<"]:\t"<<outSampleError<<endl;
			cout<<k+1<<"\t"<<inSampleError<<"\t"<<outSampleError<<endl;
		}
		
		if((kMax<0) || (outSampleError<minOutError)){
			kMax=k;
			minOutError=outSampleError;
			minInError=inSampleError;
		}else if(outSampleError==minOutError){
			if(inSampleError<minInError){
				kMax=k;
				minOutError=outSampleError;
				minInError=inSampleError;
			}

		}
	}
	inSampleError=minInError;
	outSampleError=minOutError;
	//============clean=============
	if(verbose){
		cerr<<"Cleaning up...";
	}
	for(int i=0;i<trainingSize;++i){
		delete[] projectedTraining[i];
	}
	delete[] projectedTraining;
	for(int i=0;i<testingSize;++i){
		delete[] projectedTesting[i];
	}
	delete[] projectedTesting;
	
	
	delete[] dataTraining;
	delete[] dataTesting;
	
	delete[] labelsTesting;
	delete[] labelsTraining;
	delete[] inDistance;
	delete[] outDistance;
	if(verbose){
		cerr<<"done."<<endl;
	}
	return numDirections;
}

int HDLDA_nFoldCV(double **_data, int *_labels, int *ids, int n, int m, int numClasses, int folds, 
				  double &inSampleError, double &outSampleError, double &inSampleDev, double &outSampleDev, double *&CM, double *&CMsd){
	if(n%folds!=0){
		cerr<<"Warning: the number of samples is not a multiple of the number of folds."<<endl;
	}
	double **data=new double*[n];
	int *labels=new int[n];
	int *perm=new int[n];
	permuteRelativePosition(ids, n, perm);
	applyPermutation<double*>(_data, perm, n, data);
	applyPermutation<int>(_labels, perm, n, labels);
	int blockSize=n/folds;
	int lastBlockStart=blockSize*(folds-1);
	double *inError=new double[folds];
	double *outError=new double[folds];
	double inErrorMean=0;
	double outErrorMean=0;
	//===confusion matrix with variance
	double **cm=new double*[folds];

	int *numSamples=new int[numClasses];
	for(int k=0;k<folds;++k){
		cm[k]=new double[numClasses*numClasses];
		memset(cm[k], 0, sizeof(double)*numClasses*numClasses);
		int currentBlockStart=blockSize*k;
		if(currentBlockStart!=lastBlockStart){//put the test data at the end of the array
			for(int i=0;i<blockSize;++i){
				double *ftmp=data[currentBlockStart+i];
				data[currentBlockStart+i]=data[lastBlockStart+i];
				data[lastBlockStart+i]=ftmp;
				int itmp=labels[currentBlockStart+i];
				labels[currentBlockStart+i]=labels[lastBlockStart+i];
				labels[lastBlockStart+i]=itmp;
			}
		}
		double **w;
		double *lambda;
		int *predictions=new int[n];
		int numDirections=computeHDLDA_CV(data, labels,  n, m, numClasses, false,w, lambda, blockSize*(folds-1), predictions, inSampleError, outSampleError, false);
		inError[k]=inSampleError;
		outError[k]=outSampleError;
		inErrorMean+=inError[k];
		outErrorMean+=outError[k];

		//=====fill current confusion matrix====
		memset(numSamples,0,sizeof(int)*numClasses);
		for(int i=blockSize*(folds-1);i<n;++i){
			int cell=labels[i]*numClasses+predictions[i];
			cm[k][cell]+=1;
			numSamples[labels[i]]++;
		}
		for(int i=0;i<numClasses;++i){
			if(numSamples[i]==0){
				cm[k][i*(numClasses+1)]=1;
			}
		}
		for(int i=0;i<numClasses*numClasses;++i){
			if(numSamples[i/numClasses]>0){
				cm[k][i]/=numSamples[i/numClasses];
			}
		}
		//======================================
		delete[] predictions;
		for(int i=0;i<numDirections;++i){
			delete[] w[i];
		}
		delete[] w;
		delete[] lambda;
		if(currentBlockStart!=lastBlockStart){//restore the data
			for(int i=0;i<blockSize;++i){
				double *ftmp=data[currentBlockStart+i];
				data[currentBlockStart+i]=data[lastBlockStart+i];
				data[lastBlockStart+i]=ftmp;
				int itmp=labels[currentBlockStart+i];
				labels[currentBlockStart+i]=labels[lastBlockStart+i];
				labels[lastBlockStart+i]=itmp;
			}
		}
		cerr<<"Fold "<<k<<". In-Sample error: "<<inSampleError<<"\tOut-of-sample error: "<<outSampleError<<endl;
	}
	inErrorMean/=folds;
	outErrorMean/=folds;
	inSampleDev=0;
	outSampleDev=0;
	for(int i=0;i<folds;++i){
		inSampleDev+=(inError[i]-inErrorMean)*(inError[i]-inErrorMean);
		outSampleDev+=(outError[i]-outErrorMean)*(outError[i]-outErrorMean);
	}
	inSampleDev=sqrt(inSampleDev/(folds-1));
	outSampleDev=sqrt(outSampleDev/(folds-1));
	inSampleError=inErrorMean;
	outSampleError=outErrorMean;
	CM=new double[numClasses*numClasses];
	CMsd=new double[numClasses*numClasses];
	memset(CM,0,sizeof(double)*numClasses*numClasses);
	memset(CMsd,0,sizeof(double)*numClasses*numClasses);
	for(int k=0;k<folds;++k){
		for(int j=0;j<numClasses*numClasses;++j){
			CM[j]+=cm[k][j];
		}
	}
	for(int j=0;j<numClasses*numClasses;++j){
		CM[j]/=folds;
	}
	for(int k=0;k<folds;++k){
		for(int j=0;j<numClasses*numClasses;++j){
			CMsd[j]+=(cm[k][j]-CM[j])*(cm[k][j]-CM[j]);
		}
	}
	for(int j=0;j<numClasses*numClasses;++j){
		CMsd[j]=sqrt(CMsd[j]/(folds-1));
	}
	delete[] cm;
	delete[] inError;
	delete[] outError;
	delete[] data;
	delete[] labels;
	delete[] perm;
	return 0;
}




int computeSpecializedHighDimensionalLDA(double **data, int *labels, int n, int m, int numClasses, double **excludeData, int nExclude, double lambdaExclude, double **&w, double *&lambda){//n<<m
		bool verbose=false;
		if(verbose){
			cerr<<"(n,m,k)=("<<n<<", "<<m<<", "<<numClasses<<")"<<endl;
		}
	    
	//====compute the first k eigenvectors of the between covariance matrix====
		//********allocate memory**********
		double *bmean=new double[m];
		double *bs=new double[numClasses*numClasses];
		double **wmean=new double*[numClasses];
		double *excludeMean=new double[m];
		if(verbose){
			cerr<<"Allocating memory...";
		}
		for(int i=0;i<numClasses;++i){
			wmean[i]=new double[m];
		}
		int *groupSize=new int[numClasses];
		double *bEigenValues=new double[n];
		double *wEigenValues=new double[n];
		if(verbose){
			cerr<<"done."<<endl;
		}
		//******compute the global mean and the centroids. Center the data around its own centroid*****
		if(verbose){
			cerr<<"B-Mean..."<<data[0][0];
		}
		computeMean(data, n, m, bmean);
		computeMean(excludeData, nExclude, m, excludeMean);
		if(verbose){
			cerr<<"done."<<endl;
			cerr<<"W-Means...";
		}
		computeCentroids(data, n, m, labels, numClasses, wmean, groupSize);
		if(verbose){
			cerr<<"done."<<endl;
			cerr<<"Btween covariance matrix...";
		}
		//*****compute the between covariance matrix*********
		for(int i=0;i<numClasses;++i){
			for(int j=i;j<numClasses;++j){
				double &sum=bs[i*numClasses+j];
				sum=0;
				for(int k=0;k<m;++k){
					sum+=(wmean[i][k]-bmean[k])*(wmean[j][k]-bmean[k]);
				}
				sum*=sqrt(double(groupSize[i]))*sqrt(double(groupSize[j]));
				bs[j*numClasses+i]=sum;
			}
		}
		if(verbose){
			for(int i=0;i<numClasses;++i){
				for(int j=0;j<numClasses;++j){
					cerr<<bs[i*numClasses+j]<<"\t";
				}
				cerr<<endl;
			}
			cerr<<"done."<<endl;
			cerr<<"Eigenvalue decomposition...";
		}
		symmetricEigenDecomposition(bs,bEigenValues,numClasses);
		if(verbose){
			cerr<<"done."<<endl;
			cerr<<"Recovering...";	
		}
		//----recover the eigenvectors (Turk and Pentland)----
		int nullSpaceDimension=0;
		for(int i=0;i<numClasses;++i){
			if(fabs(bEigenValues[i])<1e-6){//having problems with EPSILON=1e-7 =S
				++nullSpaceDimension;
			}
		}
		if(nullSpaceDimension==0){
			nullSpaceDimension=1;
		}
		if(verbose){
			cerr<<"done."<<endl;
		}
		int rank=numClasses-nullSpaceDimension;
		double **bEigenVectors=new double*[rank];
		for(int i=0;i<rank;++i){
			bEigenVectors[i]=new double[m];
			memset(bEigenVectors[i], 0, sizeof(double)*m);
			for(int j=0;j<m;++j){
				for(int k=0;k<numClasses;++k){
					bEigenVectors[i][j]+=sqrt(double(groupSize[k]))*(wmean[k][j]-bmean[j])*bs[(i+nullSpaceDimension)*numClasses+k];
				}
			}
			normalize<double>(bEigenVectors[i],m);
		}
		//multiply the eigenvectors (Y) by sqrt of the inverse of the eigenvalue diagonal (D) to obtain Z (H.Yu and J.Yang "High dimensional LDA")
		for(int j=0;j<rank;++j){
			double multiplier=1.0/sqrt(bEigenValues[j+nullSpaceDimension]);
			for(int i=0;i<m;++i){
				bEigenVectors[j][i]*=multiplier;
			}
		}
		
		//compute P=Z^T Phi_w
		double **P=new double*[rank];
		for(int i=0;i<rank;++i){
			P[i]=new double[n+nExclude];
			memset(P[i], 0, sizeof(double)*(n+nExclude));
			for(int j=0;j<n+nExclude;++j){
				if(j<n){
					int sel=labels[j];
					for(int k=0;k<m;++k){
						P[i][j]+=bEigenVectors[i][k]*(data[j][k]-wmean[sel][k]);
					}
				}else{
					for(int k=0;k<m;++k){
						P[i][j]+=bEigenVectors[i][k]*lambdaExclude*(excludeData[j-n][k]-excludeMean[k]);
					}
				}
			}
		}
		//compute ws=P P^T
		double *ws=new double[rank*rank];
		for(int i=0;i<rank;++i){
			for(int j=i;j<rank;++j){
				double &sum=ws[i*rank+j];
				sum=0;
				for(int k=0;k<n+nExclude;++k){
					sum+=P[i][k]*P[j][k];
				}
				ws[j*rank+i]=sum;
			}
		}
		if(verbose){
			cerr<<"Eigenvalue decomposition...";
		}
		symmetricEigenDecomposition(ws,wEigenValues,rank);
		if(verbose){
			cerr<<"done."<<endl;
		}
		for(int i=0;i<rank;++i){
			double norm=0;
			for(int j=0;j<rank;++j){
				norm+=ws[i*rank+j]*ws[i*rank+j];
			}
			norm=sqrt(norm);
			for(int j=0;j<rank;++j){
				ws[i*rank+j]/=norm;
			}
		}
		if(verbose){
			cerr<<"Sphering...";
		}
		
		//====compute the LDA matrix====
		double **wEigenVectors=new double*[rank];
		for(int i=0;i<rank;++i){
			wEigenVectors[i]=new double[m];
			memset(wEigenVectors[i], 0, sizeof(double)*m);
			for(int j=0;j<m;++j){
				double &sum=wEigenVectors[i][j];
				double *weights=&ws[i*rank];
				for(int k=0;k<rank;++k){
					sum+=weights[k]*bEigenVectors[k][j];
				}
			}
			if(wEigenValues[i]>EPSILON){
				double multiplier=1.0/sqrt(wEigenValues[i]);
				for(int j=0;j<m;++j){
					wEigenVectors[i][j]*=multiplier;
				}
			}
		}
		if(verbose){
			cerr<<"done."<<endl;
		}
		
		
		w=wEigenVectors;
		lambda=wEigenValues;
		
		//---clean up---
		for(int i=0;i<numClasses;++i){
			delete[] wmean[i];
		}
		for(int i=0;i<rank;++i){
			delete[] bEigenVectors[i];
			delete[] P[i];
		}
		delete[] bmean;
		delete[] excludeMean;
		delete[] bs;
		delete[] bEigenValues;
		delete[] bEigenVectors;
		
		delete[] wmean;
		delete[] ws;
		delete[] P;
		delete[] groupSize;
		return rank;
}




