#include <set>
#include <map>
#include <string.h>
using namespace std;

/*
	EC-QMMF node iteration with arbitrary graph topology
*/
//p[s] is te probability distribution for node s
void iterateNode(int site, int numClases, double **v, double **p, int **neighbors, int* numNeighbors, int numVertices, double lambda, double mu, double *&N, double *&D){
	if(numNeighbors[site]<1){
		return;
	}
	if(N==NULL)
         N=new double[numClases];
    if(D==NULL)
         D=new double[numClases];

	for(int k=0;k<numClases;++k){
		double num=0;
		for(int i=0;i<numNeighbors[site];++i){
			num+=p[neighbors[site][i]][k];
		}
		N[k]=lambda*num;
        D[k]=v[site][k] - mu + lambda*numNeighbors[site];
	}
	double sNum=0, sDen=0;
    for(int k=0;k<numClases;k++){
		sNum+=N[k]/D[k];
		sDen+=1.0/D[k];
	}
	double normalizacion=0;
	for(int k=0;k<numClases;k++){
		p[site][k]=(1-sNum)/(D[k]*sDen) + N[k]/D[k];
		if(p[site][k]<0)
			p[site][k]=0;
		if(p[site][k]>1)
			p[site][k]=1;
		normalizacion+=p[site][k];
	}
	for(int k=0;k<numClases;k++)
		p[site][k]/=normalizacion;
}


void updateMeanVectors(double **means, double **data, double **p, int numVertices, int dataDimmension, int numClases){
	for(int i=0;i<numClases;++i){
		memset(means[i], 0, dataDimmension*sizeof(double));
	}
	for(int j=0;j<numClases;++j){
		for(int k=0;k<dataDimmension;++k){
			double sum=0;
			for(int i=0;i<numVertices;++i){
				double pp=p[i][j]*p[i][j];
				means[j][k]+=data[i][k]*pp;
				sum+=pp;
			}
			means[j][k]/=sum;
		}
	}
}

void computeGaussianNegLogLikelihood(double **data, double **models, int numVertices, int numClases, int dataDimmension, double **v){
	for(int i=0;i<numVertices;++i){
		for(int j=0;j<numClases;++j){
			double sqDist=0;
			for(int k=0;k<dataDimmension;++k){
				sqDist+=(models[j][k]-data[i][k])*(models[j][k]-data[i][k]);
			}
			v[i][j]=sqDist;
		}
	}
}
