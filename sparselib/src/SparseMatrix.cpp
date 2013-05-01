#include "SparseMatrix.h"
#include <string.h>
#include <iostream>
#include <map>
#include <math.h>
#include "macros.h"
using namespace std;
Edge::Edge(){
	destination=-1;
	w=-1;
}

Edge::Edge(int dest, double _w){
	destination=dest;
	w=_w;
}
//-----------------------------
SparseMatrix::SparseMatrix(){
	n=0;
	k=0;
/*	indexToPosition=NULL;
	positionToIndex=NULL;*/
	edges=NULL;
	degree=NULL;
	diagonal=NULL;

}
SparseMatrix::SparseMatrix(int _n, int _k){
	n=_n;
	k=_k;
	/*indexToPosition=NULL;
	positionToIndex=NULL;*/
	edges=NULL;
	degree=NULL;
	diagonal=NULL;
}


void SparseMatrix::create(int _n, int _k/*, int totalPositions*/){
	n=_n;
	k=_k;
	degree=new int[n];
	memset(degree, 0, sizeof(int)*n);
	diagonal=new double[n];
	memset(diagonal, 0, sizeof(double)*n);
	/*indexToPosition=new int[n];
	positionToIndex=new int[totalPositions];*/
	edges=new Edge*[n];
	for(int i=0;i<n;++i){
		edges[i]=new Edge[k];
	}
}

void SparseMatrix::dellocate(void){
	for(int i=0;i<n;++i){
		delete[] edges[i];
	}
	delete[] edges;
	delete[] degree;
	delete[] diagonal;
}


void SparseMatrix::draw(unsigned char *img_data, int rows, int cols){
	memset(img_data, 0, sizeof(unsigned char)*rows*cols);
	int nz=0;
	for(int i=0;i<n;++i){
		bool seenDiagonalEntry=false;
		for(int j=0;j<k;++j)if(edges[i][j].destination!=-1){
			if(edges[i][j].destination==i){
				seenDiagonalEntry=true;
			}
			int ii=(i*rows)/n;
			int jj=(edges[i][j].destination*cols)/n;
			int dest=ii*cols+jj;
			img_data[dest]=255;
			++nz;
		}
		if((!seenDiagonalEntry) && fabs(diagonal[i])>0){
			int ii=(i*rows)/n;
			int dest=ii*cols+ii;
			img_data[dest]=255;
			++nz;
		}
	}
	cerr<<"Non-zeroes: "<<nz<<endl;
}



void SparseMatrix::multVecRight(double *in, double *out){
	for(int i=0;i<n;++i){
		double sum=diagonal[i]*in[i];
		for(int j=0;j<degree[i];++j){
			sum+=edges[i][j].w*in[edges[i][j].destination];
		}
		out[i]=sum;
	}
}

void SparseMatrix::multVecLeft(double *in, double *out){
	for(int i=0;i<n;++i){
		out[i]=diagonal[i]*in[i];
		for(int j=0;j<degree[i];++j){
			out[edges[i][j].destination]+=in[i]*edges[i][j].w;
		}
	}
}


void SparseMatrix::multDiagLeftRight(double *diagLeft, double *diagRight){
	if((diagLeft==NULL) && (diagRight==NULL)){
		return;
	}
	for(int i=0;i<n;++i){
		diagonal[i]*=diagLeft[i]*diagRight[i];
	}
	if(diagLeft!=NULL){
		if(diagRight!=NULL){
			for(int i=0;i<n;++i){
				double factor=diagLeft[i];
				for(int j=0;j<k;++j){
					int column=edges[i][j].destination;
					edges[i][j].w*=factor*diagRight[column];
				}
			}
		}else{
			for(int i=0;i<n;++i){
				double factor=diagLeft[i];
				for(int j=0;j<k;++j){
					edges[i][j].w*=factor;
				}
			}
		}
	}else if(diagRight!=NULL){
		for(int i=0;i<n;++i){
			for(int j=0;j<k;++j){
				int column=edges[i][j].destination;
				edges[i][j].w*=diagRight[column];
			}
		}
	}
}
void SparseMatrix::sumRowAbsValues(double *sums){
	for(int i=0;i<n;++i){
		sums[i]=fabs(diagonal[i]);
		for(int j=0;j<k;++j)if(edges[i][j].destination!=-1){
			sums[i]+=fabs(edges[i][j].w);
		}
	}
}

void SparseMatrix::sumRowValues(double *sums){
	for(int i=0;i<n;++i){
		sums[i]=diagonal[i];
		for(int j=0;j<k;++j)if(edges[i][j].destination!=-1){
			sums[i]+=edges[i][j].w;
		}
	}
}

void SparseMatrix::sumColumnValues(double *sums){
	memset(sums,0,sizeof(double)*n);
	for(int i=0;i<n;++i){
		sums[i]+=diagonal[i];
		for(int j=0;j<k;++j)if(edges[i][j].destination!=-1){
			sums[edges[i][j].destination]+=edges[i][j].w;
		}
	}
}

void SparseMatrix::sumColumnAbsValues(double *sums){
	memset(sums,0,sizeof(double)*n);
	for(int i=0;i<n;++i){
		sums[i]+=fabs(diagonal[i]);
		for(int j=0;j<k;++j)if(edges[i][j].destination!=-1){
			sums[edges[i][j].destination]+=fabs(edges[i][j].w);
		}
	}
}

int SparseMatrix::addEdge(int from, int to, double weight){
	if(from==to){
		diagonal[from]=weight;
		return 0;
	}
	if(degree[from]>=k){
		return -1;
	}
	for(int i=0;i<degree[from];++i){
		if(edges[from][i].destination==to){
			cerr<<"Warning: duplicate edge from "<<from<<" to "<<to<<endl;
			return -1;
		}
	}
	edges[from][degree[from]].destination=to;
	edges[from][degree[from]].w=weight;
	degree[from]++;
	return 0;
}

SparseMatrix::~SparseMatrix(){
	if(edges!=NULL){
		delete[] degree;
	}
	if(edges!=NULL){
		for(int i=0;i<n;++i){
			delete[] edges[i];
		}
		delete[] edges;
	}
	if(diagonal!=NULL){
		delete[] diagonal;
	}
}

double SparseMatrix::computeAsymetry(void){
	map<pair<int, int>, int > P;
	map<pair<int, int>, double> Q;
	for(int i=0;i<n;++i){
		for(int j=0;j<degree[i];++j){
			P[make_pair(i,edges[i][j].destination)]++;
			P[make_pair(edges[i][j].destination,i)]++;
			Q[make_pair(i,edges[i][j].destination)]=edges[i][j].w;

		}
	}
	double sum=0;
	for(map<pair<int, int>, double>::iterator it=Q.begin(); it!=Q.end();++it){
		if(it->first.first<it->first.second){
			sum+=fabs(it->second - Q[make_pair(it->first.second, it->first.first)]);
		}
	}

	int total=0;
	for(map<pair<int, int>, int>::iterator it=P.begin(); it!=P.end();++it){
		if(it->second!=2){
			cerr<<it->first.first<<","<<it->first.second<<endl;
			++total;
		}
	}
	//return double(total)/double(P.size());
	return sum;
}

double SparseMatrix::retrieve(int r, int c){
	if(r==c){
		return diagonal[r];
	}
	for(int j=0;j<k;++j){
		int jj=edges[r][j].destination;
		if(jj<0){
			continue;
		}
		if(jj==c){
			return edges[r][j].w;
		}
	}
	return 0;
}

int SparseMatrix::copyFrom(SparseMatrix &S){
	if((n!=S.n) || (k!=S.k)){
		cerr<<"copyFrom(...) at SparseMatrix.cpp: Not suported."<<endl;
		return -1;
	}
	for(int i=0;i<n;++i){
		for(int j=0;j<k;++j){
			edges[i][j].destination=S.edges[i][j].destination;
			edges[i][j].w=S.edges[i][j].w;
		}
	}
	return 0;
}

//returns the number of vertices of the submatrix
int SparseMatrix::copySubmatrix(SparseMatrix &S, int k, int *labels, int *newToOldMapping, int *oldToNewMapping){
	int newN=0;
	int newK=0;
	for(int i=0;i<S.n;++i)if(labels[i]==k){
		newToOldMapping[newN]=i;
		oldToNewMapping[i]=newN;
		++newN;
		newK=MAX(newK, S.degree[i]);
	}
	dellocate();
	create(newN, newK);
	newN=0;
	for(int i=0;i<S.n;++i)if(labels[i]==k){
		diagonal[newN]=S.diagonal[i];
		for(int j=0;j<S.degree[i];++j){
			int jj=S.edges[i][j].destination;
			if(labels[jj]!=k){
				continue;
			}
			double w=S.edges[i][j].w;
			addEdge(oldToNewMapping[i], oldToNewMapping[jj], w);
		}
		++newN;
	}
	return newN;
}

int SparseMatrix::sumToDiagonal(double *d){
	for(int i=0;i<n;++i){
		diagonal[i]+=d[i];
	}
	return 0;
}


int SparseMatrix::testEigenPair(double *evec, double eval){
	double maxDiff=0;
	for(int i=0;i<n;++i){
		double sum=diagonal[i]*evec[i];
		for(int j=0;j<k;++j)if(edges[i][j].destination>=0){
			sum+=edges[i][j].w*evec[edges[i][j].destination];
		}
		maxDiff=MAX(maxDiff, fabs(sum-eval*evec[i]));
	}
	cerr<<"Test EP: "<<maxDiff<<endl;
	return 0;

}
