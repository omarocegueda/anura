#include "linearalgebra.h"
#include "geometryutils.h"
#include "macros.h"
#include <vector>
#include <map>
#include <iostream>
using namespace std;

void triangleNormal(double *a, double *b, double *c, double *nrm){
	double x[3]={b[0]-a[0], b[1]-a[1], b[2]-a[2]};
	double y[3]={c[0]-a[0], c[1]-a[1], c[2]-a[2]};
	crossProduct(x, y, nrm);
	double n=sqrt(SQR(nrm[0])+SQR(nrm[1])+SQR(nrm[2]));
	nrm[0]/=n;
	nrm[1]/=n;
	nrm[2]/=n;
}

void crossProduct(double *x, double *y, double *cross){
	cross[0]=x[1]*y[2]-x[2]*y[1];
	cross[1]=x[2]*y[0]-x[0]*y[2];
	cross[2]=x[0]*y[1]-x[1]*y[0];
}

double dotProduct(double *x, double *y, int n){
	double prod=0;
	for(int i=0;i<n;++i){
		prod+=x[i]*y[i];
	}
	return prod;
}

double getAbsAngle(double *x, double *y, int n){
	double prod=fabs(dotProduct(x,y,n));
	double xnorm=sqrt(dotProduct(x,x,n));
	double ynorm=sqrt(dotProduct(y,y,n));
	prod=MIN(1,prod/(xnorm*ynorm));
	prod=acos(prod);
	return prod;
}

double getAbsAngleDegrees(double *x, double *y, int n){
	double prod=fabs(dotProduct(x,y,n));
	//double prod=dotProduct(x,y,n);
	double xnorm=sqrt(dotProduct(x,x,n));
	double ynorm=sqrt(dotProduct(y,y,n));
	prod=MIN(1,prod/(xnorm*ynorm));
	prod=acos(prod)*180.0/M_PI;
	return prod;
}

/*
	Generates the mtx rotation matrix for going from FROM to TO once the result is given for operate use 
	Mr = R * M * R' for matrixes or use Vr = R * V for vectors
*/
void fromToRotation(double *from, double *to, double *R){
	double u[3], v[3];/* temporary storage vectors */
	double e=dotProduct(from, to);
	double f=fabs(e);
	if(f > 1.0 - EPSILON){//"from" and "to"-vector almost parallel
		double x[3]={fabs(from[0]), fabs(from[1]), fabs(from[2])};
		int minIndex=(x[0]<x[1])?((x[0]<x[2])?(0):(2)):((x[1]<x[2])?(1):(2));
		x[0]=x[1]=x[2]=0;
		x[minIndex]=1;
		u[0] = x[0] - from[0]; u[1] = x[1] - from[1]; u[2] = x[2] - from[2];
		v[0] = x[0] - to[0];   v[1] = x[1] - to[1];   v[2] = x[2] - to[2];
	    
		double c1 = 2.0 / dotProduct(u, u);
		double c2 = 2.0 / dotProduct(v, v);
		double c3 = c1 * c2  * dotProduct(u, v);
	    
		for(int i=0;i<3;++i){
			double *r=&R[3*i];
			for(int j=0;j<3;++j){
				r[j]=-c1*u[i]*u[j]-c2*v[i]*v[j]+c3*v[i]*u[j];
			}
			r[i]=r[i]+1.0;
		}
	}else{
		crossProduct(from, to, v);
		double h=(1.0-e)/dotProduct(v, v);
		double hvx=h*v[0];
		double hvz=h*v[2];
		double hvxy=hvx*v[1];
		double hvxz=hvx*v[2];
		double hvyz=hvz*v[1];
		R[0]=e+hvx*v[0];
		R[1]=hvxy-v[2];
		R[2]=hvxz+v[1];
	    
		R[3]=hvxy+v[2];
		R[4]=e+h*v[1]*v[1];
		R[5]=hvyz-v[0];
	    
		R[6]=hvxz-v[1];
		R[7]=hvyz+v[0];
		R[8]=e+hvz*v[2];
	}
}


double euclideanDistance(double *a, double *b, int n){
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=SQR(a[i]-b[i]);
	}
	return sqrt(sum);
}
double euclideanDistanceSQR(double *a, double *b, int n){
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=SQR(a[i]-b[i]);
	}
	return sum;
}

void subdivideSphericalMesh(	double *vertices, int numVertices, int *triangles, int numTriangles, int steps, 
								double *&limitingVertices, int &numLimitingVertices, int *&limitingTriangles, int &numLimitingTriangles){
	if((limitingTriangles!=NULL)||(limitingVertices!=NULL)){
		cerr<<"Warning: passing non-NULL arrays to subdivideSphericalMesh(...), it may cause a memory leak."<<endl;
		limitingTriangles=NULL;
		limitingVertices=NULL;
		return;
	}
	vector<double *> newVertices;
	for(int i=0;i<numVertices;++i){
		double *newVertex=new double[3];
		memcpy(newVertex, &vertices[3*i], sizeof(double)*3);
		normalize<double>(newVertex, 3);
		newVertices.push_back(newVertex);
	}
	vector<int *> newTriangles;
	for(int i=0;i<numTriangles;++i){
		int *newTriangle=new int[3];
		memcpy(newTriangle, &triangles[3*i], sizeof(int)*3);
		newTriangles.push_back(newTriangle);
	}
	for(int iter=0;iter<steps;++iter){
		map<pair<int, int>, int> edges;
		int currentTriangleCount=newTriangles.size();
		for(unsigned triangle=0;triangle<currentTriangleCount;++triangle){
			int newIndices[3];
			for(int j=0;j<3;++j){
				int from=	MIN(newTriangles[triangle][j], newTriangles[triangle][(j+1)%3]);
				int to=		MAX(newTriangles[triangle][j], newTriangles[triangle][(j+1)%3]);
				pair<int, int> edge=make_pair(from, to);
				map<pair<int, int>, int>::iterator it;
				if((it=edges.find(edge))==edges.end()){
					double *newVertex=new double[3];
					newVertex[0]=newVertices[from][0]+newVertices[to][0];
					newVertex[1]=newVertices[from][1]+newVertices[to][1];
					newVertex[2]=newVertices[from][2]+newVertices[to][2];
					normalize(newVertex, 3);
					newIndices[j]=newVertices.size();
					newVertices.push_back(newVertex);
					edges[edge]=newIndices[j];
				}else{
					newIndices[j]=it->second;
				}
			}
			int triangleArray[3][3]={
				{newTriangles[triangle][0], newIndices[0], newIndices[2]},
				{newTriangles[triangle][1], newIndices[1], newIndices[0]},
				{newTriangles[triangle][2], newIndices[2], newIndices[1]}
			};
			for(int j=0;j<3;++j){
				int *newTriangle=new int[3];
				memcpy(newTriangle, triangleArray[j], sizeof(int)*3);
				newTriangles.push_back(newTriangle);
			}
			newTriangles[triangle][0]=newIndices[0];
			newTriangles[triangle][1]=newIndices[1];
			newTriangles[triangle][2]=newIndices[2];
		}
	}
	numLimitingVertices=newVertices.size();
	numLimitingTriangles=newTriangles.size();
	limitingVertices=new double[3*numLimitingVertices];
	limitingTriangles=new int[3*numLimitingTriangles];
	for(int i=0;i<numLimitingVertices;++i){
		memcpy(&limitingVertices[3*i], newVertices[i], sizeof(double)*3);
	}
	for(int i=0;i<numLimitingTriangles;++i){
		memcpy(&limitingTriangles[3*i], newTriangles[i], sizeof(int)*3);
	}
}

double pointToSegmentSQRDistance(double *A, double *B, double *C){
	double len2=sqrDistance<double>(A, B, 3);
	if(len2<EPSILON_SQR){
		double d=sqrDistance(A, C, 3);
		return d;
	}
	double AB[3]={B[0]-A[0], B[1]-A[1], B[2]-A[2]};
	double BC[3]={C[0]-B[0], C[1]-B[1], C[2]-B[2]};
	double dp=dotProduct(AB, BC, 3);
	if(dp>0){
		return sqrDistance(B,C,3);
	}
	double AC[3]={C[0]-A[0], C[1]-A[1], C[2]-A[2]};
	dp=dotProduct(AB, AC, 3);
	if(dp<0){
		return sqrDistance(A,C,3);
	}
	double cross[3];
	crossProduct(AB, AC, cross);
	double d=sqrNorm<double>(cross,3);
	return d/len2;
}