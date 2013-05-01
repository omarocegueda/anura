#ifndef GEOMETRYUTILS_H
#define GEOMETRYUTILS_H
#include "macros.h"
void triangleNormal(double *a, double *b, double *c, double *nrm);
void crossProduct(double *x, double *y, double *cross);
double dotProduct(double *x, double *y, int n=3);
double getAbsAngleDegrees(double *x, double *y, int n);
double getAbsAngle(double *x, double *y, int n);
void fromToRotation(double *from, double *to, double *R);
double euclideanDistance(double *a, double *b, int n);
double euclideanDistanceSQR(double *a, double *b, int n);
void subdivideSphericalMesh(	double *vertices, int numVertices, int *triangles, int numTriangles, int steps, 
								double *&limitingVertices, int &numLimitingVertices, int *&limitingTriangles, int &numLimitingTriangles);
double pointToSegmentSQRDistance(double *A, double *B, double *C);

template<class TYPE> double pointToLineDistance(TYPE *P, TYPE *A, TYPE *B){
	double alpha=0;
	double norm=0;
	for(int i=0;i<3;++i){
		alpha+=(B[i]-A[i])*(P[i]-A[i]);
		norm+=SQR(B[i]-A[i]);
	}
	alpha/=norm;
	norm=0;
	for(int i=0;i<3;++i){
		norm+=SQR((P[i]-A[i])-alpha*(B[i]-A[i]));
	}
	return norm;
}

#endif
