#include "hornalign.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linearalgebra.h"
#include "geometryutils.h"
#define ALIGN_WITH_SCALE 1
#define ALIGN_FOR_SPIN   2
#define ALIGN_POINT_SURFACE 4
#define MAXSCALE 10
#define MINSCALE 0.1

/**
 * Computes the weighted mean of a collection of points.
 * @param w is the weight array; w[i] is the weight of point i.
 * @param X is the coordinate array; the coordinates of the i-th
 * point are located in X[3*i], X[3*i+1], X[3*i+2].
 * @param n is the number of points contained in X.
 * @param wm contains the weighted mean at the end.
 */
void wmean(double *w, double *X, int n, double wm[3]){
  int i;
  double sum=0;
  memset(wm,0,3*sizeof(double));
  if(w!=NULL){
    for(i=n; i-- > 0;){
      wm[0]+=w[i]*X[3*i];
      wm[1]+=w[i]*X[3*i+1];
      wm[2]+=w[i]*X[3*i+2];
      sum+=w[i];
    }
    wm[0]/=sum;
    wm[1]/=sum;
    wm[2]/=sum;
  }else{
    for(i=n; i-- > 0;){
      wm[0]+=X[3*i];
      wm[1]+=X[3*i+1];
      wm[2]+=X[3*i+2];
    }
    wm[0]/=n;
    wm[1]/=n;
    wm[2]/=n;
  }
}

/**
 * Computes the weighted cross covariance matrix of two point sets.
 * @param w is the weight array; w[i] is the weight of point i.
 * @param P is the coordinate array of the first point set.
 * @param wmp contains the weighted mean of P.
 * @param X is the coordinate array of the second point set.
 * @param wmx contains the weighted mean of X.
 * @param n is the number of points contained in X and P.
 * @param spx contains the weighted cross covariance matrix of the two point sets at the end.
 */
void wcov(double *w, double *P, double wmp[3], double *X, double wmx[3], int n, double spx[3][3]){
  int i;
  double p[3],x[3];
  memset(spx,0,9*sizeof(double));
  if(w!=NULL){
    for(i=n; i-- > 0;){
      p[0] = P[3*i+0]-wmp[0];
      p[1] = P[3*i+1]-wmp[1];
      p[2] = P[3*i+2]-wmp[2];
      x[0] = X[3*i+0]-wmx[0];
      x[1] = X[3*i+1]-wmx[1];
      x[2] = X[3*i+2]-wmx[2];
      spx[0][0] += w[i]*p[0]*x[0];
      spx[0][1] += w[i]*p[0]*x[1];
      spx[0][2] += w[i]*p[0]*x[2];
      spx[1][0] += w[i]*p[1]*x[0];
      spx[1][1] += w[i]*p[1]*x[1];
      spx[1][2] += w[i]*p[1]*x[2];
      spx[2][0] += w[i]*p[2]*x[0];
      spx[2][1] += w[i]*p[2]*x[1];
      spx[2][2] += w[i]*p[2]*x[2];
    }
  }else{
    for(i=n; i-- > 0;){
      p[0] = P[3*i+0]-wmp[0];
      p[1] = P[3*i+1]-wmp[1];
      p[2] = P[3*i+2]-wmp[2];
      x[0] = X[3*i+0]-wmx[0];
      x[1] = X[3*i+1]-wmx[1];
      x[2] = X[3*i+2]-wmx[2];
      spx[0][0] += p[0]*x[0];
      spx[0][1] += p[0]*x[1];
      spx[0][2] += p[0]*x[2];
      spx[1][0] += p[1]*x[0];
      spx[1][1] += p[1]*x[1];
      spx[1][2] += p[1]*x[2];
      spx[2][0] += p[2]*x[0];
      spx[2][1] += p[2]*x[1];
      spx[2][2] += p[2]*x[2];
    }
  }

}

/**
 * Computes the quaternion that defines the optimal rotation according to Horn's alignment method.
 * @param spx is the weighted cross covariance matrix of the two point sets that need to be aligned.
 * @param q contains the quaternion that represents the optimal rotation for aligning the pointsets that formed spx.
 */
void hornQuaternion(double spx[3][3], double q[4]){
  double Q[16];
  double eigval[4];
  double eigvec[16];
  double trace;

  trace = spx[0][0]+spx[1][1]+spx[2][2];

  Q[0] = trace;
  Q[1] = Q[4]  = spx[1][2]-spx[2][1];
  Q[2] = Q[8]  = spx[2][0]-spx[0][2];
  Q[3] = Q[12] = spx[0][1]-spx[1][0];
  Q[5]  = 2*spx[0][0]-trace;
  Q[10] = 2*spx[1][1]-trace;
  Q[15] = 2*spx[2][2]-trace;
  Q[6]  = Q[9]  = spx[0][1]+spx[1][0];
  Q[7]  = Q[13] = spx[0][2]+spx[2][0];
  Q[11] = Q[14] = spx[1][2]+spx[2][1];
  symmetricEigenDecomposition(Q,eigval,4);
  for(int i=0;i<4;++i){
	  for(int j=0;j<4;++j){
		  eigvec[4*i+j]=Q[4*j+i];
	  }
  }
  if(eigvec[3] >= 0){
    q[0]=eigvec[3];
    q[1]=eigvec[7];
    q[2]=eigvec[11];
    q[3]=eigvec[15];
  }
  else{
    q[0]=-eigvec[3];
    q[1]=-eigvec[7];
    q[2]=-eigvec[11];
    q[3]=-eigvec[15];
  }
}

/**
 * Computes the rotation matrix that is equivalent to a unit quaternion
 * @param q a unit quaternion
 * @param R a 4x4 matrix containing the rotation defined by q.
 */
void qRotation(double q[4], double R[4][4]){
  double q02,q12,q22,q32,q0q1,q0q2,q0q3,q1q2,q1q3,q2q3;
  q02  = q[0]*q[0];
  q12  = q[1]*q[1];
  q22  = q[2]*q[2];
  q32  = q[3]*q[3];
  q0q1 = q[0]*q[1];
  q0q2 = q[0]*q[2];
  q0q3 = q[0]*q[3];
  q1q2 = q[1]*q[2];
  q1q3 = q[1]*q[3];
  q2q3 = q[2]*q[3];

  R[0][0] = q02+q12-q22-q32;
  R[1][1] = q02+q22-q12-q32;
  R[2][2] = q02+q32-q12-q22;
  R[0][1] = 2*(q1q2-q0q3); R[1][0] = 2*(q1q2+q0q3);
  R[0][2] = 2*(q1q3+q0q2); R[2][0] = 2*(q1q3-q0q2);
  R[1][2] = 2*(q2q3-q0q1); R[2][1] = 2*(q2q3+q0q1);

  R[0][3] = R[1][3] = R[2][3] = R[3][0] = R[3][1] = R[3][2] = 0;
  R[3][3] = 1;
}

/**
 * Computes the rotation matrix that is equivalent to a unit quaternion
 * @param q a unit quaternion
 * @param R a 4x4 matrix containing the rotation defined by q.
 */
void qRotation(double *q, double *R){
  double q02,q12,q22,q32,q0q1,q0q2,q0q3,q1q2,q1q3,q2q3;
  q02  = q[0]*q[0];
  q12  = q[1]*q[1];
  q22  = q[2]*q[2];
  q32  = q[3]*q[3];
  q0q1 = q[0]*q[1];
  q0q2 = q[0]*q[2];
  q0q3 = q[0]*q[3];
  q1q2 = q[1]*q[2];
  q1q3 = q[1]*q[3];
  q2q3 = q[2]*q[3];

  R[0] = q02+q12-q22-q32;
  R[5] = q02+q22-q12-q32;
  R[10] = q02+q32-q12-q22;
  R[1] = 2*(q1q2-q0q3); R[4] = 2*(q1q2+q0q3);
  R[2] = 2*(q1q3+q0q2); R[8] = 2*(q1q3-q0q2);
  R[6] = 2*(q2q3-q0q1); R[9] = 2*(q2q3+q0q1);

  R[3] = R[7] = R[11] = R[12] = R[13] = R[14] = 0;
  R[15] = 1;
}

/**
 * Computes the optimal uniform scaling factor for the alignment of two point sets.
 * @param w is the weight array; w[i] is the weight of point i.
 * @param P is the coordinate array of the first point set.
 * @param Y is the coordinate array of the second point set.
 * @param n is the number of points contained in Y and P.
 */
double wHornScale(double* w, double* P, double *Y, int n){
  int i;
  double wmp[3];
  double wmy[3];
  double p[3],y[3];
  double sum1=0;
  double sum2=0;

  wmean(w,P,n,wmp);
  wmean(w,Y,n,wmy);
  if(w!=NULL){
    for(i=n; i-- > 0;){
      p[0] = P[3*i+0]-wmp[0];
      p[1] = P[3*i+1]-wmp[1];
      p[2] = P[3*i+2]-wmp[2];
      y[0] = Y[3*i+0]-wmy[0];
      y[1] = Y[3*i+1]-wmy[1];
      y[2] = Y[3*i+2]-wmy[2];
      sum1 += w[i]*dotProduct(p,p);
      sum2 += w[i]*dotProduct(y,y);
    }
  }else{
    for(i=n; i-- > 0;){
      p[0] = P[3*i+0]-wmp[0];
      p[1] = P[3*i+1]-wmp[1];
      p[2] = P[3*i+2]-wmp[2];
      y[0] = Y[3*i+0]-wmy[0];
      y[1] = Y[3*i+1]-wmy[1];
      y[2] = Y[3*i+2]-wmy[2];
      sum1 += dotProduct(p,p);
      sum2 += dotProduct(y,y);
    }
  }
  return sqrt(sum1/sum2);
}

/** \brief Horn's alignment method.
  First the weighted centroids of the two point sets are computed,
  then the weighted covariance matrix is found. From that we get
  the best rotation. Then the optimal scale is computed or is set to 1.
  The scaled rotation matrix is then combined with the optimal translation
  to obtain the final transformation.
  @param w is the weight array; w[i] is the weight of point i.
  @param P is the coordinate array of the first point set.
  @param Y is the coordinate array of the second point set.
  @param n is the number of points contained in Y and P.
  @param options is the bitwise OR of flags that affect the alignment.
  @param R contains at the end the matrix that aligns the two point sets.
 */
double wHornAlign(double *w, double* P, double *Y, int n, int options, double R[4][4]){
  double q[4];
  double wmp[3];
  double wmy[3];
  double spy[3][3];
  double s;

  wmean(w,P,n,wmp);
  wmean(w,Y,n,wmy);
  wcov(w,P,wmp,Y,wmy,n,spy);

  hornQuaternion(spy,q);

  if((options & ALIGN_FOR_SPIN) && q[0] < 0.7){
    double norm=sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    double f = q[0]/norm;
    q[0]=norm;
    q[1]=-f*q[1];
    q[2]=-f*q[2];
    q[3]=-f*q[3];
  }

  qRotation(q,R);

  if(options & ALIGN_WITH_SCALE){
    s = wHornScale(w, P, Y, n);
    if(s>MAXSCALE || s <MINSCALE)
      s=1;
  }
  else
    s = 1;

/* Uncomment these lines if you need scaling */
/*
  R[0][0]*=s;
  R[0][1]*=s;
  R[0][2]*=s;
  R[1][0]*=s;
  R[1][1]*=s;
  R[1][2]*=s;
  R[2][0]*=s;
  R[2][1]*=s;
  R[2][2]*=s;
*/

  R[0][3] = wmy[0] - (R[0][0]*wmp[0]+R[0][1]*wmp[1]+R[0][2]*wmp[2]);
  R[1][3] = wmy[1] - (R[1][0]*wmp[0]+R[1][1]*wmp[1]+R[1][2]*wmp[2]);
  R[2][3] = wmy[2] - (R[2][0]*wmp[0]+R[2][1]*wmp[1]+R[2][2]*wmp[2]);

  return s;
}

/**
 * Horn's alignment method.
 * First the weighted centroids of the two point sets are computed,
 * then the weighted covariance matrix is found. From that we get
 * the best rotation. Then the optimal scale is computed or is set to 1.
 * The scaled rotation matrix is then combined with the optimal translation
 * to obtain the final transformation.
 * @param w is the weight array; w[i] is the weight of point i.
 * @param P is the coordinate array of the first point set.
 * @param Y is the coordinate array of the second point set.
 * @param n is the number of points contained in Y and P.
 * @param options is the bitwise OR of flags that affect the alignment.
 * @param R contains at the end the matrix that aligns the two point sets.
 */
double wHornAlign(double *w, double* P, double *Y, int n, int options, double *R){
  if(n<=0){
    memset(R,0,sizeof(double)*16);
    for(int i=0;i<4;++i){
      R[5*i]=1;
    }
    return 1;
  }
  double q[4];
  double wmp[3];
  double wmy[3];
  double spy[3][3];
  double s;

  wmean(w,P,n,wmp);
  wmean(w,Y,n,wmy);
  wcov(w,P,wmp,Y,wmy,n,spy);

  hornQuaternion(spy,q);

  if((options & ALIGN_FOR_SPIN) && q[0] < 0.7){
    double norm=sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    double f = q[0]/norm;
    q[0]=norm;
    q[1]=-f*q[1];
    q[2]=-f*q[2];
    q[3]=-f*q[3];
  }

  qRotation(q,R);

  if(options & ALIGN_WITH_SCALE){
    s = wHornScale(w, P, Y, n);
    if(s>MAXSCALE || s <MINSCALE)
      s=1;
  }
  else
    s = 1;

  R[3] =  wmy[0] - (R[0]*wmp[0]+R[1]*wmp[1]+R[2] *wmp[2]);
  R[7] =  wmy[1] - (R[4]*wmp[0]+R[5]*wmp[1]+R[6] *wmp[2]);
  R[11] = wmy[2] - (R[8]*wmp[0]+R[9]*wmp[1]+R[10]*wmp[2]);

  return s;
}


double HornAlign(double* P, double *reference, int n, int options, double *R){
	/*double *w=new double[n];
	for(int i=0;i<n;++i){
		w[i]=1;
	}*/
	double retVal=wHornAlign(NULL, P, reference, n, options, R);
	//delete[] w;
	return retVal;
	
}