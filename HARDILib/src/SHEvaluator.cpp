#ifdef USE_GSL
#include "SHEvaluator.h"
#include "SphericalHarmonics.h"
#include "linearalgebra.h"
#include "geometryutils.h"
#include "gsl_prototypes.h"
#include "math.h"

SHEvaluator::SHEvaluator(double *directions, int _numDirections, int lmax){
	numDirections=_numDirections;
	lmax_computed=lmax;
	nsh_computed=SphericalHarmonics::NforL(lmax_computed);
	rowLength=3*(nsh_computed+1);
	rows=new double[numDirections*rowLength];
	for(int i=0;i<numDirections;++i){
		double *row=&rows[i*rowLength];
		memcpy(row, &directions[3*i], sizeof(double)*3);
		precomputeRow(row);
	}
}

double *SHEvaluator::getRow(int n){
	return &rows[n*rowLength];
}

double *SHEvaluator::getRadiiPointer(int n){
	return &rows[n*rowLength +3];
}

double *SHEvaluator::getDazPointer(int n){
	return &rows[n*rowLength +3+nsh_computed];
}

double *SHEvaluator::getDelPointer(int n){
	return &rows[n*rowLength +3+2*nsh_computed];
}

//Addapted from MRTrix. Expects an array of size 3*(nsh+1)
void SHEvaluator::precomputeRow(double* row){
	normalize(row, 3);
	double* r	=row+3;
	double* daz	=row+3+nsh_computed;
	double* del	=row+3+nsh_computed*2;
	memset (r, 0, 3*nsh_computed*sizeof(double));

	for(int l=0;l<=lmax_computed; l+=2){
		for (int m=0;m<=l;++m){
			int idx=SphericalHarmonics::index(l,m);
			r[idx] = gsl_sf_legendre_sphPlm (l, m, row[2]);
			if(m>0){
				r[idx-2*m] = r[idx];
			} 
		}
	}

	bool atpole=(fabs(row[0])<SphericalHarmonics::EPSILON) && (fabs(row[1])<SphericalHarmonics::EPSILON);
	double az=0;
	if(!atpole){
		az=atan2 (row[1], row[0]);
	}
	for(int l=2;l<=lmax_computed;l+=2){
		int idx=SphericalHarmonics::index(l,0);
		del[idx]=r[idx+1]*sqrt(double(l*(l+1)));
	}

	for(int m=1;m<=lmax_computed;m++){
		double caz=cos(m*az); 
		double saz=sin(m*az); 
		for (int l=2*((m+1)/2);l<=lmax_computed;l+=2){
			int idx=SphericalHarmonics::index(l,m);
			del[idx]=-r[idx-1]*sqrt(double((l+m)*(l-m+1)));
			if(l>m){
				del[idx]+=r[idx+1]*sqrt(double((l-m)*(l+m+1)));
			}
			del[idx]/= 2.0;
			int idx2=idx-2*m;
			if(atpole){
				daz[idx]=-del[idx]*saz;
				daz[idx2]=del[idx]*caz;
			}
			else {
				double tmp=m*r[idx];
				daz[idx]=-tmp*saz;
				daz[idx2]=tmp*caz;
			}
			del[idx2]=del[idx]*saz;
			del[idx]*=caz;
		}
	}

	for(int m=1;m<=lmax_computed;++m){
		double caz =cos(m*az); 
		double saz =sin(m*az); 
		for(int l=2*((m+1)/2);l<=lmax_computed;l+=2){
			int idx=SphericalHarmonics::index(l,m);
			r[idx]*=caz;
			r[idx-2*m]*=saz;
		}
	}
}

int SHEvaluator::evaluateFunction_vertices(double *SHCoeffs, int numCoeffs, int lmax, double *vertices){
	int actual_lmax=SphericalHarmonics::LforN(numCoeffs);
	actual_lmax=MIN(actual_lmax, MIN(lmax, lmax_computed));
	int nsh=SphericalHarmonics::NforL(actual_lmax);

	for(int n=0;n<numDirections;++n){
		double *v=&vertices[3*n];
		double* row=getRow(n);
		double* row_r=getRadiiPointer(n);
        double r=0;
        for (int i=0;i<nsh;++i){
			r+=row_r[i]*SHCoeffs[i]; 
		}
        v[0] = r*row[0];
        v[1] = r*row[1];
        v[2] = r*row[2];
	}
	return 0;
}
int SHEvaluator::evaluateFunction_amplitudes(double *SHCoeffs, int numCoeffs, int lmax, double *amplitudes){
	int actual_lmax=SphericalHarmonics::LforN(numCoeffs);
	actual_lmax=MIN(actual_lmax, MIN(lmax, lmax_computed));
	int nsh=SphericalHarmonics::NforL(actual_lmax);

	for(int n=0;n<numDirections;++n){
		double* row=getRow(n);
		double* row_r=getRadiiPointer(n);
        double r=0;
        for (int i=0;i<nsh;++i){
			r+=row_r[i]*SHCoeffs[i]; 
		}
        amplitudes[n]=r;
	}
	return 0;
}

int SHEvaluator::evaluateFunctionAndNormals(double *SHCoeffs, int numCoeffs, int lmax, double *vertices, double *normals){
	int actual_lmax=SphericalHarmonics::LforN(numCoeffs);
	actual_lmax=MIN(actual_lmax, MIN(lmax, lmax_computed));
	int nsh=SphericalHarmonics::NforL(actual_lmax);

	for(int n=0;n<numDirections;++n){
		double *v=&vertices[3*n];
		double *N=&normals[3*n];
		double* row=getRow(n);
		double* row_r=getRadiiPointer(n);
		double* row_daz=getDazPointer(n);
        double* row_del=getDelPointer(n);
        double daz=0, del=0;
		double r=0;
        for (int i=0;i<nsh;++i){
			r+=row_r[i]*SHCoeffs[i]; 
			daz+=row_daz[i]*SHCoeffs[i]; 
			del+=row_del[i]*SHCoeffs[i]; 
		}
		v[0] = r*row[0];
        v[1] = r*row[1];
        v[2] = r*row[2];
        bool atpole=(row[0] == 0.0) && (row[1] == 0.0);
        double az=0;
		if(atpole){
			az=atan2(row[1], row[0]);
		}

        double caz=cos(az);
        double saz=sin(az);
        double cel=row[2];
        double sel=sqrt(1.0 - SQR(cel));
        double d1[3], d2[3];
        if(atpole){
          d1[0]=-r*saz;
          d1[1]=r*caz;
          d1[2]=daz;
        }else{
          d1[0]=daz*caz*sel-r*sel*saz;
          d1[1]=daz*saz*sel+r*sel*caz;
          d1[2]=daz*cel;
        }
        d2[0]=-del*caz*sel-r*caz*cel;
        d2[1]=-del*saz*sel-r*saz*cel;
        d2[2]=-del*cel+r*sel;

        crossProduct(d1,d2,N);
	}
	return 0;
}

SHEvaluator::~SHEvaluator(){
	DELETE_ARRAY(rows);
}


#endif
