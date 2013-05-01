#include "SphericalHarmonics.h"
#include <string.h>
#include <iostream>
#include "macros.h"
#include "gsl_prototypes.h"
using namespace std;

namespace SphericalHarmonics{
	void initTransformation(double *directions, int nrows, int lmax, double *sht){
		if(directions==NULL){
			return;
		}
		int ncols=NforL(lmax);
		for(int l=0; l<=lmax; l+=2){
			for(int m=0; m<=l; ++m){
				for(int i = 0; i < nrows; i++){
					double s=gsl_sf_legendre_sphPlm (l, m, cos(directions[2*i+1]));
					if(m){
						sht[i*ncols+index(l, m)] = s*cos(m*directions[2*i]);
						sht[i*ncols+index(l, -m)] = s*sin(m*directions[2*i]);
					}else{
						sht[i*ncols+index(l, 0)] = s;
					}
				}
			}
		}
	}


	double value(double azimuth, double elevation, int l, int m){
		elevation = gsl_sf_legendre_sphPlm(l, abs(m), cos(elevation));
		if(m==0){
			return elevation;
		} 
		if(m>0){
			return elevation*cos(m*azimuth);
		}
		return (elevation * sin (-m*azimuth));
	}


	void SH2RH(const std::vector<double> &SH, std::vector<double> &RH){
		RH.resize(SH.size());
		if(SH.empty()){
			return;
		}
		for (int l=SH.size()-1;l>=0;--l){
			RH[l] = SH[l]/value(0.0, 0.0, 2*l, 0);
		}
	}


	void getAzimuthElevationPairs(double *gradients, int ngrads, double *directions){
		double *g=gradients;
		double *d=directions;
		for(int i=0;i<ngrads;++i, g+=3, d+=2){
			double norm=sqrt(SQR(g[0])+SQR(g[1])+SQR(g[2]));
			d[0]=atan2(g[1], g[0]);
			d[1]=acos(g[2]/norm);
		}
	}
	void getAmplitudeAndAzimuthElevationPairs(double *gradients, int ngrads, double *directions, double *amplitudes){
		double *g=gradients;
		double *d=directions;
		for(int i=0;i<ngrads;++i, g+=3, d+=2){
			double norm=sqrt(SQR(g[0])+SQR(g[1])+SQR(g[2]));
			amplitudes[i]=norm;
			d[0]=atan2(g[1], g[0]);
			d[1]=acos(g[2]/norm);
		}
	}
	

	
}
