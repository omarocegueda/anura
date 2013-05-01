#include "dtisynthetic.h"
#include "string.h"
#include "linearalgebra.h"
#include "statisticsutils.h"
#include <algorithm>
#include <math.h>
#include <vector>
#include <set>
using namespace std;
void selectPDDs(double *pdds, int n, int k, double minAngle, double *selected, int &pStart){
	memcpy(selected, &pdds[3*pStart], sizeof(double)*3);//select first
	pStart=(pStart+1)%n;
	for(int i=1;i<k;++i){//select the rest
		bool valid=false;
		while(!valid){
			double *p=&pdds[3*pStart];
			valid=true;
			double *q=selected;
			for(int j=0;j<i;++j, q+=3){//check the angle between p and the directions previously selected is at least minAngle
				double prod=dotProduct(p, q);
				double angle=acos(fabs(prod));
				if(angle<minAngle){
					valid=false;
					break;
				}
			}
			if(valid){//then p is valid: copy to selected and the loop finishes
				memcpy(&selected[3*i], p, sizeof(double)*3);
			}
			pStart=(pStart+1)%n;//either p was valid or not, in any case we discard it as the next selection
		}
	}
}

void createRealisticSyntheticSignal(double *pdds, double *amount, int npdds, double *orientations, int nOrientations, double b, double longDiffusion, double transDiffusion, double snr, double s0, double *s, double &nS0){
	double e0[3]={1,0,0};
	double D[9]={
		longDiffusion, 0, 0,
		0, transDiffusion, 0,
		0, 0, transDiffusion
	};
	memset(s,0,sizeof(double)*nOrientations);
	for(int i=0;i<npdds;++i){
		double *dir=&pdds[3*i];
		double T[9];
		fromToRotation(e0, dir, T);
		double Ti[9]={//Ti=D*T'
			T[0]*longDiffusion, T[3]*longDiffusion, T[6]*longDiffusion,
			T[1]*transDiffusion, T[4]*transDiffusion, T[7]*transDiffusion,
			T[2]*transDiffusion, T[5]*transDiffusion, T[8]*transDiffusion
		};
		multMatrixMatrix<double>(T,Ti,3,Ti);
		for(int j=0;j<nOrientations;++j){
			double *g=&orientations[3*j];
			double eval=evaluateQuadraticForm(Ti, g, 3);
			s[j]+=amount[i]*exp(-b*eval);
		}
	}
	double sigma=s0/snr;
	for(int j=0;j<nOrientations;++j){
		s[j]=generateRician(s[j]*s0, sigma);
	}
	nS0=generateRician(s0, sigma);
}

void createRealisticSyntheticField(int nrows2, int nrows1, int ncols, double *orientations, int nOrientations, double *randomPDDs, int nRandom, double minAngle, double longDiffusion, double transDiffusion, double b, double snr, double s0, 
								double *field, double *S0, double *pdds, double *amount, int *tcount){
	int nrows=nrows1+nrows2;
	int pStart=0;
	double *s=field;
	double *nS0=S0;
	int maxPdds=2;
	for(int i=0;i<nrows2;++i){//select 2 tensors
		for(int j=0;j<ncols;++j, s+=nOrientations, ++nS0){
			tcount[i*ncols+j]=2;
			double *currentPdds=&pdds[maxPdds*3*(i*ncols+j)];
			double *currentAmounts=&amount[maxPdds*(i*ncols+j)];
			memset(currentPdds,0,sizeof(double)*maxPdds*3);
			memset(currentAmounts,0,sizeof(double)*maxPdds);
			selectPDDs(randomPDDs, nRandom, 2, minAngle, currentPdds, pStart);
			currentAmounts[0]=uniform(0.4,0.6);
			currentAmounts[1]=1.0-currentAmounts[0];
			createRealisticSyntheticSignal(currentPdds, currentAmounts, 2, orientations, nOrientations, b, longDiffusion, transDiffusion, snr, s0, s, *nS0);
		}
	}
	pStart=0;
	
	for(int i=0;i<nrows1;++i){//select 2 tensors
		for(int j=0;j<ncols;++j, s+=nOrientations, ++nS0){
			tcount[(i+nrows2)*ncols+j]=1;
			double *currentPdds=&pdds[maxPdds*3*((i+nrows2)*ncols+j)];
			double *currentAmounts=&amount[maxPdds*((i+nrows2)*ncols+j)];
			memset(currentPdds,0,sizeof(double)*maxPdds*3);
			memset(currentAmounts,0,sizeof(double)*maxPdds);
			selectPDDs(randomPDDs, nRandom, 1, minAngle, currentPdds, pStart);
			currentAmounts[0]=1;
			currentAmounts[1]=0;
			createRealisticSyntheticSignal(currentPdds, currentAmounts, 1, orientations, nOrientations, b, longDiffusion, transDiffusion,snr, s0, s,*nS0);
		}
	}
}







