#include "icp.h"
#include "string.h"
#include "hungarian.h"
#include "stdio.h"
#include "math.h"
#include "macros.h"
#include "hornalign.h"
#include <iostream>
using namespace std;

//aligns all 3D shapes to reference by a rigid transformation. If reference is NULL, then the first shape is taken as reference
void icp(double *reference, double *shapes, int nshapes, int npoints, double *aligningErrors){
	if(reference==NULL){
		reference=shapes;
		shapes+=npoints*3;
		--nshapes;
		if(aligningErrors!=NULL){
			aligningErrors[0]=0;
			++aligningErrors;
		}
	}
	Hungarian hungarianSolver(npoints);	
	int *assignment=new int[npoints];
	double *tmpShape=new double[3*npoints];
	double tolerance=1e-8;
	//FILE *F=fopen("aligning_errors.txt", "w");
	for(int s=0;s<nshapes;++s){
		double *currentShape=&shapes[s*npoints*3];
		double error=1e10;
		double newError=1e9;
		bool firstRun=true;
		double initialError;
		while(fabs(error-newError)>tolerance){
			//----------assignment-----------
			hungarianSolver.setAllCosts(1e3);
			for(int i=0;i<npoints;++i){
				for(int j=0;j<npoints;++j){
					double dst=	SQR(reference[3*i+0]-currentShape[3*j+0])+ 
								SQR(reference[3*i+1]-currentShape[3*j+1])+
								SQR(reference[3*i+2]-currentShape[3*j+2]);
					hungarianSolver.setCost(i,j,dst);
				}
			}
			double cost=hungarianSolver.solve();
			hungarianSolver.getAssignment(assignment);
			//---apply permutation---
			for(int i=0;i<npoints;++i){
				memcpy(&tmpShape[3*i], &currentShape[3*assignment[i]], sizeof(double)*3);
			}
			//-----------------------
			double R[16];
			HornAlign(tmpShape, reference, npoints, 0, R);
			//---apply transformation---
			for(int p=0;p<npoints;++p){
				double *rotated=&currentShape[3*p];
				memset(rotated, 0, sizeof(double)*3);
				for(int i=0;i<3;++i){
					rotated[i]=R[i*4+3];
					for(int j=0;j<3;++j){
						rotated[i]+=R[i*4+j]*tmpShape[3*p+j];
					}
				}
			}
			//---------compute error--------
			error=newError;
			newError=0;
			for(int p=3*npoints-1;p>=0;--p){
				newError+=SQR(currentShape[p]-reference[p]);
			}
			newError/=npoints;
			if(firstRun){
				initialError=newError;
				firstRun=false;
			}
		}
		if(aligningErrors!=NULL){
			aligningErrors[s]=newError;
		}
		
		/*if(fabs(initialError-newError)>1e-9){
			if(fabs(initialError)<1e-10){
				cerr<<"*"<<0<<endl;
			}else{
				cerr<<(initialError-newError)/initialError<<endl;
			}
		}*/
		//fprintf(F,"%lf\n", newError);
		
	}
	//fclose(F);
	delete[] assignment;
	delete[] tmpShape;
}
