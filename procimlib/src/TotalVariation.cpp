#include "TotalVariation.h"
#include <string.h>
#include <math.h>
#include "derivatives.h"
#ifndef MAX
#define MAX(a,b) (((a)<(b))?(b):(a))
#endif

void filterTotalVariation_L2(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-9;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				if(nrm>1){
					yr[i*ncols+j]/=nrm;
					yc[i*ncols+j]/=nrm;
				}
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				x[i*ncols+j]+=tau*divergence[i*ncols+j];
				x[i*ncols+j]=(x[i*ncols+j]+tau*lambda*g[i*ncols+j])/(1.0+tau*lambda);
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}

void filterTotalVariation_L1(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-5;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				//double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				double nrm=MAX(fabs(yr[i*ncols+j]), fabs(yc[i*ncols+j]));
				if(nrm<1){
					nrm=1;
				}
				yr[i*ncols+j]/=nrm;
				yc[i*ncols+j]/=nrm;
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				double arg=x[i*ncols+j]+tau*divergence[i*ncols+j];
				double obs=g[i*ncols+j];
				if((arg-obs)>(tau*lambda)){
					x[i*ncols+j]=arg-tau*lambda;
				}else if((arg-obs)<-(tau*lambda)){
					x[i*ncols+j]=arg+tau*lambda;
				}else{
					x[i*ncols+j]=obs;
				}
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
		/*if((callback!=NULL) && (iter%20==0)){
			callback(x, nrows, ncols);
		}*/
		
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}


void filterHuber_L2(double *g, int nrows, int ncols, double alpha, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-4;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				yr[i*ncols+j]/=(1+sigma*alpha);
				yc[i*ncols+j]/=(1+sigma*alpha);
				double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				if(nrm<1){
					nrm=1;
				}
				yr[i*ncols+j]/=nrm;
				yc[i*ncols+j]/=nrm;
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				x[i*ncols+j]+=tau*divergence[i*ncols+j];
				x[i*ncols+j]=(x[i*ncols+j]+tau*lambda*g[i*ncols+j])/(1.0+tau*lambda);
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
		/*if((callback!=NULL) && (iter%20==0)){
			callback(x, nrows, ncols);
		}*/
		
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}

void filterTGV_L2(double *g, int nrows, int ncols, double lambda, double alpha0, double alpha1, double tau, double sigma, double theta, double *u, double *v){
	//---primal variables---
	double *dubar_dr=new double[nrows*ncols];
	double *dubar_dc=new double[nrows*ncols];
	double *ubar=new double[nrows*ncols];
	double *vr=new double[nrows*ncols];
	double *vc=new double[nrows*ncols];
	double *vr_bar=new double[nrows*ncols];
	double *vc_bar=new double[nrows*ncols];
	double *dvr_bar_dr=new double[nrows*ncols];
	double *dvr_bar_dc=new double[nrows*ncols];
	double *dvc_bar_dr=new double[nrows*ncols];
	double *dvc_bar_dc=new double[nrows*ncols];
	
	memset(ubar, 0, sizeof(double)*nrows*ncols);
	memset(vr, 0, sizeof(double)*nrows*ncols);
	memset(vc, 0, sizeof(double)*nrows*ncols);
	memset(vr_bar, 0, sizeof(double)*nrows*ncols);
	memset(vc_bar, 0, sizeof(double)*nrows*ncols);
	memset(dvr_bar_dr, 0, sizeof(double)*nrows*ncols);
	memset(dvr_bar_dc, 0, sizeof(double)*nrows*ncols);
	memset(dvc_bar_dr, 0, sizeof(double)*nrows*ncols);
	memset(dvc_bar_dc, 0, sizeof(double)*nrows*ncols);
	//---dual variables-----
	double *pr=new double[nrows*ncols];
	double *pc=new double[nrows*ncols];
	double *qrr=new double[nrows*ncols];
	double *qcc=new double[nrows*ncols];
	double *qrc=new double[nrows*ncols];

	memset(pr, 0, sizeof(double)*nrows*ncols);
	memset(pc, 0, sizeof(double)*nrows*ncols);
	memset(qrr, 0, sizeof(double)*nrows*ncols);
	memset(qcc, 0, sizeof(double)*nrows*ncols);
	memset(qrc, 0, sizeof(double)*nrows*ncols);
	//-----------------
	double *divP=new double[nrows*ncols];
	double *divQr=new double[nrows*ncols];
	double *divQc=new double[nrows*ncols];
	double tolerance=1e-9;
	double error=1+tolerance;
	int maxIter=500;
	int iter=0;
	
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(ubar, nrows, ncols, dubar_dr, dubar_dc);
		computeGradient(vr_bar, nrows, ncols, dvr_bar_dr, dvr_bar_dc);
		computeGradient(vc_bar, nrows, ncols, dvc_bar_dr, dvc_bar_dc);
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				//--update [pr,pc]--
				pr[pos]+=sigma*(dubar_dr[pos]-vr[pos]);
				pc[pos]+=sigma*(dubar_dc[pos]-vc[pos]);
				double nrm=pr[pos]*pr[pos]+pc[pos]*pc[pos];
				nrm=sqrt(nrm)/alpha0;
				if(nrm>1){
					pr[i*ncols+j]/=nrm;
					pc[i*ncols+j]/=nrm;
				}
				//--update [qrr,qrc,qcc]--
				qrr[pos]+=sigma*(dvr_bar_dr[pos]);
				qcc[pos]+=sigma*(dvc_bar_dc[pos]);
				qrc[pos]+=sigma*0.5*(dvr_bar_dc[pos]+dvc_bar_dr[pos]);
				nrm=qrr[pos]*qrr[pos] + qcc[pos]*qcc[pos] + 2.0*qrc[pos]*qrc[pos];
				nrm=sqrt(nrm)/alpha1;
				if(nrm>1){
					qrr[pos]/=nrm;
					qcc[pos]/=nrm;
					qrc[pos]/=nrm;
				}
				//------------------------
			}
		}
		//update primal field
		computeDivergence(pr, pc, nrows, ncols, divP);
		computeDivergence(qrr, qrc, nrows, ncols, divQr);
		computeDivergence(qrc, qcc, nrows, ncols, divQc);
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				//--update u--
				double diff=-u[i*ncols+j];
				u[pos]+=tau*divP[pos];
				u[pos]=(u[pos]+tau*lambda*g[pos])/(1.0+tau*lambda);
				//update ubar
				diff+=u[pos];
				error+=diff*diff;
				ubar[pos]=u[pos]+theta*diff;
				//--update v--
				double diff_vr=-vr[pos];
				double diff_vc=-vc[pos];
				vr[i*ncols+j]+=tau*(pr[pos]+divQr[pos]);
				vc[i*ncols+j]+=tau*(pc[pos]+divQc[pos]);
				//update vbar
				diff_vr+=vr[pos];
				diff_vc+=vc[pos];
				vr_bar[pos]=vr[pos]+theta*diff_vr;
				vc_bar[pos]=vc[pos]+theta*diff_vc;
				//------------
			}
		}
	}
	if(v!=NULL){
		memcpy(v, vr, sizeof(double)*nrows*ncols);
		memcpy(&v[nrows*ncols], vc, sizeof(double)*nrows*ncols);
	}
	delete[] dubar_dr;
	delete[] dubar_dc;
	delete[] ubar;
	delete[] vr;
	delete[] vc;
	delete[] vr_bar;
	delete[] vc_bar;
	delete[] dvr_bar_dr;
	delete[] dvr_bar_dc;
	delete[] dvc_bar_dr;
	delete[] dvc_bar_dc;
	delete[] pr;
	delete[] pc;
	delete[] qrr;
	delete[] qcc;
	delete[] qrc;
	delete[] divP;
	delete[] divQr;
	delete[] divQc;
}
