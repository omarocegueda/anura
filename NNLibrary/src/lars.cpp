#include "lars.h"
#include "linearalgebra.h"
#include "nnls.h"
#include "ls.h"
#include "geometryutils.h"
#include "utilities.h"
#include <set>
#include <math.h>
#include <iostream>
#define INF_LARS 1e+200
#define EPS_LARS 1e-9
using namespace std;
//#define LARS_PRINT_PATHS
//#define LARS_DEBUG
//computes the correlation of each column of X with y
void computeCorrelations(double *X, int n, int m, double *y, double *correlations){
	for(int j=0;j<m;++j){
		double &c=correlations[j];
		c=0;
		double *x=&X[j];
		for(int i=0;i<n;++i, x+=m){
			c+=(*x)*y[i];
		}
	}
}


//computes the correlation of each column of X with (y-mu)
void computeCorrelations(double *X, int n, int m, double *mu, double *y, double *correlations, double &maxVal){
	maxVal=-INF_LARS;
	for(int j=0;j<m;++j){
		double &c=correlations[j];
		c=0;
		double *x=&X[j];
		for(int i=0;i<n;++i, x+=m){
			c+=(*x)*(y[i]-mu[i]);
		}
		if(maxVal<c){
			maxVal=c;
		}
	}
}

void computeCorrelations_ABS(double *X, int n, int m, double *mu, double *y, double *correlations, int *signs, double &maxVal){
	maxVal=-INF_LARS;
	for(int j=0;j<m;++j){
		double &c=correlations[j];
		c=0;
		double *x=&X[j];
		for(int i=0;i<n;++i, x+=m){
			c+=(*x)*(y[i]-mu[i]);
		}
		if(c<0){
			signs[j]=-1;
		}else{
			signs[j]=1;
		}
		if(maxVal<fabs(c)){
			maxVal=fabs(c);
		}
	}
}

int populateActiveSet(double *correlations, int n, double maxVal, set<int> &active, set<int> &activeC){
/*	int added=0;
#ifdef LARS_DEBUG
	double maxNotAdded=-1;
#endif
	for(int i=0;i<n;++i){
#ifdef LARS_DEBUG
		if((maxVal-EPS_LARS>=correlations[i]) &&(active.find(i)==active.end()) && (maxNotAdded<correlations[i])){
			maxNotAdded=correlations[i];
		}
#endif
		if(maxVal-EPS_LARS<correlations[i]){//it is a maximum
			if(active.find(i)==active.end()){
				++added;
				active.insert(i);
				activeC.erase(i);
			}
		}
	}
	return added;
*/
	maxVal=-1e10;
	int added=0;
	int selected=-1;
	for(set<int>::iterator it=activeC.begin();it!=activeC.end();++it){
		if(maxVal<correlations[*it]){
			added=1;
			maxVal=correlations[*it];
			selected=*it;
		}else if(maxVal==correlations[*it]){
			++added;
		}
	}
	active.insert(selected);
	activeC.erase(selected);
	return added;
}

int populateActiveSet_ABS(double *correlations, int n, double maxVal, set<int> &active, set<int> &activeC){
	int added=0;
	for(int i=0;i<n;++i)if(maxVal-EPS_LARS<fabs(correlations[i])){//it is a maximum
		if(active.find(i)==active.end()){
			++added;
			active.insert(i);
			activeC.erase(i);
		}
	}
	return added;
}

void computeEquiangularVector(double *X, int n, int m, int *I, int mm, double *XX, double *u, double &AA, double *dbeta){
	for(int i=0;i<mm;++i){
		for(int j=i;j<mm;++j){
			double *xi=&X[I[i]];
			double *xj=&X[I[j]];
			double &s=XX[i*mm+j];
			s=0;
			for(int k=0;k<n;++k, xi+=m, xj+=m){
				s+=(*xi)*(*xj);
			}
			XX[j*mm+i]=s;
		}
	}
	computeInverse(XX,mm);
	AA=0;
	for(int i=mm*mm-1;i>=0;--i){
		AA+=XX[i];
	}
	
	
	if(AA<EPS_LARS){
		memset(dbeta,0, sizeof(double)*mm);
		return;
	}

	AA=1.0/sqrt(AA);
	memset(u, 0, sizeof(double)*n);
	for(int ii=0;ii<mm;++ii){
		double *x=&XX[ii*mm];
		double sum=0;
		for(int j=0;j<mm;++j){
			sum+=x[j];
		}
		sum*=AA;
		x=&X[I[ii]];
		for(int i=0;i<n;++i, x+=m){
			u[i]+=(*x)*sum;
		}
		dbeta[ii]=sum;
	}
}

void computeEquiangularVector_ABS(double *X, int n, int m, int *signs, int *I, int mm, double *XX, double *u, double &AA, double *dbeta){
	for(int i=0;i<mm;++i){
		for(int j=i;j<mm;++j){
			int sign=signs[I[i]]*signs[I[j]];
			double *xi=&X[I[i]];
			double *xj=&X[I[j]];
			double &s=XX[i*mm+j];
			s=0;
			for(int k=0;k<n;++k, xi+=m, xj+=m){
				s+=(*xi)*(*xj);
			}
			s*=sign;
			XX[j*mm+i]=s;
		}
	}
	computeInverse(XX,mm);
	AA=0;
	for(int i=mm*mm-1;i>=0;--i){
		AA+=XX[i];
	}
	AA=1.0/sqrt(AA);
	memset(u, 0, sizeof(double)*n);

	for(int ii=0;ii<mm;++ii){
		double *x=&XX[ii*mm];
		double sum=0;
		for(int j=0;j<mm;++j){
			sum+=x[j];
		}
		sum*=AA*signs[I[ii]];
		x=&X[I[ii]];
		for(int i=0;i<n;++i, x+=m){
			u[i]+=(*x)*sum;
		}
		dbeta[ii]=sum;
	}
}



int lars(double *X, int n, int m, double *y, double *beta, double lambda){
	//---starting point---
	memset(beta, 0, sizeof(double)*m);
	//--------------------
	double *mu=new double[n];
	double *correlations=new double[m];
	double *XX=new double[m*m];
	double *u=new double[n];
	double *a=new double[m];
	double *dbeta=new double[m];
	int *I=new int[m];
	int *signs=new int[m];

	set<int> active;
	set<int> activeC;
	memset(mu, 0, sizeof(double)*n);
	for(int i=0;i<m;++i){
		activeC.insert(i);
	}
	bool finished=false;
	bool conflicted=false;
#ifdef LARS_PRINT_PATHS
	cout<<"LARS start"<<endl<<"0";
	for(int i=0;i<m;++i){
		cout<<"\t"<<0;
	}
	cout<<endl;
#endif
	while(!finished){
#ifdef LARS_DEBUG
		double sumAbs=0;
		for(int i=0;i<m;++i){
			sumAbs+=fabs(beta[i]);
		}
		cerr<<"Sum abs:"<<sumAbs<<endl;
		//---test---
		double residualNorm=0;
		for(int i=0;i<n;++i){residualNorm+=SQR(y[i]-mu[i]);}
		residualNorm=sqrt(residualNorm);
#endif		
#ifdef LARS_PRINT_PATHS
		//-------print path-----
		if(!(active.empty())){
			double abssum=0;
			for(int i=0;i<m;++i){
				abssum+=fabs(beta[i]);
			}
			cout<<abssum;
			for(int i=0;i<m;++i){
				cout<<"\t"<<beta[i];
			}
			cout<<endl;
		}
#endif
		//---compute correlations and find the maximum---
		double maxCorr;
		computeCorrelations_ABS(X, n, m, mu, y, correlations, signs, maxCorr);
		if(maxCorr<EPS_LARS){
			break;
		}
		//---populate active set and see how many new covariates enter the active set---
		int added=0;
		if(!conflicted){
			added=populateActiveSet_ABS(correlations, m, maxCorr, active, activeC);
		}
#ifdef LARS_DEBUG
		if((!conflicted)&&(added!=1)){
			cerr<<"Warning: "<<added<<" variables added at a lars step."<<endl;
		}
#endif
		conflicted=false;
		int mm=active.size();
		int ii=0;
		for(set<int>::iterator it=active.begin();it!=active.end();++it, ++ii){//put active indices in a plain array
			I[ii]=*it;
		}
		//---compute the equiangular vector and related data---
		double AA;
		computeEquiangularVector_ABS(X, n, m, signs, I, mm, XX, u, AA, dbeta);
		computeCorrelations(X, n, m, u, a);
		//---compute the min gamma factor ---
		double gamma=INF_LARS;
		if(mm<m){//active set is not full
			for(set<int>::iterator it=activeC.begin();it!=activeC.end();++it){
				if(fabs(AA-a[*it])>EPS_LARS){
					double opc=(maxCorr-correlations[*it])/(AA-a[*it]);
					if((opc>EPS_LARS) && (opc<gamma)){
						gamma=opc;
					}
				}
				if(fabs(AA+a[*it])>EPS_LARS){
					double opc=(maxCorr+correlations[*it])/(AA+a[*it]);
					if((opc>EPS_LARS) && (opc<gamma)){
						gamma=opc;
					}
				}
			}
		}else{
			gamma=maxCorr/AA;
		}
		//==============in the lasso modification, stop the step at the point at which a beta coefficient changes sign
		double gammaLASSO=INF_LARS;
		int coefficientToRemove=-1;
		for(int i=0;i<mm;++i)if(fabs(dbeta[i])>EPS_LARS){
			double gammaOpc=-beta[I[i]]/dbeta[i];//this is the point at which the i-th coefficient changes sign when moving in the equiangular direction
			if(gammaOpc>EPS_LARS){
				if(gammaOpc<gammaLASSO){
					gammaLASSO=gammaOpc;
					coefficientToRemove=I[i];
				}
			}
		}
		if(gammaLASSO<gamma){//then the LARS cannot be the LASSO solution because one coefficient changed signed while the correlation remained unchanged
			gamma=gammaLASSO;
			active.erase(coefficientToRemove);
			activeC.insert(coefficientToRemove);
#ifdef LARS_DEBUG
			cerr<<"Removed: "<<coefficientToRemove<<endl;
#endif
			conflicted=true;
		}
		double lambda0=0;
		double lambda1=0;
		for(int i=0;i<mm;++i){
			lambda0+=fabs(beta[I[i]]);
			lambda1+=fabs(beta[I[i]] + gamma*dbeta[i]);
		}

		if(lambda1>=lambda){
#ifdef LARS_DEBUG
			for(double dt=0;dt-EPS_LARS<1;dt+=0.01){
				for(int i=0;i<mm;++i){
					if(beta[I[i]]*(beta[I[i]] + dt*gamma*dbeta[i])<-EPS_LARS){
						cerr<<"Variable "<<I[i]<<" changed sign inside a single step of lars-lasso."<<endl;
					}
				}
			}
#endif
			double t=(lambda-lambda0)/(lambda1-lambda0);
			gamma=t*gamma;
			finished=true;
		}
		


		//update beta-coefficients
		for(int i=0;i<mm;++i){
			beta[I[i]]+=gamma*dbeta[i];
#ifdef LARS_DEBUG
			if(beta[I[i]]<-EPS_LARS){
				cerr<<"Negative coefficient:["<<I[i]<<"]="<<beta[I[i]]<<endl;
			}else if(beta[I[i]]<EPS_LARS){
				beta[I[i]]=0;
			}
#endif
		}

		for(int i=0;i<n;++i){
			mu[i]+=gamma*u[i];
		}
		//-----------------------------------------------
	}
	delete[] correlations;
	delete[] XX;
	delete[] u;
	delete[] a;
	delete[] dbeta;
	delete[] I;
	delete[] mu;
	delete[] signs;
	return 0;
}

void standardizeMatrix(double *X, int n, int m, double *XX, double *nu){
	for(int j=0;j<m;++j){
		double mu=0;
		for(int i=0;i<n;++i){
			mu+=X[i*m+j];
		}
		mu/=n;
		double nrm=0;
		for(int i=0;i<n;++i){
			nrm+=SQR(X[i*m+j]-mu);
		}
		nu[j]=1.0/sqrt(nrm);
		for(int i=0;i<n;++i){
			XX[i*m+j]=(X[i*m+j]-mu)*nu[j];
		}
	}
}

void standardizeVector(double *y, int n, double *yy){
	double mu=0;
	for(int i=0;i<n;++i){
		mu+=y[i];
	}
	mu/=n;
	for(int i=0;i<n;++i){
		yy[i]=y[i]-mu;
	}

}

void standardizeRegressionProblem(double *Phi, int n, int m, double *alpha){
}

int nnlars_old(double *inputX, int n, int m, double *inputy, double *beta, double lambda, double *error, bool standardize){
	double *X=inputX;
	double *y=inputy;
	double *nu=NULL;
	if(standardize){
		X=new double[m*n];
		y=new double[n];
		nu=new double[m];
		standardizeMatrix(inputX, n,m,X, nu);
		standardizeVector(inputy, n,y);
	}
	//---starting point---
	memset(beta, 0, sizeof(double)*m);
	//--------------------
	double *mu=new double[n];
	double *correlations=new double[m];
	double *XX=new double[m*m];
	double *u=new double[n];
	double *a=new double[m];
	double *dbeta=new double[m];
	int *I=new int[m];

	set<int> active;
	set<int> activeC;
	memset(mu, 0, sizeof(double)*n);
	for(int i=0;i<m;++i){
		activeC.insert(i);
	}
	bool finished=false;
	bool conflicted=false;
#ifdef LARS_PRINT_PATHS
	cout<<"NNLARS start"<<endl<<"0";
	for(int i=0;i<m;++i){
		cout<<"\t"<<0;
	}
	cout<<endl;
#endif
	while(!finished){
#ifdef LARS_DEBUG
		double sumAbs=0;
		for(int i=0;i<m;++i){
			sumAbs+=fabs(beta[i]);
		}
		cerr<<"Sum abs:"<<sumAbs<<endl;
		//---test---
		double residualNorm=0;
		for(int i=0;i<n;++i){residualNorm+=SQR(y[i]-mu[i]);}
		residualNorm=sqrt(residualNorm);
#endif
#ifdef LARS_PRINT_PATHS
		//-------print path-----
		if(!(active.empty())){
			double abssum=0;
			for(int i=0;i<m;++i){
				abssum+=fabs(beta[i]);
			}
			cout<<abssum;
			for(int i=0;i<m;++i){
				cout<<"\t"<<beta[i];
			}
			cout<<endl;
		}
#endif
		//---compute correlations and find the maximum---
		double maxCorr;
		computeCorrelations(X, n, m, mu, y, correlations, maxCorr);
		if(maxCorr<EPS_LARS){
			break;
		}
		//---populate active set and see how many new covariates enter the active set---
		int maximumCount=0;
		if(!conflicted){
			maximumCount=populateActiveSet(correlations, m, maxCorr, active, activeC);
		}
#ifdef LARS_DEBUG
		if((!conflicted)&&(maximumCount>1)){
			cerr<<"Warning [NNLARS]: "<<maximumCount<<" variables had max correlation but only one was added."<<endl;
		}
#endif
		conflicted=false;
		int mm=active.size();
		int ii=0;
		for(set<int>::iterator it=active.begin();it!=active.end();++it, ++ii){//put active indices in a plain array
			I[ii]=*it;
		}
		//---compute the equiangular vector and related data---
		double AA;
		computeEquiangularVector(X, n, m, I, mm, XX, u, AA, dbeta);
		computeCorrelations(X, n, m, u, a);
		//---compute the min gamma factor ---
		double gamma=INF_LARS;
		if(mm<m){//active set is not full
			for(set<int>::iterator it=activeC.begin();it!=activeC.end();++it){
				if(fabs(AA-a[*it])>EPS_LARS){
					double opc=(maxCorr-correlations[*it])/(AA-a[*it]);
					if((opc>EPS_LARS) && (opc<gamma)){
						gamma=opc;
					}
				}
			}
		}else{
			gamma=maxCorr/AA;
		}
		//==============in the lasso modification, stop the step at the point at which a beta coefficient changes sign
		double gammaLASSO=INF_LARS;
		int coefficientToRemove=-1;
		for(int i=0;i<mm;++i)if(fabs(dbeta[i])>EPS_LARS){
			double gammaOpc=-beta[I[i]]/dbeta[i];//this is the point at which the i-th coefficient changes sign when moving in the equiangular direction
			if(gammaOpc>EPS_LARS){
				if(gammaOpc<gammaLASSO){
					gammaLASSO=gammaOpc;
					coefficientToRemove=I[i];
				}
			}
		}
		if(gammaLASSO<gamma){//then the LARS cannot be the LASSO solution because one coefficient changed signed while the correlation remained unchanged
			gamma=gammaLASSO;
			active.erase(coefficientToRemove);
			activeC.insert(coefficientToRemove);
#ifdef LARS_DEBUG
			cerr<<"Removed: "<<coefficientToRemove<<endl;
#endif			
			conflicted=true;
		}
		double lambda0=0;
		double lambda1=0;
		for(int i=0;i<mm;++i){
			lambda0+=fabs(beta[I[i]]);
			lambda1+=fabs(beta[I[i]] + gamma*dbeta[i]);
#ifdef LARS_DEBUG
			if((beta[I[i]] + gamma*dbeta[i])<-EPS_LARS){
				cerr<<"Upcomming negative coefficient:["<<I[i]<<"]="<<beta[I[i]]<<endl;
			}
#endif
		}

		if(lambda1>=lambda){
#ifdef LARS_DEBUG
			for(double dt=0;dt-EPS_LARS<1;dt+=0.01){
				for(int i=0;i<mm;++i){
					if(beta[I[i]]*(beta[I[i]] + dt*gamma*dbeta[i])<-EPS_LARS){
						cerr<<"Variable "<<I[i]<<" changed sign inside a single step of lars-lasso."<<endl;
					}
				}
			}
#endif
			double t=(lambda-lambda0)/(lambda1-lambda0);
			gamma=t*gamma;
			finished=true;
		}
		//update beta-coefficients
		for(int i=0;i<mm;++i){
			beta[I[i]]+=gamma*dbeta[i];
#ifdef LARS_DEBUG
			if(beta[I[i]]<-EPS_LARS){
				cerr<<"Negative coefficient:["<<I[i]<<"]="<<beta[I[i]]<<endl;
			}
#endif
			if(beta[I[i]]<EPS_LARS){
				beta[I[i]]=0;
			}
		}
		//update y-approximation
		for(int i=0;i<n;++i){
			mu[i]+=gamma*u[i];
		}
		//-----------------------------------------------
	}
	
	if(error!=NULL){
		*error=0;
		multMatrixVector<double>(X,beta,n,m,u);
		for(int i=0;i<n;++i){
			*error+=SQR(u[i]-y[i]);
		}
		*error=sqrt(*error/n);
	}
	//----clean up----
	if(standardize){
		for(int j=0;j<m;++j){
			beta[j]*=nu[j];
		}
		delete[] X;
		delete[] y;
		delete[] nu;
	}
	delete[] correlations;
	delete[] XX;
	delete[] u;
	delete[] a;
	delete[] dbeta;
	delete[] I;
	delete[] mu;
	return 0;
}

int nnlars(double *inputX, int n, int m, double *inputy, double *beta, double lambda, double *error, bool standardize){
	//--
	double *X=inputX;
	double *y=inputy;
	//===========pre-compute products=============
	double *Xty=new double[m];
	multVectorMatrix<double>(y,X,n,m,Xty);
	double *XtX=new double[m*m];
	for(int i=0;i<m;++i){
		for(int j=0;j<m;++j){
			double *xi=&X[i];
			double *xj=&X[j];
			double &s=XtX[i*m+j];
			s=0;
			for(int k=0;k<n;++k, xi+=m, xj+=m){
				s+=*xi**xj;
			}
		}
	}
	//-----compute the rank----
	/*double *evec=new double[m*m];
	double *evals=new double[m];
	memcpy(evec, XtX, sizeof(double)*m*m);
	symmetricEigenDecomposition(evec,evals,m);
	int maxActive=0;
	for(int i=0;i<m;++i)if(evals[i]>1e-5){
		++maxActive;
	}
	delete[] evec;
	delete[] evals;*/
	//===========initialize=============
	double *c=new double[m];
	memset(beta, 0, sizeof(double)*m);
	memcpy(c, Xty, sizeof(double)*m);//initial beta is zero
	int j=getMaxIndex(c,m);
	double C=c[j];
	if(C<=lambda){//the optimal beta is zero
		delete[] Xty;
		delete[] XtX;
		delete[] c;
		*error=0;
		for(int i=0;i<n;++i){
			*error+=SQR(y[i]);
		}
		*error=sqrt(*error/n);
		return 0;
	}
	int p=m;
	set<int> I;
	for(int i=0;i<m;++i){
		I.insert(i);
	}
	set<int> A;
	double *XtXbeta=new double[m];
	memset(XtXbeta, 0, sizeof(double)*m);
	//-------first step--------
	double currentLambda=C;
	A.insert(j);
	I.erase(j);
	
	double averageCorrelation=C;
	double *s=new double[m];
	int *II=new int[m];
	double *EE=new double[m*m];
	double *w=new double[m];
	double *XtXw=new double[m];
	
	int maxActive=MIN(n,m);
	while((averageCorrelation>EPS_LARS) && (lambda<C) && A.size()<maxActive){
		int mm=0;
		for(set<int>::iterator it=A.begin();it!=A.end();++it, ++mm){
			s[mm]=1;
			II[mm]=*it;
		}
		for(int i=0;i<mm;++i){
			for(int j=0;j<mm;++j){
				EE[j*mm+i]=XtX[II[i]*m+II[j]];
			}
		}
		solveLeastSquares(EE, mm, mm, s, w);//moving beta in the direction of w, equaly decreases the correlation of the residual with the active set

		for(int i=0;i<m;++i){//update the correlation increment (gradient of OLS) on all components
			XtXw[i]=0;
			for(int j=0;j<mm;++j){
				XtXw[i]+=XtX[i*m+II[j]]*w[j];
			}
		}
		double minGamma=INF_LARS;
		double activeRef=XtXw[*A.begin()];
		
		for(set<int>::iterator it=I.begin();it!=I.end();++it){
			double opc=(C-c[*it])/(activeRef-XtXw[*it]);//inactive variable *it will join the acive set after a step of size opc
			if((opc>EPS_LARS) && (opc<minGamma)){
				minGamma=opc;
			}
		}
		activeRef=c[*A.begin()]/activeRef;//all active correlations will drop to zero after a step of size activeRef
		if((activeRef>EPS_LARS) && (activeRef<minGamma)){
			minGamma=activeRef;
		}
		int pos=0;
		int varOut=-1;
		for(set<int>::iterator it=A.begin();it!=A.end();++it, ++pos){
			double opc=-beta[*it]/w[pos];//active coefficient beta[*it] will drop to zero after a step of size opc (and will become inactive)
			if((opc>EPS_LARS) && (opc<minGamma)){
				minGamma=opc;
				varOut=*it;
			}
		}
		//---update beta---
		pos=0;
		for(set<int>::iterator it=A.begin();it!=A.end();++it, ++pos){
			
			if(beta[*it]+minGamma*w[pos]<-EPS_LARS){
				cerr<<"Warning: negative beta coefficient. "<<beta[*it]+minGamma*w[pos]<<endl;
			}
			beta[*it]+=minGamma*w[pos];
		}
		if(varOut>=0){
			I.insert(varOut);
			A.erase(varOut);

		}
		multMatrixVector<double>(XtX,beta, m, m, XtXbeta);
		linCombVector<double>(Xty,1,XtXbeta,-1,m,c);
		j=-1;
		for(set<int>::iterator it=I.begin();it!=I.end();++it){
			if((j<0) || (c[j]<c[*it])){
				j=*it;
			}
		}
		//----check many-at-a-time condition----
		for(set<int>::iterator it=I.begin();it!=I.end();++it){
			if((j!=*it) && (c[j]==c[*it])){
				cerr<<"Many-at-a-time condition detected"<<endl;
				for(int ii=0;ii<n;++ii){
					cerr<<X[ii*m+j]<<"\t"<<X[ii*m+(*it)]<<endl;
				}
			}
		}
		//--------------------------------------
		averageCorrelation=0;
		if(j>=0){
			C=c[j];
			if(varOut<0){
				A.insert(j);
				I.erase(j);
			}
		}else{
			cerr<<"Warning[lars.cpp]: inactive set empty."<<endl;
			continue;
		}
		for(set<int>::iterator it=A.begin();it!=A.end();++it){
			averageCorrelation+=c[*it];
		}
		averageCorrelation/=A.size();
	}
	if(error!=NULL){
		*error=0;
		double *tmp=new double[n];
		multMatrixVector<double>(X,beta,n,m,tmp);
		for(int i=0;i<n;++i){
			*error+=SQR(tmp[i]-y[i]);
		}
		*error=sqrt(*error/n);
		delete[] tmp;
	}
	delete[] Xty;
	delete[] XtX;
	delete[] XtXbeta;
	delete[] c;
	delete[] s;
	delete[] II;
	delete[] EE;
	delete[] w;
	delete[] XtXw;
	return 0;
}


int nnlars_addaptiveScale(double *inputX, int n, int m, double *inputy, double *beta, double mu, double lambda, double *error){
	double tol=1e-6;
	double diff=tol+1;
	double k=1;
	double *s=new double[n];
	double *phiAlpha=new double[n];
	int iter=0;
	while(tol<diff){
		multVectorScalar(inputy,k,n,s);
		nnlars(inputX, n, m, s, beta, lambda, error, false);
		multMatrixVector(inputX, beta, n,m,phiAlpha);
		double num=dotProduct(phiAlpha, s, n);
		double den=dotProduct(s, s, n);
		double newk=(num+mu)/(den+mu);
		diff=fabs(newk-k);
		k=newk;
		++iter;
	}
	return 0;
}