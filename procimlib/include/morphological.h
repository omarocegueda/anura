#ifndef MORPHOLOGICAL_H
#define MORPHOLOGICAL_H
#include "macros.h"
template<class T> void erode(int niter, T *vol, int nslices, int nrows, int ncols){
	int n=nslices*nrows*ncols;
	for(int i=0;i<n;++i){
		if(vol[i]!=(T)(0)){
			vol[i]=(T)(1);
		}
	}
	for(int t=1;t<=niter;++t){
		int p=0;
		for(int s=0;s<nslices;++s){
			for(int r=0;r<nrows;++r){
				for(int c=0;c<ncols;++c, ++p){
					if(vol[p]<t){
						continue;
					}
					int cnt=0;
					int occ=0;
					for(int ds=-1;ds<=1;++ds){
						for(int dr=-1;dr<=1;++dr){
							for(int dc=-1;dc<=1;++dc){
								int ss=s+ds;
								int rr=r+dr;
								int cc=c+dc;
								int q=ss*nrows*ncols+rr*ncols+cc;
								if(IN_RANGE(ss, 0, nslices) && IN_RANGE(rr, 0, nrows) && IN_RANGE(cc, 0, ncols)){
									++cnt;
									if(vol[q]>=t){
										++occ;
									}
								}
							}
						}
					}
					if(occ>=cnt){
						vol[p]=t+1;
					}

				}
			}
		}
	}
	for(int i=0;i<n;++i){
		vol[i]=(vol[i]>niter);
	}
}




#endif