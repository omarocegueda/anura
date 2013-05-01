#include "rgbtriplet.h"
RGBTriplet::RGBTriplet(){
	r=0;
	g=0;
	b=0;
}

RGBTriplet::RGBTriplet(unsigned char val){
	r=val;
	g=val;
	b=val;
}

RGBTriplet::RGBTriplet(unsigned char _r, unsigned char _g, unsigned char _b){
	r=_r;
	g=_g;
	b=_b;
}

const bool RGBTriplet::operator<(const RGBTriplet &T)const{
	if(r<T.r){
		return true;
	}
	if(r==T.r){
		if(g<T.g){
			return true;
		}
		if(g==T.g){
			return b<T.b;
		}
		return false;
	}
	return false;
}

