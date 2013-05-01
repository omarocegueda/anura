#ifndef RGBTRIPLET_H
#define RGBTRIPLET_H
struct RGBTriplet{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	RGBTriplet();
	RGBTriplet(unsigned char val);
	RGBTriplet(unsigned char _r, unsigned char _g, unsigned char _b);
	const bool operator<(const RGBTriplet &T)const;
};
#endif
