#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <vector>
class Histogram{
	protected:
		struct Bucket{
			int frequency;
			double value;
		};
		std::vector<Bucket> list;
		bool good_flag;
    public:
      Histogram(double *v, int n, int nbuckets);
	  ~Histogram();
      int getFrequency(int index)const;
      double getValue(int index)const;
      unsigned size()const;
      double getFirstMin(void)const;
	  double getLastPeak(int windowSize)const;
	  bool good(void);
	  void printFrequencies(const char *fname);
};


#endif
