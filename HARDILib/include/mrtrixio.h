#ifndef MRTRIXIO_H
#define MRTRIXIO_H
#include <map>
#include <string>
int read_mrtrix_tracks(const char *fname, double **&tracks, int *&trackSizes, int &numTracks, std::map<std::string, std::string> &properties);
int read_mrtrix_image(const char *fname, double *&img, int &ndims, int *&dims, int *&vox, int *&layout, int *&layout_orientations, double *&transform);
int write_mrtrix_image(const char *fname, double *img, int ndim, int *dims, int *voxSizes, int *layout, int *layout_orientations, double *transform=NULL);
//---wrappers---
int write_mrtrix_image(const char *fname, double *img, int nslices, int nrows, int ncols);
int write_mrtrix_image(const char *fname, double *img, int nslices, int nrows, int ncols, int signalLength);



#endif
