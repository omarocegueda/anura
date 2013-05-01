#ifndef MULTITENSORFIELD_H
#define MULTITENSORFIELD_H
#include "MultiTensor.h"
#include <string>
#include <vector>
#include "SparseMatrix.h"
enum MultiTensorFieldVisualizationType{MTFVT_Orientation, MTFVT_Alpha, MTFVT_Error, MTFVT_Sparcity, MTFVT_Arrows, MTFVT_ApparentDiffusion, MTFVT_OrientationDiffusion, MTFVT_ClusterColors, MTFVT_SampledFunction};
class MultiTensorField{
	private:
		double mat[16];//transformation matrix to world coordinates
		double **DBFDirections;//for visualization only
	protected:
		int nrows;
		int ncols;
		int nslices;
		MultiTensor *voxels;
		double *dualLatticeRow;
		double *dualLatticeCol;
		double *dualLatticeSlice;
		MultiTensorFieldVisualizationType visType;
		bool showGroupColors;
		//----for visualization purposes----
		double *error;

		double *samplingPoints;
		int numSamplingPoints;
	public:
		MultiTensorField();
		MultiTensorField(int ns, int nr, int nc);
		~MultiTensorField();
		void setDBFDirections(double **_DBFDirections);
		void loadFromTxt(const std::string &fname);
		void saveToTxt(const std::string &fname);

		void loadFromBinary(const std::string &fname);
		void saveToBinary(const std::string &fname);

		void saveSliceToTxt(int slice, const std::string &fname);
		void loadFromNifti(const std::string &nTensFname, const std::string &sizeCompartmentFname, const std::string &fPDDFname);
		void loadFromNifti(const std::string &fPDDFname);
		void buildFromCompartments(int nslices, int nrows, int ncols, int maxCompartments, int *numCompartments, double *compartmentSizes, double *pddField);
		void saveAsNifti(const std::string &nTensFname, const std::string &sizeCompartmentFname, const std::string &fPDDFname);
		void saveSinglePeaksFile(const std::string &peaksFileName, int maxCompartments);
		void copyFrom(MultiTensorField &M);
		void copyFrom(MultiTensorField &M, std::set<int> &slices, int sliceType=0);
		void dellocate(void);
		void allocate(int ns, int nr, int nc);
		void generateDWMRIVolume(double *gradients, int numGradients, double b, int SNR, double *&s0Volume, double *&dwVolume);
		void saveODFFieldFromAlpha(double *ODFDirections, int nODFDirections, double *DBFDirections, int nDBFDirections, double *lambda, FILE *F);
		int computeLocalCoherenceIndex(double voidCost, double *LCI);
		//---accessors---
		int getMaxCompartments(void);
		int getMaxCompartments(std::set<int> &slices);
		int getTotalCompartments(void);
		int getTotalCompartments(std::set<int> &slices);
		int getNumRows(void)const;
		int getNumCols(void)const;
		int getNumSlices(void)const;
		MultiTensor *getVoxels(void);
		MultiTensor *getVoxelAt(int slice, int row, int col);
		double *getError(void);
		const MultiTensor *getVoxels(void)const;
		void setDualLatticeCol(int s, int r, int c, double val);
		void setDualLatticeRow(int s, int r, int c, double val);
		void setDualLatticeSlice(int s, int r, int c, double val);
		void setDualLattice(int voxIndexA, int voxIndexB, double val);
		double *getDualLatticeRow(void);
		double *getDualLatticeCol(void);
		double *getDualLatticeSlice(void);
		void setShowGroupColors(bool b);
		bool getShowGroupColors(void);

		//---visualization---
		void setVisualizationType(MultiTensorFieldVisualizationType _visType);
		void drawMultiTensor(MultiTensor &mt, int component, double px, double py, double pz, double intensity);
		void drawTensorField(void);
		void drawPDDField(void);
		void drawPDDSlice(int slice, int k_pdd);
		void drawLatticeSeparation(int voxIndexA, int voxIndexB, double val);
		void drawLatticeSlice(int slice);
		void normalizeLatticeSeparations(void);
		void drawArrow(double *position, double *dir);

		//---clustering analysis---
		int extractCodifiedTensorList(int neighSize, double *dir, int nDir, double *&positions, double *&orientations, double *&alpha, int &numTensors, int *&tensorIndex, int &maxTensorsPerVoxel, int *&angleIndex, int slice=-1);
		int buildNCutSparseMatrix(double d0, double theta0, double dTheta0, double offset, SparseMatrix &S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex);
		int buildNCutSparseMatrixBestAssignment(double d0, double theta0, double dTheta0, double offset, SparseMatrix &S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex);
		int buildFullSimilarityMatrix(double d0, double theta0, double dTheta0, double *&S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex);
		int ncutDiscretization(double *evec, int k, int n, int *discrete);
		int splitSlice(int slice, MultiTensorField &mtf);
		int splitSliceToAngleVolume(int slice, MultiTensorField &mtf);

		void loadSamplingPoints(const char *fname);
		double *getSamplingPoints(void);
		int getNumSamplingPoints(void);
};
void evaluatePositiveCompartmentCount(const MultiTensorField &GT, const MultiTensorField &E, std::vector<double> &vec);
void evaluateNegativeCompartmentCount(const MultiTensorField &GT, const MultiTensorField &E, std::vector<double> &vec);
void evaluateMissingWMVoxels(const MultiTensorField &GT, const MultiTensorField &E, std::vector<double> &vec);
void evaluateExtraWMVoxels(const MultiTensorField &GT, const MultiTensorField &E, std::vector<double> &vec);
void evaluateAngularPrecision(const MultiTensorField &GT, const MultiTensorField &E, std::vector<double> &vec);
void evaluateODFAccuracy(const MultiTensorField &GT, const MultiTensorField &E, double *directions, int numDirections, std::vector<double> &vec);
MultiTensorField *createRealisticSyntheticField(int nrows, int ncols, int nslices, double minAngle, double *diffusivities,double *randomPDDs, int nRandom);
#endif
