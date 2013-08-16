#Add cxxxxxt again
cmake_minimum_required(VERSION 2.8)
PROJECT(anura)
SET(ANURA_INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")
SET(ANURA_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
SET(ANURA_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")

option ( USE_QT "Use Qt?" OFF)
option ( USE_VTK "Use VTK?" OFF)
option ( USE_ARPACK "Use Arpack?" OFF)
option ( USE_GSL "Use GSL?" OFF)
option ( USE_MUMPS "Use MUMPS?" OFF)
option ( USE_MATLAB "Use Matlab?" OFF)

IF (WIN32)
	add_definitions( -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS /wd4996)
	add_definitions( /wd4244 /wd4018)
	set(CLAPACK_INCLUDE_DIR "C:/Program Files/clapack/include" CACHE STRING "clapack base directory")
	set(CLAPACK_LIBRARY_DIR "C:/Program Files/clapack/lib" CACHE STRING "clapack base directory")
	set(CLAPACK_LIBS
		blas.lib
		lapack.lib
		libf2c.lib
	)
ELSE (WIN32)
	set(CLAPACK_INCLUDE_DIR "/usr/local/include/clapack" CACHE STRING "clapack base directory")
	set(CLAPACK_LIBRARY_DIR "/usr/local/lib" CACHE STRING "clapack base directory")
	set(CLAPACK_LIBS lapack blas f2c)
ENDIF (WIN32)

add_definitions(-DHAVE_ZLIB)
#######################################################################
######################CONFIGURE QT LIBRARY#############################
#######################################################################
IF(USE_QT)
	add_definitions( -DUSE_QT)
	find_package(Qt4 REQUIRED)
	IF(NOT QT4_FOUND)
		message(SEND_ERROR "Failed to find Qt4")
	ELSE(NOT QT4_FOUND)
		set(QT_USE_QTMAIN 1)
		set(QT_USE_QTOPENGL 1)
		include(${QT_USE_FILE})
	ENDIF(NOT QT4_FOUND)
ENDIF(USE_QT)	
#######################################################################
######################CONFIGURE GSL LIBRARY############################
#######################################################################
IF(USE_GSL)
	add_definitions( -DUSE_GSL)
	IF(WIN32)
		SET(GSL_LIBRARY_DIR "${PROJECT_SOURCE_DIR}/third_party/gsl/windows" CACHE STRING "gsl library dir")
		SET(GSL_LIBRARIES "libgsl.lib libgslcblas.lib" CACHE STRING "gsl libraries")
	ELSE(WIN32)
		SET(GSL_LIBRARY_DIR "${PROJECT_SOURCE_DIR}/third_party/gsl/mac" CACHE STRING "gsl library dir")
		SET(GSL_LIBRARIES "gsl gslcblas" CACHE STRING "gsl librares")
	ENDIF(WIN32)
	LINK_DIRECTORIES(${GSL_LIBRARY_DIR})
ENDIF(USE_GSL)
#######################################################################
######################CONFIGURE ARPACK LIBRARY#########################
#######################################################################
IF(USE_ARPACK)
	add_definitions( -DUSE_ARPACK)
	IF(WIN32)
		SET(ARPACK_LIBRARY_DIR "${PROJECT_SOURCE_DIR}/third_party/arpack/windows" CACHE STRING "arpack library dir")
		SET(ARPACK_LIBRARY "arpack_win32.lib" CACHE STRING "arpack library")
	ELSE(WIN32)
		SET(ARPACK_LIBRARY_DIR "${PROJECT_SOURCE_DIR}/third_party/arpack/mac" CACHE STRING "arpack library dir")
		SET(ARPACK_LIBRARY "gfortran arpack_lion" CACHE STRING "arpack librares")
		SET(GFORTRAN_LIBRARY_DIR "/usr/local/gfortran/lib" CACHE STRING "gfortran library")
		LINK_DIRECTORIES(${GFORTRAN_LIBRARY_DIR})
	ENDIF(WIN32)
	LINK_DIRECTORIES(${ARPACK_LIBRARY_DIR})
ENDIF(USE_ARPACK)
#######################################################################
######################CONFIGURE NIFTI I/O LIBRARY######################
#######################################################################
SET(CPACK_PACKAGE_VERSION_MAJOR "2")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(NIFTI_SHAREDLIB_VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH} )
STRING( REGEX MATCH "^[0-9]+" NIFTI_SHAREDLIB_SOVERSION ${NIFTI_SHAREDLIB_VERSION})
SET(NIFTI_LIBRARY_PROPERTIES VERSION ${NIFTI_SHAREDLIB_VERSION} SOVERSION ${NIFTI_SHAREDLIB_SOVERSION})
SET(NIFTI_INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")
SET(NIFTI_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
SET(NIFTI_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/nifti")

#######################################################################
######################CONFIGURE ZLIB I/O LIBRARY#######################
#######################################################################
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/zlib)
SET(ZLIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/zlib CACHE STRING "zlib include directory")
SET(ZLIB_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/zlib CACHE STRING "zlib library directory")
IF(WIN32)
	SET(ZLIB_LIBRARY "zlib" CACHE STRING "zlib library")
ELSE(WIN32)
	SET(ZLIB_LIBRARY "libz.a" CACHE STRING "zlib library")
ENDIF(WIN32)
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})
LINK_DIRECTORIES(${ZLIB_LIBRARY_DIR})
SET(PACKAGE_PREFIX "")
INCLUDE(InstallRequiredSystemLibraries)

subdirs(	zlib 
			znzlib 
			niftilib 
			procimlib
			timelib
			sparselib
			NNLibrary 
			HARDILib
			optimizationlib
			imageiolib
			dimreductionlib
			MultiTensorFieldExplorer
		)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/niftilib
	${PROJECT_SOURCE_DIR}/znzlib
	${PROJECT_SOURCE_DIR}/procimlib/include
	${PROJECT_SOURCE_DIR}/NNLibrary/include
	${PROJECT_SOURCE_DIR}/FastEMD
	${PROJECT_SOURCE_DIR}/HARDILib/include
	${PROJECT_SOURCE_DIR}/third_party/gsl/include
	${PROJECT_SOURCE_DIR}/optimizationlib/include
	${PROJECT_SOURCE_DIR}/imageiolib/include
	${PROJECT_SOURCE_DIR}/dimreductionlib/include
	${PROJECT_SOURCE_DIR}/timelib/include
	${PROJECT_SOURCE_DIR}/sparselib/include
	${PROJECT_SOURCE_DIR}/MultiTensorFieldExplorer/include
)
