cmake_minimum_required(VERSION 2.6)
project(NNLibrary)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")
include_directories(${MATLAB_DIR}) 

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)
include_directories(${CLAPACK_INCLUDE_DIR})
link_directories( ${OPENCV_LIBRARY_DIR})
link_directories( ${MATLAB_LIBRARY_DIR})
link_directories( ${CLAPACK_LIBRARY_DIR})

add_library(NNLibrary ${SRC_CPP} ${INCLUDES})
target_link_libraries(NNLibrary niftiio znz zlib ${CLAPACK_LIBS} )

SET(NNLIBRARY_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/NNLibrary")
INSTALL(TARGETS NNLibrary
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${NNLIBRARY_INSTALL_INCLUDE_DIR} COMPONENT Development)	
