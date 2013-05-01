cmake_minimum_required(VERSION 2.6)
project(HARDILib)
find_package(OpenCV REQUIRED)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)
include_directories(${CLAPACK_INCLUDE_DIR})
link_directories(${CLAPACK_LIBRARY_DIR})
link_directories(${OPENCV_LIBRARY_DIR})
add_library(HARDILib ${SRC_CPP} ${INCLUDES} )
target_link_libraries(HARDILib niftiio znz zlib NNLibrary optimizationlib sparselib ${OpenCV_LIBS} ${CLAPACK_LIBS} ${OPENGL_glu_LIBRARY} ${OPENGL_gl_LIBRARY})


SET(HARDILIB_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/HARDILib")
INSTALL(TARGETS HARDILib
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${HARDILIB_INSTALL_INCLUDE_DIR} COMPONENT Development)	
