cmake_minimum_required(VERSION 2.6)
project(imageiolib)
find_package(OpenCV REQUIRED)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")

include_directories(.)
include_directories(${CLAPACK_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${CLAPACK_LIBRARY_DIR})
link_directories(${OPENCV_LIBRARY_DIR})
add_library(imageiolib ${SRC_CPP} ${INCLUDES})
IF(USE_VTK)
	target_link_libraries(imageiolib QVTK ${OpenCV_LIBS})
ELSE(USE_VTK)
	target_link_libraries(imageiolib ${OpenCV_LIBS})
ENDIF(USE_VTK)

SET(IMAGEIOLIB_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/imageiolib")
INSTALL(TARGETS imageiolib
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${IMAGEIOLIB_INSTALL_INCLUDE_DIR} COMPONENT Development)	
