CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
PROJECT(dualISO)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{SRC_ROOT}/cmake)


FIND_PACKAGE(VclLib REQUIRED)

  #
  # Custom cmake modules, like FindXPackage.cmake
  #
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

FIND_PACKAGE (OpenEXR REQUIRED)
INCLUDE_DIRECTORIES(${OpenEXR_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenEXR_LIBRARY_DIRS})
SET( OPENEXR_LIBRARIES Half IlmImf Iex Imath IlmThread)

INCLUDE_DIRECTORIES(/usr/local/include/eigen3)



VCL_ADD_LIBRARY(image/VclExr VclExr False)
VCL_ADD_LIBRARY(opensource/rawspeed rawspeed True)
VCL_ADD_LIBRARY(image/VclRaw VclRaw True)
VCL_ADD_LIBRARY(util/VclMatlab VclMatlab False)
VCL_ADD_LIBRARY(opensource/arg arg False)


set(SOURCEFILES calibrateCamera.cpp)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
ADD_SVNVER(SOURCEFILES)
ADD_EXECUTABLE(dualISO ${SOURCEFILES})
target_link_libraries(dualISO ${OPENEXR_LIBRARIES} arg VclExr VclRaw exiv2 jpeg rawspeed VclMatlab hdf5 matio opencv_core opencv_calib3d opencv_imgproc opencv_highgui)

#VCL_BUILD_32_BIT_CHECK(dualISO)

