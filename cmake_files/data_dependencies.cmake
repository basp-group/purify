# Download ngc3256 data from the web
set(NGC5921_2_TGZ "${PROJECT_SOURCE_DIR}/data/NGC5921_2.tar.gz")
set(NGC5921_2_MS "${PROJECT_BINARY_DIR}/data/NGC5921_2.ms")

# untar the data
if(EXISTS "${NGC5921_2_TGZ}" AND NOT EXISTS "${NGC5921_2_MS}")
  find_program(TAR_PROGRAM tar)
  if(NOT TAR_PROGRAM)
    message(FATAL_ERROR "Cannot untar data without tar")
  endif()
  message(STATUS "Untar of test data")
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/data)
  execute_process(
    COMMAND ${TAR_PROGRAM} -zxf ${NGC5921_2_TGZ}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/data)
endif()
