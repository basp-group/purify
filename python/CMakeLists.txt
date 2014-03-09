if(python)
  include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
  add_definitions("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
  set(PYPURIFY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/purify)

  # Configuration file with some defines
  set(PURIFY_DOTESTS ${tests})
  configure_file(
    PyPurifyConfig.in.h
    ${CMAKE_CURRENT_BINARY_DIR}/PyPurifyConfig.h
    @ONLY
  )
  

  add_library (pypurify SHARED module.c)
  set_target_properties(pypurify PROPERTIES OUTPUT_NAME _purify)
  set_target_properties(pypurify PROPERTIES PREFIX "")
  set_target_properties(pypurify PROPERTIES SUFFIX ".so")
  target_link_libraries(pypurify ${PYTHON_LIBRARIES})
  # Change path to something such that we can import it in tests.
  set_target_properties(pypurify PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYPURIFY_DIRECTORY})
  add_dependencies(pypurify PyPurifyConfig.in.h)

  # Copy python files to output package
  file(GLOB input_sources *.py)
  foreach(input ${input_sources})
    add_custom_command(
      TARGET pypurify
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${input} ${PYPURIFY_DIRECTORY}
    )
  endforeach()


  # Installation
  install(TARGETS pypurify DESTINATION ${PYTHON_PKG_DIR}/purify)
  install(FILES ${LIBCRYSTAL_PYSOURCES} DESTINATION ${PYTHON_PKG_DIR}/purify)
endif()

add_subdirectory(tests)
