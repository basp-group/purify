# include <stdio.h>
# include <complex.h>

//! \brief Returns a new valid visibility descriptor
//! \returns the descriptor type number
static PyArray_Descr* get_new_visibility_descriptor() {

  // Build the names of the types
  char float_type[4];
  char complex_type[4];
  sprintf(float_type, "f%u", (unsigned)sizeof(t_purify_real));
  sprintf(complex_type, "c%u", (unsigned)sizeof(t_purify_complex));

  // Build the dictionary describing the array
  PyObject* dtype_string = Py_BuildValue(
      "[(s, s), (s, s), (s, s), (s, s), (s, s)]",
      "u", float_type,
      "v", float_type,
      "w", float_type,
      "noise", complex_type, 
      "measurement", complex_type
  );
  if(dtype_string == NULL) return NULL;

  PyArray_Descr *descriptor = NULL;
  int const success = PyArray_DescrConverter(dtype_string, &descriptor);
  Py_DECREF(dtype_string);
  if(success == 0) return NULL;

  return descriptor;
}

//! \brief Returns borrowed reference to visibility descriptor
//! \details Should not be called before visibility descriptor has been set in the module state.
static PyArray_Descr* get_visibility_descriptor() {
# if PY_MAJOR_VERSION >= 3
    PyObject* pypurify = PyImport_ImportModule("purify");
    if(pypurify == NULL) return NULL;
# endif

  PyArray_Descr* const dtype = PURIFY_STATE->visibility_descriptor;

# if PY_MAJOR_VERSION >= 3
    Py_DECREF(pypurify);
# endif

  return dtype;
}

// \brief Checks that an array is compatible with visibility
// \returns 
//    - 0 if not compatible
//    - > 0 if compatible
//    - < 0 if error
//
//    Does not raise a python exception if array is incompatible.
static int PyVisibility_Check(PyObject *_array) {
  if(PyArray_Check(_array) == 0) return 0;

  PyArray_Descr* visibility = get_visibility_descriptor();
  if(visibility == NULL) return -1;

  // descriptor is a *borrowed* reference to input array's descriptor
  PyArray_Descr* descriptor = PyArray_DESCR((PyArrayObject*)_array);
  int const result = PyArray_CanCastTypeTo(visibility, descriptor, NPY_NO_CASTING);
  return result;
}

// \brief Asserts that an array is compatible with visibility
// \returns 
//    - 0 if not compatible or error
//    - > 0 if compatible
//
//    Raises a python exception if not compatible
static int PyVisibility_Assert(PyObject *_array) {
  int const result = PyVisibility_Check(_array);
  if(result < 0) return 0;
  if(result > 0) return 1;
  PyErr_SetString(PyExc_TypeError, "Input is not a visibility");
  return 0;
}

//! \brief wraps a python visibility into a C visibility
//! \param[in] _python The python object to wrap
//! \param[inout] _c The thin C objects that accesses the same memory a the python object. It does
//! not own the memory. It should not exist beyond the existence of the input array.
//! \returns 0 on failure, 1 on success. Sets an error on failure.
static int wraps_purify_visibility(purify_visibility &_c, PyArrayObject* _python) {
  PyArrayObject * const python = (PyArrayObject*) _python;
  if(PyVisibility_Assert(python) == 0) return 0;
  if(PyArray_NDIM(python) != 1) {
    PyErr_SetString(PyExc_ValueError, "visibility should be a 1d array.");
    return 0;
  }
  _c.nmeas = PyArray_DIM(python, 0);
  _c.u = 
}
