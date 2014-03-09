//! \brief Checks that object is indeed a visibility
static PyObject * _is_a_visibility(PyObject *_module, PyObject* _array) {
  int const is_visibility = PyVisibility_Check(_array);
  if(is_visibility < 0) return NULL;
  else if(is_visibility == 0) Py_RETURN_FALSE;
  Py_RETURN_TRUE;
}

//! \brief Change the sign of given element.
//! \details Assumes all elements are double. 
static PyObject * _flip_element_sign(PyObject *_module, PyObject *_args) {
  PyArrayObject *array;
  int i, j, k;
  if(PyArg_ParseTuple(_args, "Oii", &array, &i, &j) == 0) return NULL;

  int const is_visibility = PyVisibility_Check((PyObject*)array);
  if(is_visibility < 0) return NULL;
  else if(is_visibility == 0) { 
    PyErr_SetString(PyExc_TypeError, "Input argument is not a visibility");
    return NULL;
  }
  if(PyArray_NDIM(array) != 1) { 
    PyErr_SetString(PyExc_TypeError, "Input array should be 1d");
    return NULL;
  }
  
  int const nitems = 7;
  int const length = PyArray_DIM(array, 0);
  if(i < 0) i += length;
  if(j < 0) j += nitems;
  if(i < 0 || i >= length || j < 0 || j > nitems ) {
    PyErr_SetString(PyExc_IndexError, "input indices out of range");
    return NULL;
  }

  
  t_purify_real *data_ptr = (t_purify_real*)PyArray_GETPTR1(array, i);
  *(data_ptr + j) *= -1.0;
  
  Py_RETURN_NONE;
}
