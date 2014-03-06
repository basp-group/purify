# include <stdio.h>
# include <complex.h>

//! Creates the numpy descriptor for visibilities
static PyObject* get_visibility_descriptor() {

  // Build the names of the types
  char float_type[4];
  char complex_type[4];
  sprintf(float_type, "f%u", (unsigned)sizeof(double));
  sprintf(complex_type, "c%u", (unsigned)sizeof(complex double));

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

  PyArray_Descr *descriptor;
  int const success = PyArray_DescrConverter(dtype_string, &descriptor);
  Py_DECREF(dtype_string);
  if(success == 0) return NULL;

  return (PyObject*)descriptor;
}
