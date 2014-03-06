#include <Python.h>
#include <numpy/arrayobject.h>

// Add static functions: these functions are implemenation specific and should not be part of
// anything sent to the outside world
# include "visibility.c"

//! Contains all purify methods
static PyMethodDef purify_methods[] = {
    {NULL, NULL, 0, NULL}
};

/********************************/
/*********** PYTHON 3 ***********/
/********************************/
#if PY_MAJOR_VERSION >= 3
  //! Defines the python 3 purify module
  static struct PyModuleDef moduledef = {
          PyModuleDef_HEAD_INIT,
          "_purify",
          NULL,
          -1,   // m_size
          purify_methods,
          NULL, // m_reload
          NULL, // m_traverse
          NULL, // m_clear
          NULL  // m_free
  };

# define PURIFY_CREATE_MODULE PyModule_Create(&moduledef);
# define PURIFY_DECLARE_MODULE PyMODINIT_FUNC PyInit__purify() { return initialize_purify_module(); }
# define PURIFY_DECREF_MODULE Py_DECREF(module)
#else
/********************************/
/*********** PYTHON 2 ***********/
/********************************/
# define PURIFY_CREATE_MODULE Py_InitModule("_purify", purify_methods);
# define PURIFY_DECLARE_MODULE PyMODINIT_FUNC init_purify(void) { initialize_purify_module(); }
# define PURIFY_DECREF_MODULE 
#endif

/**********************************/
/*********** PYTHON ANY ***********/
/**********************************/

// Makes adding module variables easier
# define PURIFY_ADDVAR(NAME, INITIALIZATION)                                           \
    PyObject * const NAME = INITIALIZATION;                                            \
    if(NAME == NULL) goto fail;                                                        \
    int const success_ ## NAME = PyModule_AddObject(module, #NAME, NAME);              \
    if(success_ ## NAME == -1) goto fail;         

// import array returns NULL... can't use it in initialize_purify_module
static void do_import_array() { import_array(); }

//! Creates the actual module and initializes values.
static PyObject * initialize_purify_module() {

  // First, import numpy. Otherwise all hell breaks loose when we call numpy functions.
  do_import_array();
  if(PyErr_Occurred() != NULL) return NULL;

  // Create the module
  PyObject *module = PURIFY_CREATE_MODULE;
  if (module == NULL) return NULL;

  // Add the visibility descriptor object
  PURIFY_ADDVAR(visibility_descriptor, get_visibility_descriptor());

  return module;

  fail:
    PURIFY_DECREF_MODULE;
    return NULL;
}

PURIFY_DECLARE_MODULE;
# undef PURIFY_CREATE_MODULE
# undef PURIFY_DECLARE_MODULE
# undef PURIFY_MODULE_STATE
# undef PURIFY_ADDVAR
