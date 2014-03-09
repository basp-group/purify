#include <Python.h>
#include <numpy/arrayobject.h>

#include "PyPurifyConfig.h"

//! Holds state information from the module
struct ModuleState {
  PyArray_Descr* visibility_descriptor;
};

#if PY_MAJOR_VERSION >= 3
# define PURIFY_STATE ((struct ModuleState*)PyModule_GetState(pypurify))
#else
# define PURIFY_STATE (&global_module_state)
  static struct ModuleState global_module_state;
#endif

// Add static functions: these functions are implemenation specific and should not be part of
// anything sent to the outside world
# include "visibility.c"

// Tests function should be included below
// Making all but the init function static means we do have to jump through some hoops to tests
// included and get those tests to see the static functions.
// The alternative is to declare a purify python api and expose via capsules. That's even heavier.
# ifdef PURIFY_DOTESTS
#   include "tests/visibility.c"
# endif


// A macro to easily add functions to method table
#define PURIFY_ADD_FUNCTION(NAME, TYPE, DOC) \
    {#NAME, (PyCFunction)(NAME), METH_ ## TYPE, DOC}

//! Contains all purify methods
static PyMethodDef purify_methods[] = {
#   ifdef PURIFY_DOTESTS
      PURIFY_ADD_FUNCTION(_is_a_visibility, O, "Debug function. Checks visibility from python."),
      PURIFY_ADD_FUNCTION(_flip_element_sign, VARARGS, "Debug function. Changes element sign."),
#   endif
    {NULL, NULL, 0, NULL}
};
#undef PURIFY_ADD_FUNCTION

/********************************/
/*********** PYTHON 3 ***********/
/********************************/
#if PY_MAJOR_VERSION >= 3
  static int traverse(PyObject *m, visitproc visit, void *arg) {
      Py_VISIT((PyObject*)PURIFY_STATE(m)->visibility_descriptor);
      return 0;
  }

  static int clear(PyObject *m) {
      Py_CLEAR((PyObject*)PURIFY_STATE(m)->visibility_descriptor);
      return 0;
  }
  //! Defines the python 3 purify module
  static struct PyModuleDef moduledef = {
          PyModuleDef_HEAD_INIT,
          "_purify",
          NULL,
          sizeof(ModuleState),   // m_size
          purify_methods,
          NULL, // m_reload
          traverse, // m_traverse
          clear, // m_clear
          NULL  // m_free
  };

# define PURIFY_CREATE_MODULE PyModule_Create(&moduledef);
# define PURIFY_DECLARE_MODULE PyMODINIT_FUNC PyInit__purify() { return initialize_purify_module(); }
# define PURIFY_DECREF_MODULE Py_DECREF(pypurify)
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
    int const success_ ## NAME = PyModule_AddObject(pypurify, #NAME, NAME);            \
    if(success_ ## NAME == -1) goto fail;         

// import array returns NULL... can't use it in initialize_purify_module
static void do_import_array() { import_array(); }

//! Creates the actual module and initializes values.
static PyObject * initialize_purify_module() {

  // First, import numpy. Otherwise all hell breaks loose when we call numpy functions.
  do_import_array();
  if(PyErr_Occurred() != NULL) return NULL;

  // Create the module
  PyObject *pypurify = PURIFY_CREATE_MODULE;
  if (pypurify == NULL) return NULL;

  // Create visibility descriptor
  PURIFY_STATE->visibility_descriptor = get_new_visibility_descriptor();
  if(PURIFY_STATE->visibility_descriptor == NULL) goto fail;
  // Add another visibility descriptor object
  PURIFY_ADDVAR(visibility_descriptor, (PyObject*) get_new_visibility_descriptor());

  return pypurify;

  fail:
    PURIFY_DECREF_MODULE;
    return NULL;
}

PURIFY_DECLARE_MODULE;
# undef PURIFY_CREATE_MODULE
# undef PURIFY_DECLARE_MODULE
# undef PURIFY_MODULE_STATE
# undef PURIFY_ADDVAR
