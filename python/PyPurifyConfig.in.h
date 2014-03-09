#ifndef PURIFY_PYTHON_CONFIG
#include <complex.h>

#cmakedefine PURIFY_DOTESTS

// A short type hierarchy, so we can change types easily, at least in python code.
//! Type for real numbers
typedef double t_purify_real;
//! Type for complex numbers
typedef complex double t_purify_complex;
//! Type for integer numbers
typedef int t_purify_int;

static const char visibility_kind = 'V';
#endif
