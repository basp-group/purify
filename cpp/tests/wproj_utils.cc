#include "purify/config.h"
#include <iostream>
#include "catch.hpp"
#include "purify/directories.h"
#include "purify/logging.h"
#include "purify/types.h"
#include "purify/wproj_utilities.h"
using namespace purify;
using namespace purify::notinstalled;

TEST_CASE("wprojection_matrix") {
  //! test if convolution is identity 
  const Vector<t_real> w_components = Vector<t_real>::Random(12) * 0;
  const t_int Nx = 128;
  const t_int Ny = 128;
  const t_int rows = w_components.size();
  const t_int cols = Nx * Ny;

  Sparse<t_complex> I(rows, cols);
  I.reserve(Vector<t_int>::Constant(rows, 1));
  for (t_int i = 0; i < std::min(rows, cols); i++) {
    I.coeffRef(i, i) = 1;
  }

  auto G = wproj_utilities::wprojection_matrix(I, Nx, Ny,
      w_components, 10, 10, 0.9, 0.9);

  CHECK(G.nonZeros() == I.nonZeros());
  for (t_int k = 0; k < G.outerSize(); ++k)
      for (Sparse<t_complex>::InnerIterator it(G, k); it; ++it)
         {
             CHECK(it.value() == 1.);
             CHECK(it.row() == it.col());   // row index
           }
}
