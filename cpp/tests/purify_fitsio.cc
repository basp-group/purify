#include "catch.hpp"
#include "purify/directories.h"
#include "purify/pfitsio.h"
using namespace purify;
using namespace purify::notinstalled;

TEST_CASE("readwrite2d", "purify fitsio") {

  Image<t_complex> input = pfitsio::read2d(image_filename("M31.fits"));
  pfitsio::write2d(input.real(), output_filename("fits_output.fits"));
  Image<t_complex> input2 = pfitsio::read2d(output_filename("fits_output.fits"));
  CHECK(input.isApprox(input2, 1e-6));
}

TEST_CASE("readwrite3d", "purify fitsio") {

  std::vector<Image<t_complex>> input = pfitsio::read3d(image_filename("cube_example.fits"));
  CHECK(input.size() == 4);
  CHECK(input[0].size() == 200 * 200);
  std::vector<Image<t_real>> input_real;
  for (int i = 0; i < input.size(); i++) {
    input_real.push_back(input[i].real());
  }
  pfitsio::write3d(input_real, output_filename("cube_output.fits"));
  std::vector<Image<t_complex>> input2 = pfitsio::read3d(output_filename("cube_output.fits"));
  for (int i = 0; i < input.size(); i++) {
    CHECK(input[i].isApprox(input2[i], 1e-6));
  }
}
