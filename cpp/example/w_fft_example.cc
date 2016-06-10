//example of data with w effect - output of convolution is tested in MATLAB 
#include "MeasurementOperator.h"
#include "utilities.h"
#include "pfitsio.h"
#include "directories.h"
#include <array>
#include <memory>
#include <random>
#include "sopt/relative_variation.h"
#include "sopt/sdmm.h"
#include <sopt/l1_padmm.h>
#include "sopt/utilities.h"
#include "sopt/wavelets.h"
#include "sopt/wavelets/sara.h"
#include "FFTOperator.h"
#include "regressions/cwrappers.h"
#include "types.h"
#include <fstream>
#include <iostream>
#include <boost/math/special_functions/erf.hpp>
#include <unsupported/Eigen/SparseExtra>
#include <boost/lexical_cast.hpp>
#include <math.h>
#include <string>
#include <stdio.h>
#include <omp.h>
#include <new>
       
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_AUTOMATIC_RESIZING
 
int main( int nargs, char const** args ){

  std::cout<< Eigen::nbThreads()<<"\n";
  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  FFTOperator fftop;


std::cout<<"\nStarting\n";
#pragma omp parallel for num_threads(30)
// private(test)
for(t_int i=0; i<1000; ++i){
  t_int tid =omp_get_thread_num();

  std::cout<<" @"<<tid<<"@ ";
  fflush(stdout); 
  Matrix<t_complex> chirpI=Matrix<t_complex>::Random(1024,1024);
  Matrix<t_complex> test(1024,1024);
  test= fftop.forward(chirpI);

  if (utilities::mod(i,500) ==0){
    std::cout<<" --"<<i<<"-- ";
    fflush(stdout); 
  }
}

std::cout<<"\nFFT test Done!\n";
  
return 0;

}



