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



int main( int nargs, char const** args ){
  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  FFTOperator fftop;

 /* Inputs of the program */
  std::cout<<"\nReading inputs ..\n";
  std::string inVisName=args[1];
  std::string ec = args[2];
  const t_real energyC=std::stod(ec);
  std::string imsz=args[3];
  t_int imsize=std::stod(imsz);
  std::string outputdir=args[4];


  


  std::cout<< "**--------------------------------------------------------------**\n";
  std::cout<< "      This is an example RI imaging to generate the G matrix\n";
  std::cout<< "      UVW coverage: "<<inVisName<<"\n";
  std::cout<< "      Sparsity on C " << energyC <<"\n";
  std::cout<< "      Result saved in : "<<outputdir<<"\n";
  std::cout<< "      Reconstruction using PADMM\n";
  std::cout<< "**---------------------------------------------------------------**\n";

  //input files
  std::string const vis_file = inVisName;//image_filename("coverages/" + inVisName);
  
  //consts
  std::cout<<"\nIt s a minor issue - but just make sure to update the freq of the uv coverage adopted \n";
  const t_real C = 299792458;
  const t_real freq0 = 3499.51e6;
  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_real  arcsecTrad = purify_pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
  bool use_w_term = true; // correct for w

  // uv data struct
  utilities::vis_params uv_data;

  /* 
     Gridding parameters
  */
  std::string gridKernel = "kb"; // choice of the gridding kernel
  t_real overSample = 2; // oversampling ratio in Fourier due to gridding
  t_int gridSizeU = 4;
  t_int gridSizeV = 4;

  std::string weighting_type = "none";
  t_real RobustW = 0; // robust weighting parameter
  // Read sky model image & dimensions
 
  t_int heightOr = imsize;
  t_int widthOr = imsize;
  std::cout<< "\n"<< "sky image  of size " << heightOr<< " x "<<widthOr<< " pixels. \n";

  // Read visibilities | uvw coverages from files 
  bool read_w = true; // read w from uv coverage
  uv_data = utilities::read_uvw_real( vis_file, lambda, read_w); // uvw are  from now  on in units of lambda
  const Vector<t_real> & u = uv_data.u.cwiseAbs();
  const Vector<t_real> & v = uv_data.v.cwiseAbs();
  const Vector<t_real> & w = uv_data.w.cwiseAbs();
  const Vector<t_complex> & visi = uv_data.vis;
  std::cout<< "\n"<<  "Reading data done " <<" \n";


  //setting resolution - cellsize
  t_real maxBaseline = lambda *(((u.array() * u.array() + v.array() * v.array() + w.array() * w.array()).sqrt()).maxCoeff()) ;
  t_real maxProjectedBaseline = lambda *(((u.array() * u.array() + v.array() * v.array()).sqrt()).maxCoeff()) ;
  t_real thetaResolution = 1.22* lambda / maxProjectedBaseline ;
  t_real cellsize = (thetaResolution / arcsecTrad ) / 2;  // Nyquist sampling
  std::cout<< "\n"<<  "Setting  params done " <<" \n";
  // setting FoV on L & M axis 
  const t_real theta_FoV_L = cellsize * widthOr * arcsecTrad; 
  const t_real theta_FoV_M = cellsize * heightOr * arcsecTrad;

  std::cout << "\nFreq0 "<< freq0 <<", max baseline " << maxBaseline << " m,  angular resolution " << thetaResolution << " rad, cellsize "
   << cellsize <<  " arcsec, "<< "FoV " <<theta_FoV_L  <<" rad. \n";

  /* 
     Set w terms
  */
  const t_real L = 2 * std::sin(theta_FoV_L  * 0.5);
  const t_real M = 2 * std::sin(theta_FoV_M  * 0.5);
  const t_real wlimit = widthOr / (L * M); 
  t_real w_max = (uv_data.w.cwiseAbs()).maxCoeff();
  std::cout <<"\nINFO: original w_max " << w_max <<", limits on w: wrt FoV & Npix " << wlimit << ", wrt b_max " << maxBaseline /lambda << ".\n" ;

   
  std::cout <<"wmax/wlimit = " << w_max/wlimit <<".\n" ;
 
  // Upsampling in Fourier due to w component - image size setting - keep even size for now

  t_int multipleOf = 1;// size should be multiple of 2^level (level of DB wavelets)
  t_real upsampleRatio = utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr,multipleOf);
  Matrix<t_complex> skyOr= Matrix<t_complex>::Zero(widthOr, heightOr);  
  auto skypaddedFFT = utilities::re_sample_ft_grid_sim(fftop.forward(skyOr), upsampleRatio, multipleOf);
  auto skyUP = fftop.inverse(skypaddedFFT);
  t_int heightUp = skyUP.rows();//size of upsampled image
  t_int widthUp = skyUP.cols();

  if (upsampleRatio!=1){
    cellsize = cellsize/upsampleRatio; //update cellsize if upsampling
    std::cout <<  "Upsampling: cellsize " << cellsize <<  "arcsec, upsampling ratio: "<< upsampleRatio << " New Image size " << heightUp<< " x "<<widthUp<< " pixels. \n";
  }
  else{ std::cout<<"INFO: no upsampling required.\n";}
 

  /*  
     Measurement operator 
  */
  MeasurementOperator SimMeasurements(uv_data, gridSizeU, gridSizeV,gridKernel, 
                      widthUp, heightUp,overSample,cellsize, cellsize, weighting_type,
                      RobustW, use_w_term,  energyC, upsampleRatio,"none",false); // Create Measurement Operator

  t_real sparsityGO = utilities::sparsity_sp(SimMeasurements.G);// sparsity of G -nber on non zero elts over of total number of elts

 
  auto sky= SimMeasurements.grid(uv_data.vis);
  saveMarket( SimMeasurements.G,outputdir +"G.txt") ;
  std::string const sky_model_fits = output_filename(outputdir +"_skyObs.fits");
  pfitsio::write2d((Image<t_real>)sky.real(), sky_model_fits);
  std::cout<<"\nDONE! \n ";
  fflush(stdout); 
  
  

  

  return 0;    



}



