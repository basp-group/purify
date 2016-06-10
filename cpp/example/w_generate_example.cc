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
  std::string inSkyName=args[1];
  std::string inVisName=args[2];
  std::string eg = args[3];
  t_real energyG=std::stod(eg);

  std::string ec = args[4];
  t_real energyC=std::stod(ec);

  std::string outputdir=args[5];
  t_real inputSNR=atoi(args[6]); 
  t_real inWFRAC=std::stod(args[7]);
  std::string wfrac =args[7];

  t_real inputFOV=std::stod(args[9]);//NOT USED AT THE MOMENT
  t_int runi=std::stod(args[8]);
  


  std::cout<< "**--------------------------------------------------------------**\n";
  std::cout<< "      This is an example RI imaging with w correction\n";
  std::cout<< "      Sky Model image: "<<inSkyName<<"\n";
  std::cout<< "      UVW coverage: "<<inVisName<<"\n";
  std::cout<< "      Sparsity on G " << energyG <<"\n";
  std::cout<< "      Sparsity on C " << energyC <<"\n";
  std::cout<< "      WRAC " << wfrac <<"\n";
  std::cout<< "      FoV (!not used) " << inputFOV <<"\n";
  std::cout<< "      SNR on the data " << inputSNR <<"\n";
  std::cout<< "      Result saved in : "<<outputdir<<"\n";
  std::cout<< "      Reconstruction using PADMM\n";
  std::cout<< "**---------------------------------------------------------------**\n";

  //input files
  std::string const fitsfile = image_filename(inSkyName+".fits");
  std::string const vis_file = inVisName;//image_filename("coverages/" + inVisName);
  //ouput files -vars
  // std::string const dirty_image_fits = output_filename(outputdir+inSkyName+"_sara_dirty.fits");  
  // std::string const ResultsLog = output_filename(outputdir+inSkyName+"G"+eg+"Wfrac"+wfrac+"results.txt");
  Vector<t_real> SNR(runi);
  Vector<t_real> solverTime(runi);
  
  //consts
  std::cout<<"\nIt s a minor issue - but just make sure to update the freq of the uv coverage adopted \n";
  const t_real C = 299792458;
  const t_real freq0 = 1000e6;
  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_real  arcsecTrad = purify_pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
  const t_int wavelet_level =4; // max level for the DB decomposition considred for now
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

   
  // Read sky model image & dimensions
  auto skyOr = pfitsio::read2d(fitsfile);
  t_real const maxsky = skyOr.array().abs().maxCoeff();
  skyOr = skyOr / maxsky;
  t_int heightOr = skyOr.rows();
  t_int widthOr = skyOr.cols();
  std::cout<< "\n"<<  "Original test image  of size " << heightOr<< " x "<<widthOr<< " pixels. \n";
  std::cout<<  "\nINFO: Original test image  is normalized to 1 \n";

  // Read visibilities | uvw coverages from files 
  bool read_w = true; // read w from uv coverage
  uv_data = utilities::read_uvw_sim( vis_file, lambda, read_w); // uvw are  from now  on in units of lambda
  const Vector<t_real> & u = uv_data.u.cwiseAbs();
  const Vector<t_real> & v = uv_data.v.cwiseAbs();
  const Vector<t_real> & w = uv_data.w.cwiseAbs();
  
  //setting resolution - cellsize
  t_real maxBaseline = lambda *(((u.array() * u.array() + v.array() * v.array() + w.array() * w.array()).sqrt()).maxCoeff()) ;
  t_real maxProjectedBaseline = lambda *(((u.array() * u.array() + v.array() * v.array()).sqrt()).maxCoeff()) ;
  t_real thetaResolution = 1.22* lambda / maxProjectedBaseline ;
  t_real cellsize = (thetaResolution / arcsecTrad ) / 2;  // Nyquist sampling

  // setting FoV on L & M axis 
  const t_real theta_FoV_L = cellsize * widthOr * arcsecTrad; 
  const t_real theta_FoV_M = cellsize * heightOr * arcsecTrad;

  std::cout << "\nFreq0 "<< freq0 <<", max baseline " << maxBaseline << " m,  angular resolution " << thetaResolution << " rad, cellsize "
   << cellsize <<  " arcsec, "<< "FoV " <<theta_FoV_L  <<" rad. \n";

  /* 
     Set w terms
  */
  const t_real L = 2 * std::sin(theta_FoV_L  * 0.5);
  const t_real M = 2 * std::sin( theta_FoV_M * 0.5);
  const t_real wlimit = widthOr / (L * M); 
  t_real w_max = (uv_data.w.cwiseAbs()).maxCoeff();
  std::cout <<"\nINFO: original w_max " << w_max <<", limits on w: wrt FoV & Npix " << wlimit << ", wrt b_max " << maxBaseline /lambda << ".\n" ;
  if (inWFRAC >0){
     std::cout<< "Rescaling the w component.. ";
     uv_data.w = uv_data.w * inWFRAC * wlimit / w_max ;
     w_max = (uv_data.w.cwiseAbs()).maxCoeff();

      /*std::cout<< "Generating random  w components.. ";
      std::random_device rand;
      std::mt19937 gen(rand());
      std::uniform_real_distribution<t_real> dist(-1*inWFRAC*wlimit, 1*inWFRAC*wlimit);
      auto generator = std::bind(dist, gen);
      for (int i = 0; i < u.size(); ++i){
           uv_data.w(i) = inWFRAC*wlimit*dist(gen);
      }
      w_max = (uv_data.w.cwiseAbs()).maxCoeff();
      */
  } 
  else{
    if (inWFRAC==-1)   std::cout << "Keeping  original w components ";
  }     

  std::string const fileWvalues = inVisName+".Wfrac"+wfrac+".txt";
  std::cout<<"\nWriting the w values in Filename:\n " <<fileWvalues<<"\n";

  std::ofstream Wvals;
  Wvals.open(fileWvalues, std::ios::app);
  for (t_int m = 0; m < uv_data.u.size(); ++m){
       Wvals<< uv_data.w(m) <<"\n";
  }
  Wvals.close();

   
  std::cout <<"wmax/wlimit = " << w_max/wlimit <<", wfrac:" << inWFRAC <<".\n" ;
 
  // Upsampling in Fourier due to w component - image size setting - keep even size for now

  t_int multipleOf =pow(2,wavelet_level)/overSample; // size should be multiple of 2^level (level of DB wavelets)
  t_real upsampleRatio = utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr,multipleOf);
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
  
  //parameters
  // std::string weighting_type = "none";
  // t_real RobustW = 0; // robust weighting parameter
  // uv_data = utilities::set_cell_size_sim(uv_data, upsampleRatio*maxProjectedBaseline /lambda,upsampleRatio*maxProjectedBaseline / lambda); // scale uv coordinates to correct pixel size and to units of 2pi
  // uv_data = utilities::uv_scale(uv_data, floor(widthUp * overSample), floor(heightUp * overSample)); // scale uv coordinates to units of Fourier grid size
  
  //nber of pixels to upsample wrt to each w term - to be saved in a file - just for info.
  // Vector<t_real> uvdist = (u.array() * u.array() + v.array() * v.array()).sqrt();
  // t_real Blength = 2 * uvdist.maxCoeff();
  // Vector<t_real> w_kernel_size = (widthOr *uv_data.w * L / Blength);
  // std::string const fileWSize = output_filename(outputdir+"WSizeOr.txt");
  // std::ofstream WSize;
  // WSize.open(fileWSize, std::ios::app);
  // for (t_int m = 0; m < uv_data.u.size(); ++m){
  //      WSize<< floor(std::abs(w_kernel_size(m))) <<"\n";
  // }
  // WSize.close();

  

  return 0;    



}



