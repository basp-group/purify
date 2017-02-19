#include <array>
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <math.h>
#include <string>
#include <stdio.h>
#include <omp.h>
#include "purify/config.h"
#include <array>
#include <ctime>
#include <memory>
#include <random>
#include <boost/math/special_functions/erf.hpp>
#include <sopt/imaging_padmm.h>
#include <sopt/relative_variation.h>
#include <sopt/utilities.h>
#include <sopt/wavelets.h>
#include <sopt/wavelets/sara.h>
#include "purify/MeasurementOperator.h"
#include "purify/directories.h"
#include "purify/logging.h"
#include "purify/pfitsio.h"
#include "purify/types.h"
#include "purify/utilities.h"
#include "purify/wproj_utilities.h"
#include <Unsupported/Eigen/SparseExtra>

#include <assert.h>
       
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_AUTOMATIC_RESIZING

 
int main( int nargs, char const** args ){
  std::cout<<nargs;
  if(nargs != 11) {
    std::cerr << "Wrong number of arguments! " << '\n';
    return 1;
  }

  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  purify::logging::initialize();
  sopt::logging::set_level("debug");
  purify::logging::set_level("debug");

  const t_real pi = constant::pi;
  const t_real C = constant::c;
  const t_real  arcsec2rad = pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian


  /* reading entries */
  const std::string inSkyName = args[1];
  const std::string inVisName = args[2];
  const t_real energyG = std::stod(static_cast<std::string>(args[3]));
  const t_real energyC = std::stod(static_cast<std::string>(args[4]));
  const std::string outputdir = args[5];
  t_real iSNR =  std::stod(static_cast<std::string>(args[6]));
  t_real iWFRAC = std::stod(static_cast<std::string>(args[7]));
  const std::string wfrac = args[7];
  const t_int runs = std::stod(args[8]);
  t_real resolution = std::stod(args[9]); 
  const t_real freq0 = std::stod(args[10]); 

  std::cout<< "**---------------------------------------------------------------------------**\n";
  std::cout<< "  This is an example of RI imaging: Super-resolution in presence of w-term\n";
  std::cout<< "  Sky Model image: "<<inSkyName<<"\n";
  std::cout<< "  UVW coverage: "<<inVisName<<"\n";
  std::cout<< "  Sparsity on G: "<<energyG <<"\n";
  std::cout<< "  Random w-terms; w_f: " << wfrac <<"\n";
  std::cout<< "  Nber of runs: " << runs <<"\n";
  std::cout<< "  UV down-scaling factor: " << resolution <<"\n";
  std::cout<< "  iSNR: " << iSNR <<"\n";
  std::cout<< "  Results saved in : "<<outputdir<<"\n";
  std::cout<< "**---------------------------------------------------------------------------**\n";

  /*   Gridding parameters */
  const std::string  kernel_name = "kb"; // choice of the gridding kernel
  const t_real overSample = 2; // oversampling ratio in Fourier due to gridding
  const t_int kernelSizeU = 8;
  const t_int kernelSizeV = 8;
  
  /* constants */
  const t_real lambda = C/freq0; // wavelength 
  const t_int wavelet_level = 4; // max level for the DB decomposition considred for now
  bool correct_w_term = true; // correct for w
  utilities::vis_params uv_data;
  t_int norm_iterations =10;

  /*Output files & vars*/
  Vector<t_real> SNR(runs);
  Vector<t_real> MR(runs);
  Vector<t_real> solverTime(runs); 

  /* Read sky model image & dimensions*/
  PURIFY_INFO("Reading image and any available data!");
  const std::string  fitsfile = image_filename(inSkyName);
  const std::string  vis_file =  inVisName;//image_filename("coverages/" + inVisName);
  Image<t_complex> ground_truth = pfitsio::read2d(fitsfile);
  t_real const ground_truth_peak = ground_truth.array().abs().maxCoeff();
  ground_truth = ground_truth / ground_truth_peak;
  t_int heightOr = ground_truth.rows();
  t_int widthOr = ground_truth.cols();
  t_int finalszx = widthOr*overSample;
  t_int finalszy = heightOr*overSample;
  
  PURIFY_HIGH_LOG("Original test image  is normalized to 1!");
  PURIFY_INFO("Original test image is of size {}  x  {} pixels",heightOr,widthOr);

  for (t_int cmt = runs-1; cmt < runs; ++cmt) {
      /* Read visibilities or  uvw coverages from  files */
      uv_data = utilities::read_uvw(vis_file, lambda); // uvw are from now on in units of lambda
      const Vector<t_real> & u = uv_data.u.cwiseAbs();
      const Vector<t_real> & v = uv_data.v.cwiseAbs();
      const Vector<t_real> & w = uv_data.w.cwiseAbs();
      PURIFY_INFO("Number of measurements: {}",u.size());
      t_real maxBaseline = lambda *(((u.array() * u.array() + v.array() * v.array() + w.array() * w.array()).sqrt()).maxCoeff()) ;
      t_real maxProjectedBaseline = lambda *(((u.array() * u.array() + v.array() * v.array()).sqrt()).maxCoeff()) ;
      t_real thetaResolution = 1.22* lambda / (maxProjectedBaseline) ;
      t_real cell_x = (thetaResolution / arcsec2rad ) / 2;  // Nyquist sampling
      t_real cell_y = cell_x;
      // FoV on L & M axis 
      const t_real theta_FoV_L = cell_x * widthOr * arcsec2rad; 
      const t_real theta_FoV_M = cell_y * heightOr * arcsec2rad;
      PURIFY_INFO("Observation Specs: freq0 {}, max baseline {} meters, angular resolution {}, cell size {} arcsec, FoV {} rad.", freq0, maxBaseline, thetaResolution,cell_x,theta_FoV_L);

       /* Set w terms */
      const t_real L = 2 * std::sin(theta_FoV_L  * 0.5); //projected 2D plane 
      const t_real M = 2 * std::sin( theta_FoV_M * 0.5); //projected 2D plane 
      const t_real wlimit = widthOr / (L * M); 
      t_real w_max = (uv_data.w.cwiseAbs()).maxCoeff();
      PURIFY_INFO("W-terms: original w_max {}, limits on w: wrt FoV & Npix {} and wrt b_max {}",w_max,wlimit,maxBaseline /lambda);

      /* Measurement operator parameters */
      std::string weighting_type = "none";
      t_real RobustW = 0; // robust weighting parameter     
      uv_data.u = ( uv_data.u / (maxProjectedBaseline /lambda) )*resolution * pi;// converting u,v to [-pi pi]
      uv_data.v = ( uv_data.v / (maxProjectedBaseline /lambda) )*resolution * pi;
      uv_data = utilities::uv_scale(uv_data, floor(widthOr * overSample), floor(heightOr * overSample)); // scale uv coordinates to units of Fourier grid size
      const t_real energy_fraction_chirp =energyC;
      const t_real energy_fraction_wproj =energyG;
      PURIFY_INFO("Setting measurement operator");
      MeasurementOperator SimMeasurements(uv_data, kernelSizeU, kernelSizeV,kernel_name, 
                      widthOr, heightOr,norm_iterations,overSample,cell_x, cell_y, weighting_type,
                      RobustW, false, energy_fraction_chirp,energy_fraction_wproj,"none",false); // Create Measurement Operator
      t_real sigma =1;
      t_real mean =0;
    
      Sparse<t_complex> vect  = wproj_utilities::generate_vect(finalszx,finalszy,sigma,mean);
      Eigen::saveMarket(vect.transpose(),"./outputs/vect.txt");
      for (t_int m=1; m<SimMeasurements.G.outerSize() ;m++){
      PURIFY_DEBUG("CURRENT WPROJ - Kernel index [{}]",m);

      Eigen::SparseVector<t_complex> G_bis = SimMeasurements.G.row(m);
      string name ="./outputs/gbis"+std::to_string(m)+".txt";
      Eigen::saveMarket(G_bis,name);

      t_int cmpt1=0;     
      std::cout<<"\n"; fflush(stdout);
      
      std::cout<<"\nIn  1: "<<G_bis.nonZeros()<<" elements";fflush(stdout);
      std::cout<<"\nIn  2: "<<vect.nonZeros()<<" elements";fflush(stdout); 
      std::cout<<"\n"; fflush(stdout);
      // Sparse<t_complex> out_conv(finalszx*finalszx,1);
      auto out_conv= wproj_utilities::row_wise_convolution(G_bis,vect,finalszx,finalszy);
      string name_ ="./outputs/out"+std::to_string(m)+".txt";
      Eigen::saveMarket(out_conv.transpose(),name_);

      std::cout<<"\nOUT VECT 1: "<<out_conv.nonZeros()<<" elements\n";fflush(stdout);
     
      
      std::cout<<"\n"; fflush(stdout);
    }
      
    }
  return 0; 

}