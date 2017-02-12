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
  const t_int kernelSizeU = 4;
  const t_int kernelSizeV = 4;
  
  /* constants */
  const t_real lambda = C/freq0; // wavelength 
  const t_int wavelet_level = 4; // max level for the DB decomposition considred for now
  bool correct_w_term = true; // correct for w
  utilities::vis_params uv_data;
  t_int norm_iterations =200;

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
      
      if (iWFRAC >0){
          PURIFY_INFO("W-terms: Generating random w component.. ");
          std::random_device randi;
          std::mt19937 gen(randi());
          std::uniform_real_distribution<t_real> dist(-1, 1);
          auto generator = std::bind(dist, gen);
          for (int p = 0; p < u.size(); ++p)
              uv_data.w(p) = (1-resolution)* iWFRAC*wlimit*dist(gen);        
          w_max = (uv_data.w.cwiseAbs()).maxCoeff();
      } 
      if (iWFRAC == -1)   PURIFY_INFO("W-terms: Keeping  original w components ");
      if (iWFRAC == 0 ){
        uv_data.w = uv_data.w *0.0;
        PURIFY_INFO( "W-terms: no w-modulation");
      }
      w_max = (uv_data.w.cwiseAbs()).maxCoeff();

      /* Upsampling in Fourier due to w component - image size setting - keep even size for now */
      t_int multipleof = pow(2,wavelet_level)/overSample;   // size should be multiple of 2^level (level of DB wavelets)
      t_real upsampleRatio = 1;// utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr, multipleof);
      Matrix<t_complex> skyUP = ground_truth;//fftop.inverse(skypaddedFFT);
      t_real const maxskyUp = skyUP.array().abs().maxCoeff();
      t_int heightUp = skyUP.rows();
      t_int widthUp = skyUP.cols();
      upsampleRatio = (t_real)heightUp / (t_real)heightOr;

      if (upsampleRatio >1){
          cell_x = cell_x/upsampleRatio; //update cellsize
          cell_y = cell_x;
      }
      else{std::cout<<"INFO: no upsampling.\n";}
      PURIFY_HIGH_LOG( "Testing Super Resolution: scaling down uv by: {}  ",resolution);
      // uv_data.u=uv_data.u*resolution;
      // uv_data.v=uv_data.v*resolution;

      /* Measurement operator parameters */
      std::string weighting_type = "none";
      t_real RobustW = 0; // robust weighting parameter     
      uv_data.u = ( uv_data.u / (upsampleRatio*maxProjectedBaseline /lambda) )*resolution * pi;// converting u,v to [-pi pi]
      uv_data.v = ( uv_data.v / (upsampleRatio*maxProjectedBaseline /lambda) )*resolution * pi;
      uv_data = utilities::uv_scale(uv_data, floor(widthUp * overSample), floor(heightUp * overSample)); // scale uv coordinates to units of Fourier grid size
      const t_real energy_fraction_chirp =energyC;
      const t_real energy_fraction_wproj =energyG;
      PURIFY_INFO("Setting measurement operator");
      MeasurementOperator SimMeasurements(uv_data, kernelSizeU, kernelSizeV,kernel_name, 
                      widthUp, heightUp,norm_iterations,overSample,cell_x, cell_y, weighting_type,
                      RobustW, true, energy_fraction_chirp,energy_fraction_wproj,"none",false); // Create Measurement Operator
      SimMeasurements.C.resize(0,0);

      //Generating DATA 
      Image<t_complex> sky(widthOr, heightOr);
      sky.real()=ground_truth.real();
      sky.imag()=ground_truth.imag()*0;
      t_int height = sky.rows();
      t_int width = sky.cols();
      std::cout<<"\nGenerating noise free measurement.\n";
      uv_data.vis = SimMeasurements.degrid(sky);
      std::string const sky_model_fits = output_filename(outputdir +inSkyName+"_GroundTruth.fits");
      pfitsio::write2d((Image<t_real>)sky.real(), sky_model_fits);
     
      // PURIFY_INFO("Setting operators for SOPT!"); 
      // auto measurements_transform = linear_transform(SimMeasurements, uv_data.vis.size());
      // std::vector<std::tuple<std::string, t_uint>> wavelets;    
      //   wavelets.push_back(std::make_tuple("Dirac" ,3u));
      //   wavelets.push_back(std::make_tuple("DB1" ,3u));
      //   wavelets.push_back(std::make_tuple("DB2" ,3u));
      //   wavelets.push_back(std::make_tuple("DB3" ,3u));
      //   wavelets.push_back(std::make_tuple("DB4" ,3u));
      //   wavelets.push_back(std::make_tuple("DB5" ,3u));
      //   wavelets.push_back(std::make_tuple("DB6" ,3u));
      //   wavelets.push_back(std::make_tuple("DB7" ,3u));
      //   wavelets.push_back(std::make_tuple("DB8" ,3u));    
      // sopt::wavelets::SARA const sara(wavelets.begin(), wavelets.end());
      // auto const Psi = sopt::linear_transform<t_complex>(sara, SimMeasurements.imsizey(), SimMeasurements.imsizex());

      // // working out value of sigma given SNR of 30
      // t_real sigma = utilities::SNR_to_standard_deviation(uv_data.vis, iSNR);
      // // adding noise to visibilities
      // Vector<t_complex> const inputData0 = utilities::add_noise(uv_data.vis, 0., sigma);
      // Vector<> dimage = (measurements_transform.adjoint() * uv_data.vis).real();
      // Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());

      // auto const epsilon = utilities::calculate_l2_radius(uv_data.vis, sigma);
      // auto const purify_gamma
      //    = (Psi.adjoint() * (measurements_transform.adjoint() * uv_data.vis)).real().maxCoeff() * 1e-3;
      // t_int iters = 0;
      // auto convergence_function = [&iters](const Vector<t_complex> &x) { iters = iters + 1; return true; };    
      
      // std::string const model_fits = output_filename(outputdir +inSkyName+"_sara_model"+std::to_string(cmt)+".fits");
      // std::string const residual_fits = output_filename(outputdir +inSkyName+"_sara_residual"+std::to_string(cmt)+".fits");
      // std::string const residual_fitsT = output_filename(outputdir +inSkyName+"_sara_residualTrue"+std::to_string(cmt)+".fits");

      // PURIFY_HIGH_LOG("Starting sopt!");
      // PURIFY_MEDIUM_LOG("Epsilon {}", epsilon);
      // PURIFY_MEDIUM_LOG("Gamma {}", purify_gamma);
      // auto const padmm = sopt::algorithm::ImagingProximalADMM<t_complex>(uv_data.vis)
      //                    .gamma(purify_gamma)
      //                    .relative_variation(1e-5)
      //                    .l2ball_proximal_epsilon(epsilon * 1.001)
      //                    .tight_frame(false)
      //                    .l1_proximal_tolerance(1e-2)
      //                    .l1_proximal_nu(1)
      //                    .l1_proximal_itermax(50)
      //                    .l1_proximal_positivity_constraint(true)
      //                    .l1_proximal_real_constraint(true)
      //                    .residual_convergence(epsilon * 1.001)
      //                    .lagrange_update_scale(0.9)
      //                    .nu(1e0)
      //                    .Psi(Psi)
      //  .itermax(2)
      //  .is_converged(convergence_function)
      //                    .Phi(measurements_transform);

      // // Solver Timing 
      // std::clock_t c_start = std::clock();
      // auto const result = padmm();
      // std::clock_t c_end = std::clock();
      // auto total_time = (c_end - c_start) / CLOCKS_PER_SEC; // total time for solver to run in seconds

      // // Reading if algo has converged
      // t_int converged = 0;
      // if (result.good) 
      //   converged = 1;
      // const t_uint maxiters = iters;
      
      // //saving results
      // Image<t_complex> estimated_image = Image<t_complex>::Map(result.x.data(), SimMeasurements.imsizey(), SimMeasurements.imsizex());
      // pfitsio::write2d((Image<t_real>)estimated_image.real(), model_fits);

      // auto model_visibilities = SimMeasurements.degrid(estimated_image);
      // auto residual = SimMeasurements.grid(inputData0 - model_visibilities);
      // pfitsio::write2d((Image<t_real>)residual.real(), residual_fits);

      // SNR(cmt) = wproj_utilities::snr_metric(sky.real(),estimated_image.real());
      // MR(cmt) = wproj_utilities::mr_metric(sky.real(),estimated_image.real());
      // solverTime(cmt) = total_time;
      // PURIFY_HIGH_LOG("[PADMM -- Thread Nbr: {} -  Gsparse: {} -  Run:{}] SNR:{}  &&  MR: {} && CT:{}",omp_get_thread_num(),energyG,cmt,SNR(cmt),MR(cmt),solverTime(cmt));
  }
  return 0; 

}



