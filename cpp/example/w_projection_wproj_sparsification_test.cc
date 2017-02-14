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

  // std::cout<< Eigen::nbThreads();
  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  FFTOperator fftop;

  /* Inputs of the program */
  const std::string inSkyName = args[1];
  const std::string inVisName = args[2];
  const t_real energy_fraction_chirp = std::stod(static_cast<std::string>(args[4]));
  const std::string egFile = args[3];
  const std::string outputdir = args[5];
  t_real iSNR =  std::stod(static_cast<std::string>(args[6]));
  t_real iWFRAC = std::stod(static_cast<std::string>(args[7]));
  const std::string wfrac = args[7];
  const t_int runs = std::stod(args[8]);
  t_real resolution = std::stod(args[9]); 
  const t_real freq0 = std::stod(args[10]); 

  t_real energy_fraction_wproj = 0.999999;
  const t_real pi = constant::pi;
  const t_real C = constant::c;
  const t_real  arcsec2rad = pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
  const t_int norm_iterations =100;


  std::cout<< "**--------------------------------------------------------------**\n";
  std::cout<< "      This is an example RI imaging with w correction: Study of C sparsity\n";
  std::cout<< "      Sky Model image: "<<inSkyName<<"\n";
  std::cout<< "      UVW coverage: "<<inVisName<<"\n";
  std::cout<< "      Sparsity on G "<<energy_fraction_chirp <<"\n";
  std::cout<< "      Sparsity on C " << egFile <<"\n";
  std::cout<< "      RANDOM w terms -- WFRAC " << wfrac <<"\n";
  std::cout<< "      SNR on the data " << iSNR <<"\n";
  std::cout<< "      Results saved in : "<<outputdir<<"\n";
  std::cout<< "      Reconstruction using PADMM\n";
  std::cout<< "**---------------------------------------------------------------**\n";
  
  /* input files */
  // std::cout<<"\nReading image and any available data\n";
  const std::string  fitsfile = image_filename(inSkyName+".fits");
  const std::string  vis_file =  inVisName;//image_filename("coverages/" + inVisName);
  /*Output files & vars*/
  
  const std::string  ResultsLog = output_filename(outputdir+inSkyName+".RAND.C.Wfrac"+wfrac+".results.txt");
  Vector<t_real> SNR(runs);
  Vector<t_real> MR(runs);
  Vector<t_real> solverTime(runs); 
  

  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_int wavelet_level =4; // max level for the DB decomposition considred for now
  bool use_w_term = true; // correct for w
  utilities::vis_params uv_data;
  t_int nbLevelC=1;


  std::vector<t_real> cList;  
  std::string line;
  std::ifstream clevel (egFile);
  t_real val;
  if (clevel.is_open())
  {
    while ( std::getline(clevel,line) )
    {
      val = std::stod(line);
      cList.push_back(val);
    }
    clevel.close();
    nbLevelC =cList.size();
  }
  
  else std::cout << "Unable to open file!!! \n"; 
  

  /*   Gridding parameters */
  const std::string  kernel_name = "kb"; // choice of the gridding kernel
  const t_real overSample = 2; // oversampling ratio in Fourier due to gridding
  const t_int Ju = 4;
  const t_int Jv = 4;

  /* Read sky model image & dimensions */
  Image<t_complex> skyOr = pfitsio::read2d(fitsfile);
  t_real const maxsky = skyOr.array().abs().maxCoeff();
  skyOr = skyOr / maxsky;
  t_int heightOr = skyOr.rows();
  t_int widthOr = skyOr.cols();
  std::cout<< "\n"<<  "Original test image  of size " << heightOr<< " x "<<widthOr<< " pixels.";
  std::cout<<  "INFO: Original test image  is normalized to 1";

  /* Read visibilities or  uvw coverages from  files */
  bool read_w = true; // read w from uv coverage
  uv_data = utilities::read_uvw( vis_file, lambda); // uvw are  from now  on in units of lambda
  const Vector<t_real> & u = uv_data.u.cwiseAbs();
  const Vector<t_real> & v = uv_data.v.cwiseAbs();
  const Vector<t_real> & w = uv_data.w.cwiseAbs();

  t_real maxBaseline = lambda *(((u.array() * u.array() + v.array() * v.array() + w.array() * w.array()).sqrt()).maxCoeff()) ;
  t_real maxProjectedBaseline = lambda *(((u.array() * u.array() + v.array() * v.array()).sqrt()).maxCoeff()) ;
  t_real thetaResolution = 1.22* lambda / maxProjectedBaseline ;
  t_real cellsize = (t_real)(thetaResolution / arcsec2rad ) / 2;  // Nyquist sampling

  // FoV on L & M axis 
  const t_real theta_FoV_L =  cellsize * widthOr * arcsec2rad; 
  const t_real theta_FoV_M = cellsize * heightOr * arcsec2rad;
  std::cout << "\nObservation Specs: freq0 "<< freq0 <<", max baseline " << maxBaseline << " m,  angular resolution " << thetaResolution << " rad, cellsize "
   << cellsize <<  " arcsec, "<< "FoV " <<theta_FoV_L  <<" rad.";

  /* Set w terms */
  const t_real L = 2 * std::sin(theta_FoV_L  * 0.5);
  const t_real M = 2 * std::sin(theta_FoV_M * 0.5);
  const t_real wlimit = widthOr / (L * M); 
  t_real w_max = (uv_data.w.cwiseAbs()).maxCoeff();
  std::cout <<"\nINFO: original w_max " << w_max <<", limits on w: wrt FoV & Npix " << wlimit << ", wrt b_max " << maxBaseline /lambda ;  

  if (iWFRAC >0){
    std::cout<< "\nGenerating random w component.. ";
      std::random_device rand;
      std::mt19937 gen(1);
      std::uniform_real_distribution<t_real> dist(-1, 1);
      auto generator = std::bind(dist, gen);
      for (int i = 0; i < u.size(); ++i){
           uv_data.w(i) = iWFRAC*wlimit*dist(gen);
      }
      w_max = (uv_data.w.cwiseAbs()).maxCoeff();
  } 
  else{
    if (iWFRAC==-1)   std::cout << "\nINFO: Keeping  original w components ";
  }
       
  std::cout <<"  wmax/wlimit = " << w_max/wlimit <<", wfrac:" << iWFRAC <<".\n" ;
   fflush(stdout); 
  /* Upsampling in Fourier due to w component - image size setting - keep even size for now */
  t_int multipleof =pow(2,wavelet_level)/overSample;   // size should be multiple of 2^level (level of DB wavelets)
  t_real upsampleRatio = wproj_utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr, multipleof);
  Matrix<t_complex> skypaddedFFT = utilities::re_sample_ft_grid(fftop.forward(skyOr), upsampleRatio);
  Matrix<t_complex> skyUP = fftop.inverse(skypaddedFFT);
  t_real const maxskyUp = skyUP.array().abs().maxCoeff();
  t_int heightUp = skyUP.rows();
  t_int widthUp = skyUP.cols();
  upsampleRatio = (t_real)heightUp / (t_real)heightOr;
  if (upsampleRatio >1){
    std::cout <<  "\nUpsampling: cellsize " << cellsize <<  "arcsec, upsampling ratio: "<< upsampleRatio << " New Image size " << heightUp<< " x "<<widthUp<< " pixels. \n";
    cellsize = cellsize/upsampleRatio; //update cellsize
  }
  else{std::cout<<"INFO: no upsampling.\n";}

  /* Measurement operator parameters */
  std::string weighting_type = "none";
  t_real RobustW = 0; // robust weighting parameter
  uv_data = utilities::set_cell_size(uv_data, upsampleRatio*maxProjectedBaseline /lambda,upsampleRatio*maxProjectedBaseline / lambda); // scale uv coordinates to correct pixel size and to units of 2pi
  uv_data = utilities::uv_scale(uv_data, floor(widthUp * overSample), floor(heightUp * overSample)); // scale uv coordinates to units of Fourier grid size
  
  //nber of pixels to upsample wrt to each w term - to be saved in a file
  Vector<t_real> uvdist = (u.array() * u.array() + v.array() * v.array()).sqrt();
  t_real Blength = 2 * uvdist.maxCoeff();


  /*--------------------------------------------------------------------------*/
  // Building the data
  /*--------------------------------------------------------------------------*/
  PURIFY_INFO("Setting measurement operator ..");
  MeasurementOperator SimMeasurements(uv_data, Ju, Jv,kernel_name, 
                      widthUp, heightUp,norm_iterations,overSample,cellsize, cellsize, weighting_type,
                      RobustW, use_w_term, energy_fraction_chirp,energy_fraction_wproj,"none",false); 

  //Generating DATA 
  
  Image<t_complex> sky(widthOr, heightOr);
  
  sky.real()=skyOr.real();
  sky.imag()=skyOr.imag()*0;
  t_int height = sky.rows();
  t_int width = sky.cols();
 
  PURIFY_INFO("Generating noise free measurements");
  uv_data.vis = SimMeasurements.degrid(sky);
  std::string const sky_model_fits = output_filename(outputdir +inSkyName+"_GroundTruth.fits");
  pfitsio::write2d((Image<t_real>)sky.real(), sky_model_fits);

  PURIFY_INFO("Setting operators for SOPT");
  
  /* SARA Dictionairies */
  auto measurements_transform = linear_transform(SimMeasurements, uv_data.vis.size());
      std::vector<std::tuple<std::string, t_uint>> wavelets;    
        wavelets.push_back(std::make_tuple("Dirac" ,3u));
        wavelets.push_back(std::make_tuple("DB1" ,3u));
        wavelets.push_back(std::make_tuple("DB2" ,3u));
        wavelets.push_back(std::make_tuple("DB3" ,3u));
        wavelets.push_back(std::make_tuple("DB4" ,3u));
        wavelets.push_back(std::make_tuple("DB5" ,3u));
        wavelets.push_back(std::make_tuple("DB6" ,3u));
        wavelets.push_back(std::make_tuple("DB7" ,3u));
        wavelets.push_back(std::make_tuple("DB8" ,3u));    
      sopt::wavelets::SARA const sara(wavelets.begin(), wavelets.end());
      auto const Psi = sopt::linear_transform<t_complex>(sara, SimMeasurements.imsizey(), SimMeasurements.imsizex());

      /* Data */
      // working out value of sigma given SNR of 30
      t_real sigma = utilities::SNR_to_standard_deviation(uv_data.vis, iSNR);
      // adding noise to visibilities
      Vector<t_complex> const y0 = uv_data.vis;
      Vector<t_complex> const inputData0 = utilities::add_noise(uv_data.vis, 0., sigma);
      auto const epsilon = utilities::calculate_l2_radius(uv_data.vis, sigma);
      auto const purify_gamma
         = (Psi.adjoint() * (measurements_transform.adjoint() * uv_data.vis)).real().maxCoeff() * 1e-3;
      t_int iters = 0;
      auto convergence_function = [&iters](const Vector<t_complex> &x) { iters = iters + 1; return true; };    
      PURIFY_INFO("Chirp sparsification tests ...");  
  
    for (t_int CMPT= 0; CMPT < cList.size(); ++CMPT){
        
        energy_fraction_wproj = cList[CMPT];
        std::string eg=std::to_string(CMPT);
        PURIFY_INFO(" Thread Nb:{} Energy Fraction on Chirp: {} ",omp_get_thread_num(),energy_fraction_wproj);

        MeasurementOperator SimMeasurementsSP(uv_data, Ju, Jv,kernel_name, 
                      widthUp, heightUp,norm_iterations,overSample,cellsize, cellsize, weighting_type,
                      RobustW, use_w_term, energy_fraction_chirp,energy_fraction_wproj,"none",false); // Create Measurement Operator
 
        auto measurements_transform_approx = linear_transform(SimMeasurementsSP, uv_data.vis.size());
       
        /* Reconstruction */
        Image<t_complex> skyDirty = (SimMeasurementsSP.grid(y0));
        std::string const dirty_image_fits = output_filename(outputdir +inSkyName+".EcIdx"+eg+".C_sara_dirty.fits");  
        pfitsio::write2d((Image<t_real>)skyDirty.real(), dirty_image_fits);
        std::cout << "\n\nStarting SOPT!\n";
       
        Vector<t_real> dimage = (measurements_transform_approx.adjoint() * y0).real();//dirty image 
        Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());

        #pragma omp parallel for num_threads(12)
        for (t_int i = 0; i < runs; ++i){
  
            // std::cout << "DEBUG 11!\n";  fflush(stdout);
            std::time_t startT = std::time(NULL);
            Vector<t_complex> inputData = utilities::add_noise(y0, 0., sigma);//dirty(y0, mersenne, inputSNR);//  noisy vis
            std::string const model_fits = output_filename(outputdir +inSkyName+"G.EgIdx"+eg+"_sara_model"+std::to_string(i)+".fits");
            std::string const residual_fits = output_filename(outputdir +inSkyName+"G.EgIdx"+eg+"_sara_residual"+std::to_string(i)+".fits");
            std::string const dirty_fits = output_filename(outputdir +inSkyName+"G.EgIdx"+eg+"_sara_dirty"+std::to_string(i)+".fits");

            
            /* SOLVER: Proximal ADMM */
            #pragma omp critical (print2)
            {
              PURIFY_INFO("RAND-W EG:{}  RUN: {} ",energy_fraction_wproj, i);
            }
            t_real solver_beta=0.001;
            auto purify_gamma = (Psi.adjoint() * (measurements_transform_approx.adjoint() * (inputData.array()).matrix()) ).real().cwiseAbs().maxCoeff() * solver_beta;          
          
            PURIFY_HIGH_LOG("Starting sopt!");
            PURIFY_MEDIUM_LOG("Epsilon {}", epsilon);
            PURIFY_MEDIUM_LOG("Gamma {}", purify_gamma);
            auto const padmm = sopt::algorithm::ImagingProximalADMM<t_complex>(uv_data.vis)
                         .gamma(purify_gamma)
                         .relative_variation(1e-5)
                         .l2ball_proximal_epsilon(epsilon * 1.005)
                         .tight_frame(false)
                         .l1_proximal_tolerance(1e-2)
                         .l1_proximal_nu(1)
                         .l1_proximal_itermax(50)
                         .l1_proximal_positivity_constraint(true)
                         .l1_proximal_real_constraint(true)
                         .residual_convergence(epsilon * 1.001)
                         .lagrange_update_scale(0.9)
                         .nu(1e0)
                         .Psi(Psi)
       .itermax(2500)
       .is_converged(convergence_function)
                         .Phi(measurements_transform_approx);

      // Solver Timing 
      std::clock_t c_start = std::clock();
      auto const result = padmm();
      std::clock_t c_end = std::clock();
      auto total_time = (c_end - c_start) / CLOCKS_PER_SEC; // total time for solver to run in seconds

      // Reading if algo has converged
      t_int converged = 0;
      if (result.good) 
        converged = 1;
      const t_uint maxiters = iters;
      Image<t_complex> estimated_image = Image<t_complex>::Map(result.x.data(), SimMeasurementsSP.imsizey(), SimMeasurementsSP.imsizex());
      pfitsio::write2d((Image<t_real>)estimated_image.real(), model_fits);

      auto model_visibilities = SimMeasurementsSP.degrid(estimated_image);
      auto residual = SimMeasurementsSP.grid(inputData - model_visibilities);
      auto dirty_image  =SimMeasurementsSP.grid(inputData);
      pfitsio::write2d((Image<t_real>)residual.real(), residual_fits);
      pfitsio::write2d((Image<t_real>)dirty_image.real(), dirty_fits);
      SNR(i) = wproj_utilities::snr_metric(sky.real(),estimated_image.real());
      MR(i) = wproj_utilities::mr_metric(sky.real(),estimated_image.real());
      solverTime(i) = total_time;
      PURIFY_HIGH_LOG("[PADMM: Thread Nbr: {} -  ChirP Energy: {} -  Run:{}] SNR:{}  &&  MR: {} && CT:{}",omp_get_thread_num(),energy_fraction_chirp,i,SNR(i),MR(i),solverTime(i));
      }
  

          SimMeasurementsSP.G.resize(0,0);
          
          
          //SNR
          std::string const fileSNRLog = output_filename(outputdir+inSkyName+"W."+wfrac+"_SNR.txt");
          std::ofstream SNRLog;   
          SNRLog.open(fileSNRLog, std::ios::app);
          for (t_int i = 0; i < runs; ++i){
                SNRLog<<eg<<" "<< SNR(i) << "\n";
          }
          SNRLog.close();

          //MR
          std::string const fileMRLog = output_filename(outputdir+inSkyName+"W."+wfrac+"_MR.txt");
          std::ofstream MRLog;   
          MRLog.open(fileMRLog, std::ios::app);
          for (t_int i = 0; i < runs; ++i){
                MRLog<<eg<<" "<< MR(i) << "\n";
          }
          MRLog.close();
        
          // Solver compt. time
          std::string const fileSolverTLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_CTime.txt");
          std::ofstream CTLog;
          CTLog.open(fileSolverTLog, std::ios::app);
          for (t_int i = 0; i < runs; ++i){
               CTLog<<eg<<" "<< solverTime(i) << "\n";  // CTLog<<<< endl;td::fixed <<std::setprecision(10)<<
          }
          CTLog.close();
        
          // // G - sparsity
          // std::string const fileGspLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_Gsp.txt");
          // std::ofstream GSLog;   
          // GSLog.open(fileGspLog, std::ios::app);
          // GSLog<<ec << " " <<sparsityG<< std::endl;
          // GSLog.close();


          // // C - sparsity
          // std::string const fileCspLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_Csp.txt");
          // std::ofstream CSLog;   
          // CSLog.open(fileCspLog, std::ios::app);
          // CSLog<<ec << " " <<sparsityC<< std::endl;
          // CSLog.close();
      
      }
  return 0; 

}



