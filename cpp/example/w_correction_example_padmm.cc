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

#include <cstddef>
#include <cstdlib>
#include <iterator>


int main( int nargs, char const** args ){
  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  FFTOperator fftop;

 /* Inputs of the program */
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
  // std::string inSkyName=args[1];

  // std::string inVisName=args[2];

  // std::string eg = args[3];
  // t_real energyG=std::stod(eg);
  
  // std::string ec = args[4];
  // t_real energyC=std::stod(ec);

  // std::string outputdir=args[5];

  // t_real inputSNR=atoi(args[6]); 

  // t_real inWFRAC=std::stod(args[7]);
  // std::string wfrac =args[7];

  // t_real inputFOV=std::stod(args[8]);//NOT USED AT THE MOMENT



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

  
  std::string const fitsfile = image_filename(inSkyName+".fits");
  std::string const vis_file = inVisName;//image_filename("coverages/" + inVisName);

  std::string const dirty_image_fits = output_filename(outputdir+inSkyName+"_sara_dirty.fits");  
  // std::string const G_read = output_filename(outputdir +inSkyName+"_GFull.txt");
  // std::string const dirty_image_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_dirty.fits");  
  // std::string const checkSparse_dirty_image_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_dirty_Greloaded_Sparse.fits");  
  std::string const ResultsLog = output_filename(outputdir+inSkyName+"G"+eg+"Wfrac"+wfrac+"results.txt");

  
  Vector<t_real> SNR(runi);
  Vector<t_real> solverTime(runi);
  
  const t_real C = 299792458;
  const t_real freq0 = 800e6;
  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_real  arcsecTrad = purify_pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
  const t_int wavelet_level =4; // max level for the DB decomposition considred for now

  bool use_w_term = true; // correct for w
  utilities::vis_params uv_data;

  /* Gridding parameters */
  std::string gridKernel = "kb"; // choice of the gridding kernel
  t_real overSample = 2; // oversampling ratio in Fourier due to gridding
  t_int gridSizeU = 4;
  t_int gridSizeV = 4;

  /* Read sky model image & dimensions */
  auto skyOr = pfitsio::read2d(fitsfile);
  t_real const maxsky = skyOr.array().abs().maxCoeff();
  skyOr = skyOr / maxsky;
  t_int heightOr = skyOr.rows();
  t_int widthOr = skyOr.cols();
  std::cout<< "\n"<<  "Original test image  of size " << heightOr<< " x "<<widthOr<< " pixels. \n";
  std::cout<<  "\nINFO: Original test image  is normalized to 1 \n";

  /* Read visibilities or  uvw coverages from  files */
  bool read_w = true; // read w from uv coverage
  uv_data = utilities::read_uvw_sim( vis_file, lambda, read_w); // uvw are  from now  on in units of lambda
  const Vector<t_real> & u = uv_data.u.cwiseAbs();
  const Vector<t_real> & v = uv_data.v.cwiseAbs();
  const Vector<t_real> & w = uv_data.w.cwiseAbs();

  t_real maxBaseline = lambda *(((u.array() * u.array() + v.array() * v.array() + w.array() * w.array()).sqrt()).maxCoeff()) ;
  t_real maxProjectedBaseline = lambda *(((u.array() * u.array() + v.array() * v.array()).sqrt()).maxCoeff()) ;
  t_real thetaResolution = 1.22* lambda / maxProjectedBaseline ;
  t_real cellsize = (thetaResolution / arcsecTrad ) / 2;  // Nyquist sampling

  /* Field of view on L & M axis */
  const t_real theta_FoV_L = cellsize * widthOr * arcsecTrad; 
  const t_real theta_FoV_M = cellsize * heightOr * arcsecTrad;
  std::cout << "freq0 "<< freq0 <<", max baseline " << maxBaseline << " m,  angular resolution " << thetaResolution << " rad, cellsize "
   << cellsize <<  " arcsec, "<< "FoV " <<theta_FoV_L  <<" rad. \n";
  /* Set w terms */
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
       

  std::cout <<"wmax/wlimit = " << w_max/wlimit <<", wfrac:" << inWFRAC <<".\n" ;
  std::cout <<"\n";

  /* Upsampling in Fourier due to w component - image size setting - keep even size for now */
  t_int multipleof =pow(2,wavelet_level)/overSample; // size should be multiple of 2^level (level of DB wavelets)
  t_real upsampleRatio = utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr,multipleof);

  auto skypaddedFFT = utilities::re_sample_ft_grid_sim(fftop.forward(skyOr), upsampleRatio, multipleof);
  auto sky = fftop.inverse(skypaddedFFT);
  t_real const maxskyUp = sky.array().abs().maxCoeff();
  sky = sky / maxskyUp;

  t_int heightUp = sky.rows();
  t_int widthUp = sky.cols();

  upsampleRatio = (heightUp * purify_pi)/ (heightOr  * purify_pi);
  cellsize = cellsize/upsampleRatio;
  if (upsampleRatio >1)
  std::cout <<  "Upsampling: cellsize " << cellsize <<  "arcsec, upsampling ratio: "<< upsampleRatio << " New Image size " << heightUp<< " x "<<widthUp<< " pixels. \n";
 
  /* Measurement operator parameters */
  std::string weighting_type = "none";
  t_real RobustW = 0; // robust weighting parameter
  uv_data = utilities::set_cell_size_sim(uv_data, upsampleRatio*maxProjectedBaseline /lambda,upsampleRatio*maxProjectedBaseline / lambda); // scale uv coordinates to correct pixel size and to units of 2pi
  uv_data = utilities::uv_scale(uv_data, floor(widthUp * overSample), floor(heightUp * overSample)); // scale uv coordinates to units of Fourier grid size
  
  /* nber of pixels to upsample wrt to each w term - to be saved in a file*/
  Vector<t_real> uvdist = (u.array() * u.array() + v.array() * v.array()).sqrt();
  t_real Blength = 2 * uvdist.maxCoeff();
  Vector<t_real> w_kernel_size = (widthOr *uv_data.w * L / Blength);
  std::string const fileWSize = output_filename(outputdir+"WSizeOr.txt");
  std::ofstream WSize;
  WSize.open(fileWSize, std::ios::app);
  // WSize<<"#number of pixels for each w kernel containing most of the energy";
  for (t_int m = 0; m < uv_data.u.size(); ++m){
       WSize<< floor(std::abs(w_kernel_size(m))) <<"\n";
  }
  WSize.close();

  /*--------------------------------------------------------------------------*/
  std::time_t start_t = std::time(NULL);
  std::cout << "Setting measurement operator .. \n";
  MeasurementOperator SimMeasurements(uv_data, 
                                      gridSizeU, gridSizeV,gridKernel, 
                                      widthUp, heightUp,
                                      overSample,
                                      cellsize, cellsize,
                                      weighting_type, RobustW,
                                      use_w_term,energyC, 1,"none", false); // Create Measurement Operator

  uv_data.vis = SimMeasurements.degrid(sky);
  Image<t_real> skyKb = (SimMeasurements.grid(uv_data.vis)).real();
  std::cout << "writing Dirty image to fits file \n";
  pfitsio::write2d(skyKb.real(), dirty_image_fits);
  std::string const sky_model_fits = output_filename(outputdir +inSkyName+"_upsampled_sky.fits");
  pfitsio::write2d(sky.real(), sky_model_fits);

  std::cout << "\nINFO: Time to generate the data with w term:  " << std::difftime(std::time(NULL), start_t) << "s\n"  ;
  std::cout << "\nINFO: The chirpmatrix is no longer required --> it is now set to 0 \n";
  t_real sparsityC = utilities::sparsity_im(SimMeasurements.C);

  SimMeasurements.C.resize(0,0);
  std::cout << "\nINFO: by default sparsify G by loosing "<< 1-energyG << " of the total energy on each row \n";
  
  /* Sparsify G matrix */
  SimMeasurements.G = utilities::sparsify_rows_matrix(SimMeasurements.G,energyG);
  t_real sparsityG = utilities::sparsity_sp(SimMeasurements.G);

  /* Saving data  */
  std::cout<<"\nSaving the following params in txt files: ";
  
  std::cout << "the new w values ";
  std::string const wFile= output_filename(outputdir +"wterms.txt");
  std::ofstream wVal;
  wVal.open(wFile);
  for (int i = 0; i < uv_data.u.size(); ++i){
       wVal<<uv_data.w(i)<<"\n";
  }
  wVal.close();
  //l0 norm of Gmat rows
  Vector<t_int> const  l0RowsGmat = utilities::l0_row_matrix(SimMeasurements.G); // l0 norm of each row of G
  std::cout << "L0 norm of G mat. \n\n";
  std::string const spRowGFile= output_filename(outputdir +"l0RowsGfull.txt");
  std::ofstream spRowG;
  spRowG.open (spRowGFile);
  spRowG << "# L0 norm of the Gmatrix rows - total Npix per row = "<<SimMeasurements.G.cols() <<"\n";
  spRowG <<l0RowsGmat;
  spRowG.close();
  //test specifications
  std::string const fileSpec = output_filename(outputdir+inSkyName+"G"+eg+"Wfrac"+wfrac+"_TestSpec.txt");
  std::ofstream TestSpec;
  TestSpec.open(fileSpec);
  TestSpec << "sky "<<inSkyName <<"\nNxo "<<heightOr<<"\nNyo "<<widthOr<<"\nvis " <<inVisName <<"\nNvis "<<uv_data.u.size();
  TestSpec <<"\nFoV "<<L <<"\nFreq0 "<<freq0<<"\nWfrac "<<wfrac<<"\nUpRatio "<<upsampleRatio<<"\nOvFactor "<<overSample;
  TestSpec<<"\nNxtot "<<floor(heightUp * overSample)<<"\nNytot "<<floor(widthUp * overSample);
  TestSpec <<"\nenergyC "<<1-energyC<<"\nSparsityC "<<sparsityC <<"\nenergyG "<<1-energyG<<"\nSparsityG "<<sparsityG <<"\nnormG "<<SimMeasurements.norm;//<<"\nSparsityC "<<sparsityC;
  TestSpec.close();

  std::cout<<"\nData is generated successfuly! Starting reconstruction ";

 /*--------------------------------------------------------------------------*/
  // SOPT 
  /*--------------------------------------------------------------------------*/
  sky=skyOr;
  /* SARA Dictionairies */
    sopt::wavelets::SARA const sara{std::make_tuple("dirac", 3u),std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u), std::make_tuple("DB3", 3u), std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u), std::make_tuple("DB6", 3u), std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
    auto const Psi = sopt::linear_transform<t_complex>(sara, heightUp,widthUp);
 
  /* Operators Forward & Backword  */    
    auto direct = [&SimMeasurements, &sky](Vector<t_complex> &out, Vector<t_complex> const &x) {
          assert(x.size() == sky.size());
          auto const image = Image<t_complex>::Map(x.data(), sky.rows(), sky.cols());
          out =   SimMeasurements.degrid(image);
    };
    auto adjoint = [&SimMeasurements, &sky](Vector<t_complex> &out, Vector<t_complex> const &x) {
          auto image = Image<t_complex>::Map(out.data(), sky.rows(), sky.cols());
          image = SimMeasurements.grid(x);
    };
    auto measurements_transform = sopt::linear_transform<Vector<t_complex>>(
      direct, {0, 1, static_cast<t_int>(uv_data.vis.size())},
      adjoint, {0, 1, static_cast<t_int>(sky.size())}
    );

    Vector<t_complex> const y0 = (measurements_transform * Vector<t_complex>::Map(sky.data(), sky.size()));    
    t_real sigma = y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(inputSNR / 20.0));
    auto const epsilon =  std::sqrt(y0.size() + 2 * std::sqrt(y0.size())) * sigma;
    auto const inputData0 = utilities::add_noise(y0, 0., sigma);

    Vector<> dimage = (measurements_transform.adjoint() * inputData0).real();//dirty image
    Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());
   
  /* Reconstruction */
    std::cout << "Starting SOPT!" << '\n';
    #pragma omp parallel for
    for (t_int i = 0; i < runi; ++i){
        std::time_t startT = std::time(NULL);
        Vector<t_complex>  const inputData = utilities::add_noise(y0, 0., sigma);//dirty(y0, mersenne, inputSNR);//  noisy vis
        std::cout<<inputData.size();
 
        std::string const model_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_model"+std::to_string(i)+".fits");
        std::string const residual_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_residual"+std::to_string(i)+".fits");
       
        /* SOLVER: Proximal ADMM */
        std::cout<<" --------------  Proximal - ADMM : RUN " << i << " ----------------- \n";
        auto const padmm = sopt::algorithm::L1ProximalADMM<t_complex>(inputData)
                         .itermax(1500)
                         .gamma((measurements_transform.adjoint() * inputData).real().maxCoeff() * 1e-3)
                         .relative_variation(1e-5)
                         .l2ball_proximal_epsilon(epsilon)
                         .tight_frame(false)
                         .l1_proximal_tolerance(1e-2)
                         .l1_proximal_nu(1e0)
                         .l1_proximal_itermax(50)
                         .l1_proximal_positivity_constraint(true)
                         .l1_proximal_real_constraint(true)
                         .residual_convergence(epsilon * 1.001)
                         .lagrange_update_scale(0.9)//was 0.9
                         .nu(1e0)
                         .Psi(Psi)
                         .Phi(measurements_transform);
          auto const result = padmm(initial_estimate);
          assert(result.x.size() == sky.size());
          Image<t_complex> EModel = Image<t_complex>::Map(result.x.data(), sky.rows(), sky.cols());
          pfitsio::write2d(EModel.real(), model_fits);

          auto vis = SimMeasurements.degrid(EModel);
          auto residual = SimMeasurements.grid(inputData - vis);
          pfitsio::write2d(residual.real(), residual_fits);
          SNR(i) = utilities::snr_(sky.real(),EModel.real());
          solverTime(i) = std::difftime(std::time(NULL), startT) ;//((endT-startT)/double(CLOCKS_PER_SEC));
          std::cout << "SNR  " <<SNR(i) << " \nCompt time for PADMM " << solverTime(i) <<  "\n" ;

    }
     

    /* SOLVER: SDMM

    std::cout << "Starting SOPT!" << '\n';
    #pragma omp parallel for
    for (t_int i = 0; i < runi; ++i){
         // int startT=clock();
          std::time_t startT = std::time(NULL);

         // std::mt19937_64 noise;
          // auto  seed = std::random_device();// std::time(0);
          // std::srand((unsigned int)seed);
          // std::mt19937_64 mersenne(seed);

          std::random_device device;
          std::mt19937_64 mersenne(device());
                   // std::mt19937_64 noise;

         auto const inputData = dirty(y0, mersenne, inputSNR);//  noisy vis

         std::cout<<" -------------- SDMM RUN " << i << " ----------------- \n";
 
         std::string const model_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_model"+std::to_string(i)+".fits");
         std::string const residual_fits = output_filename(outputdir +inSkyName+"G"+eg+"_sara_residual"+std::to_string(i)+".fits");
         auto const sdmm
            = sopt::algorithm::SDMM<t_complex>()
              .itermax(1000)
              .gamma((measurements_transform.adjoint() * inputData).real().maxCoeff() * 1e-2)
              .is_converged(sopt::RelativeVariation<t_complex>(5e-4))
              .conjugate_gradient(100, 1e-3)
              .append(sopt::proximal::translate(sopt::proximal::L2Ball<t_complex>(epsilon), -inputData),
                      measurements_transform)
              .append(sopt::proximal::l1_norm<t_complex>, Psi.adjoint(), Psi)
              .append(sopt::proximal::positive_quadrant<t_complex>);

         auto const result = sdmm(initial_estimate);
         assert(result.out.size() == sky.size());
         Image<t_complex> EModel = Image<t_complex>::Map(result.out.data(), sky.rows(), sky.cols());
         pfitsio::write2d(EModel.real(), model_fits);
         auto vis = SimMeasurements.degrid(EModel);
         auto residual = SimMeasurements.grid(inputData - vis);

         pfitsio::write2d(residual.real(), residual_fits);
         // int endT=clock();
         SNR(i) = utilities::snr_(sky.real(),EModel.real());
         solverTime(i) = std::difftime(std::time(NULL), startT) ;//((endT-startT)/double(CLOCKS_PER_SEC));
         std::cout << "SNR  " <<SNR(i) << " \nCompt time for solver " << solverTime(i) <<  "\n" ;
    }
    */


  /* Saving Results */

  // Test specifications
  std::string const fileSpecRes = output_filename(outputdir+inSkyName+"G"+eg+"Wfrac"+wfrac+"TestSpecRes.txt");
  std::ofstream TestSpecRes;
  TestSpecRes.open(fileSpecRes);
  TestSpecRes << "#sky Nxo Nyo vis Nvis lambda Wfrac UpRatio OvFactor Nxtot Nytot energyC energyG sparsityG normG\n ";
  TestSpecRes <<inSkyName<<" "<<heightOr<<" "<<widthOr<<" "<<inVisName<<" "<<uv_data.u.size()<<" "<<lambda<<" "<<upsampleRatio<<" "<<overSample//
  <<" "<<floor(heightUp * overSample)<<" "<<floor(widthUp * overSample)<<" "<<1-energyC<<" "<<1-energyG<<" "<<SimMeasurements.norm;
  TestSpecRes << "\n#################################\n#L0 norm of each row of the sparse G matrix \n"<<l0RowsGmat;
  TestSpecRes.close();

  //SNR
  std::string const fileSNRLog = output_filename(outputdir+inSkyName+"Wfrac"+wfrac+"SNR.txt");
  std::ofstream SNRLog;
  SNRLog.open(fileSNRLog, std::ios::app);
  SNRLog<<"#EG  SNR\n";
  for (t_int i = 0; i < runi; ++i){
       SNRLog<<1-energyG<<" "<< SNR(i) << "\n";
  }
  SNRLog.close();
  // Solver compt. time
  std::string const fileSolverTLog = output_filename(outputdir+inSkyName+"Wfrac"+wfrac+"CTime.txt");
  std::ofstream CTLog;
  CTLog.open(fileSolverTLog, std::ios::app);
  CTLog<<"#EG  CTime\n";
  for (t_int i = 0; i < runi; ++i){
       CTLog<<1-energyG<<" "<< solverTime(i) << "\n";  // CTLog<<<< endl;td::fixed <<std::setprecision(10)<<

  }
  CTLog.close();
  // G - sparsity
  std::string const fileGspLog = output_filename(outputdir+inSkyName+"Wfrac"+wfrac+"Gsp.txt");
  std::ofstream GSLog;
  GSLog.open(fileGspLog, std::ios::app);
  GSLog<<"#EG  SparsityG\n";//std::endl;
  GSLog<<1-energyG << " " <<sparsityG<< std::endl;
  GSLog.close();



  return 0;    



}



