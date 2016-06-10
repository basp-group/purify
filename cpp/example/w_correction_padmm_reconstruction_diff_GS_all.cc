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

  // std::cout<< Eigen::nbThreads();
  using namespace purify;
  using namespace purify::notinstalled;
  sopt::logging::initialize();
  FFTOperator fftop;

  /* Inputs of the program */
  std::cout<<"Reading parameters\n";
  const std::string inSkyName=args[1];
  const std::string inVisName=args[2];

  const std::string egFile = args[3];

  std::string ec = args[4];
  t_real energyC=std::stod(ec);

  const std::string outputdir=args[5];
  t_real inputSNR=atoi(args[6]); 
  t_real inWFRAC=std::stod(args[7]);
  const std::string wfrac =args[7];

  t_real inputFOV=std::stod(args[9]);//NOT USED AT THE MOMENT
  const t_int runi=std::stod(args[8]);
  


  std::cout<< "**--------------------------------------------------------------**\n";
  std::cout<< "      This is an example RI imaging with w correction\n";
  std::cout<< "      Study of G sparsity\n";
  std::cout<< "      Sky Model image: "<<inSkyName<<"\n";
  std::cout<< "      UVW coverage: "<<inVisName<<"\n";
  std::cout<< "      Sparsity on G "<<egFile <<"\n";
  std::cout<< "      Sparsity on C " << energyC <<"\n";
  std::cout<< "      WFRAC " << wfrac <<"\n";
  std::cout<< "      Nber of runs " << runi <<"\n";
  std::cout<< "      FoV (!not used) " << inputFOV <<"\n";
  std::cout<< "      SNR on the data " << inputSNR <<"\n";
  std::cout<< "      Results saved in : "<<outputdir<<"\n";
  std::cout<< "      Reconstruction using PADMM\n";
  std::cout<< "**---------------------------------------------------------------**\n";
  
  /* input files */
  std::cout<<"\nReading image and any available data\n";
  const std::string  fitsfile = image_filename(inSkyName+".fits");
  const std::string  vis_file =  inVisName;//image_filename("coverages/" + inVisName);
  /*Output files & vars*/
  
  const std::string  ResultsLog = output_filename(outputdir+inSkyName+"G.Wfrac"+wfrac+"results.txt");
  Vector<t_real> SNR(runi);
  Vector<t_real> solverTime(runi); 
  

  const t_real C = purify_c;
  const t_real freq0 = 1000e6;
  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_real  arcsecTrad = purify_pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
  const t_int wavelet_level =4; // max level for the DB decomposition considred for now
  bool use_w_term = true; // correct for w
  utilities::vis_params uv_data;
  t_int nbLevelG=1;


  std::vector<t_real> gList;  
  std::string line;
  std::ifstream glevel (egFile);
  t_real val;
  if (glevel.is_open())
  {
    while ( std::getline(glevel,line) )
    {
      val = std::stod(line);
      gList.push_back(val);
    }
    glevel.close();
    // energyG = gList[1];
    // eg=std::to_string(energyG);
    // std::cout<<"nber of levels: "<< gList.size()  <<gList[3]<<"\n";

    nbLevelG =gList.size();
  }
  
  else std::cout << "Unable to open file!!! \n"; 
  

  /*   Gridding parameters */
  const std::string  gridKernel = "kb"; // choice of the gridding kernel
  const t_real overSample = 2; // oversampling ratio in Fourier due to gridding
  const t_int gridSizeU = 4;
  const t_int gridSizeV = 4;

  /* Read sky model image & dimensions */
  Image<t_complex> skyOr = pfitsio::read2d(fitsfile);
  t_real const maxsky = skyOr.array().abs().maxCoeff();
  skyOr = skyOr / maxsky;
  t_int heightOr = skyOr.rows();
  t_int widthOr = skyOr.cols();
  std::cout<< "\n"<<  "Original test image  of size " << heightOr<< " x "<<widthOr<< " pixels. \n";
  std::cout<<  "INFO: Original test image  is normalized to 1 \n";

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

  // FoV on L & M axis 
  const t_real theta_FoV_L = cellsize * widthOr * arcsecTrad; 
  const t_real theta_FoV_M = cellsize * heightOr * arcsecTrad;
  std::cout << "\nObservation Specs: freq0 "<< freq0 <<", max baseline " << maxBaseline << " m,  angular resolution " << thetaResolution << " rad, cellsize "
   << cellsize <<  " arcsec, "<< "FoV " <<theta_FoV_L  <<" rad. \n";

  /* Set w terms */
  const t_real L = 2 * std::sin(theta_FoV_L  * 0.5);
  const t_real M = 2 * std::sin( theta_FoV_M * 0.5);
  const t_real wlimit = widthOr / (L * M); 
  t_real w_max = (uv_data.w.cwiseAbs()).maxCoeff();

  std::cout <<"\nINFO: original w_max " << w_max <<", limits on w: wrt FoV & Npix " << wlimit << ", wrt b_max " << maxBaseline /lambda << ".\n" ;  

  if (inWFRAC >0){
    std::cout<< "\nGenerating random w component.. ";
      std::random_device rand;
      std::mt19937 gen(1);
      std::uniform_real_distribution<t_real> dist(-1, 1);
      auto generator = std::bind(dist, gen);
      for (int i = 0; i < u.size(); ++i){
           uv_data.w(i) = inWFRAC*wlimit*dist(gen);
      }
      w_max = (uv_data.w.cwiseAbs()).maxCoeff();
  } 
  else{
    if (inWFRAC==-1)   std::cout << "\nINFO: Keeping  original w components ";
  }
       
  std::cout <<"  wmax/wlimit = " << w_max/wlimit <<", wfrac:" << inWFRAC <<".\n" ;
  std::cout <<"\n";
   fflush(stdout); 
  /* Upsampling in Fourier due to w component - image size setting - keep even size for now */
  t_int multipleof =pow(2,wavelet_level)/overSample;   // size should be multiple of 2^level (level of DB wavelets)
  t_real upsampleRatio = utilities::upsample_ratio_sim(uv_data, L, M, widthOr, heightOr, multipleof);
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
  uv_data = utilities::set_cell_size_sim(uv_data, upsampleRatio*maxProjectedBaseline /lambda,upsampleRatio*maxProjectedBaseline / lambda); // scale uv coordinates to correct pixel size and to units of 2pi
  uv_data = utilities::uv_scale(uv_data, floor(widthUp * overSample), floor(heightUp * overSample)); // scale uv coordinates to units of Fourier grid size
  
  //nber of pixels to upsample wrt to each w term - to be saved in a file
  Vector<t_real> uvdist = (u.array() * u.array() + v.array() * v.array()).sqrt();
  t_real Blength = 2 * uvdist.maxCoeff();
  Vector<t_real> w_kernel_size = (widthOr *uv_data.w * L / Blength);
  std::string const fileWSize = output_filename(outputdir+"WSizeOr.txt");
  std::ofstream WSize;
  WSize.open(fileWSize, std::ios::app);

  // writing WSize<<"#number of pixels for each w kernel containing most of the energy";
  for (t_int m = 0; m < uv_data.u.size(); ++m){
       WSize<< floor(std::abs(w_kernel_size(m))) <<"\n";
  }
  WSize.close();

  /*--------------------------------------------------------------------------*/
  // Building the data
  /*--------------------------------------------------------------------------*/
  std::cout << "\nSetting measurement operator .. \n";
  fflush(stdout); 
  MeasurementOperator SimMeasurements(uv_data, 
                                      gridSizeU, gridSizeV,gridKernel, 
                                      widthUp, heightUp,
                                      overSample,
                                      cellsize, cellsize,
                                      weighting_type, RobustW,
                                      use_w_term,  energyC, upsampleRatio,"none",false); // Create Measurement Operator

  t_real sparsityGO = utilities::sparsity_sp(SimMeasurements.G);// sparsity of G -nber on non zero elts over of total number of elts
  t_real sparsityC = utilities::sparsity_im(SimMeasurements.C);



  std::cout << "\nINFO: The chirpmatrix is no longer required --> it is now set to 0 \n";
  fflush(stdout); 
  SimMeasurements.C.resize(0,0);
  std::cout << "DEBUG 0BIS!\n";   fflush(stdout);

  //Generating DATA 
  
  Image<t_complex> sky(widthOr, heightOr);
  std::cout << "DEBUG 1BIS!\n";   fflush(stdout);
  sky.real()=skyOr.real();
  std::cout << "DEBUG 2BIS!\n";   fflush(stdout);
  sky.imag()=skyOr.imag()*0;
  std::cout << "DEBUG 2BIS2!\n";   fflush(stdout);
  t_int height = sky.rows();
  std::cout << "DEBUG 3BIS!\n";   fflush(stdout);

  t_int width = sky.cols();
  std::cout << "DEBUG 4BIS!\n";   fflush(stdout);

  std::cout<<"\nGenerating noise free measurement.\n";
  uv_data.vis = SimMeasurements.degrid(sky);
  std::cout << "DEBUG 5BIS!\n";   fflush(stdout);
  std::string const sky_model_fits = output_filename(outputdir +inSkyName+"_GroundTruth.fits");
   std::cout << "DEBUG 6BIS!\n";   fflush(stdout);
  pfitsio::write2d((Image<t_real>)sky.real(), sky_model_fits);
  std::cout<<"\nData is generated successfuly! Now Reconstruction! :) \n ";
  std::cout << "\nINFO: The focus is on sparsifying the G matrix for now. \n";
  fflush(stdout); 
  
  /*--------------------------------------------------------------------------*/
  std::cout<<"Setting operators for SOPT\n";  fflush(stdout); 
  /*--------------------------------------------------------------------------*/
  
  /* SARA Dictionairies */
  std::cout << "DEBUG 1!\n";   fflush(stdout);
    sopt::wavelets::SARA const sara{std::make_tuple("dirac", 3u),std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u), std::make_tuple("DB3", 3u), std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u), std::make_tuple("DB6", 3u), std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
    auto const Psi = sopt::linear_transform<t_complex>(sara, height,width);
 
  /* Data */
  std::cout << "DEBUG 2!\n";   fflush(stdout);
    Vector<t_complex> const y0 = uv_data.vis; //(measurements_transform * Vector<t_complex>::Map(sky.data(), sky.size()));    
    std::cout << "DEBUG 2--BIS0!\n";   fflush(stdout);
    t_real sigma = y0.norm() / std::sqrt(y0.size()) * std::pow(10.0, -(inputSNR / 20.0));//y0.stableNorm()
    std::cout << "DEBUG 2--BIS1!\n";   fflush(stdout);
    t_real const epsilon =  std::sqrt(y0.size() + 2 * std::sqrt(y0.size())) * sigma;
    std::cout << "DEBUG 2--BIS2!\n";   fflush(stdout);
    Vector<t_complex> const inputData0 = utilities::add_noise(y0, 0.0, sigma);

  /*--------------------------------------------------------------------------*/
   std::cout<<"G sparsification tests\n";  fflush(stdout); 
  /*--------------------------------------------------------------------------*/
    uv_data.w=uv_data.w*0;

    #pragma omp parallel num_threads(6)
    {

      #pragma omp for 
      for (t_int CMPT= 0; CMPT < 3; ++CMPT){
      
        t_real energyG = gList[CMPT];
        std::cout<<"\nEnergy G  :   "<<energyG<<"\n";
        fflush(stdout); 
        std::string eg=std::to_string(CMPT);
        std::cout << "DEBUG 3!\n";   fflush(stdout);
        MeasurementOperator SimMeasurementsSP(uv_data, 
                                      gridSizeU, gridSizeV,gridKernel, 
                                      widthUp, heightUp,
                                      overSample,
                                      cellsize, cellsize,
                                      weighting_type, RobustW,
                                      use_w_term,  1, upsampleRatio,"none",false); // Create Measurement Operator
        std::cout << "DEBUG 4!\n";   fflush(stdout);
        SimMeasurementsSP.G = utilities::sparsify_rows_matrix(SimMeasurements.G, energyG);
        std::cout<<"Sparsification Done!\n";
        fflush(stdout); 
        #pragma omp  critical (norm)
        {
          if (inWFRAC != 0){ 
          std::cout << "DEBUG 5!\n";   fflush(stdout);
               fflush(stdout);
          //Restimating the new MO' norm 
            SimMeasurementsSP.norm = 1;
            t_real estimate_eigen_value = 1;

            Image<t_complex> estimate_eigen_vector = Image<t_complex>::Random(heightOr,widthOr); 
            Matrix<t_complex> new_estimate_eigen_vector(heightOr,widthOr);   
            for (t_int i = 0; i < 20; ++i) {
               new_estimate_eigen_vector = (SimMeasurementsSP.grid(SimMeasurementsSP.degrid(estimate_eigen_vector)));
               estimate_eigen_value = new_estimate_eigen_vector.matrix().norm();
               estimate_eigen_vector = (Image<t_complex>) new_estimate_eigen_vector.array()/estimate_eigen_value;
            }
            std::cout << "DEBUG 6bis!\n";   fflush(stdout);
            estimate_eigen_vector.resize(0,0);
            std::cout << "DEBUG 7bis!\n";   fflush(stdout);
            new_estimate_eigen_vector.resize(0,0);
            SimMeasurementsSP.norm =std::sqrt(estimate_eigen_value);
            std::cout << "INFO: Estimated norm of the new operator: " << SimMeasurementsSP.norm << "\n" ; 
            fflush(stdout); 

        }
        

        else{
              SimMeasurementsSP.norm=SimMeasurements.norm;
        }
        }
        std::cout << "DEBUG 8!\n";   fflush(stdout);
        // sparsification  results 
        t_real sparsityG = utilities::sparsity_sp(SimMeasurementsSP.G); // sparsity of G in percentile of total number of elements in G
        std::cout<<"\nSPARSITY OF  G MATRIX: "<< sparsityG<<"\n";
        fflush(stdout); 
        Vector<t_int> const  l0RowsGmat = utilities::l0_row_matrix(SimMeasurementsSP.G); // l0 norm of the  rows of G

        /* Operators Forward & Backword  */    
        auto direct = [&SimMeasurementsSP, &sky](Vector<t_complex> &out, Vector<t_complex> const &x) {
          assert(x.size() == sky.size());
          auto const image = Image<t_complex>::Map(x.data(), sky.rows(), sky.cols());
          out =   SimMeasurementsSP.degrid(image);
        };

        auto adjoint = [&SimMeasurementsSP, &sky](Vector<t_complex> &out, Vector<t_complex> const &x) {  
            auto image = Image<t_complex>::Map(out.data(), sky.rows(), sky.cols());
            image = SimMeasurementsSP.grid(x);
        };
        
        auto measurements_transform = sopt::linear_transform<Vector<t_complex>>(
          direct, {0, 1, static_cast<t_int>(uv_data.vis.size())},
          adjoint, {0, 1, static_cast<t_int>(sky.size())}
        );
   
        /* Reconstruction */
        std::cout << "DEBUG 9!\n";   fflush(stdout);
        Image<t_complex> skyDirty = (SimMeasurementsSP.grid(y0));
        std::cout << "DEBUG 9--BIS0!\n";   fflush(stdout);
        std::string const dirty_image_fits = output_filename(outputdir +inSkyName+".EgIdx"+eg+".G_sara_dirty.fits");  
        std::cout << "DEBUG 9--BIS1!\n";   fflush(stdout);
        pfitsio::write2d((Image<t_real>)skyDirty.real(), dirty_image_fits);
        std::cout << "\n\nStarting SOPT!\n";
       
        Vector<t_real> dimage = (measurements_transform.adjoint() * y0).real();//dirty image 
        Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());
        std::cout << "DEBUG 10!\n";   fflush(stdout);

        // #pragma omp parallel for  
        // //num_threads(5)
        for (t_int i = 0; i < runi; ++i){
            std::cout << "DEBUG 11!\n";  fflush(stdout);
            std::time_t startT = std::time(NULL);
            Vector<t_complex> inputData = utilities::add_noise(y0, 0., sigma);//dirty(y0, mersenne, inputSNR);//  noisy vis
            std::cout<<inputData.size()<<" meas. \n";
            fflush(stdout);
            std::string const model_fits = output_filename(outputdir +inSkyName+"G.EgIdx"+eg+"_sara_model"+std::to_string(i)+".fits");
            std::string const residual_fits = output_filename(outputdir +inSkyName+"G.EgIdx"+eg+"_sara_residual"+std::to_string(i)+".fits");
       
            /* SOLVER: Proximal ADMM */
            std::cout<<" --------------  EG:"<<energyG<<"  Proximal - ADMM : RUN " << i << " ----------------- \n";
             fflush(stdout); 
            auto const padmm = sopt::algorithm::L1ProximalADMM<t_complex>(inputData)
                         .itermax(50)
                         .gamma((measurements_transform.adjoint() * inputData).real().maxCoeff() * 1e-3)
                         .relative_variation(1e-5)
                         .l2ball_proximal_epsilon(epsilon)
                         .tight_frame(false)
                         .l1_proximal_tolerance(1e-2)
                         .l1_proximal_nu(1e0)
                         .l1_proximal_itermax(20)
                         .l1_proximal_positivity_constraint(true)
                         .l1_proximal_real_constraint(true)
                         .residual_convergence(epsilon * 1.01)
                         .lagrange_update_scale(0.9)//was 0.9
                         .nu(1e0)
                         .Psi(Psi)
                         .Phi(measurements_transform);
            auto const result = padmm(initial_estimate);
            assert(result.x.size() == sky.size());
            Image<t_complex> EModel = Image<t_complex>::Map(result.x.data(), sky.rows(), sky.cols());
            pfitsio::write2d((Image<t_real>)EModel.real(), model_fits);

            auto vis = SimMeasurementsSP.degrid(EModel);
            auto residual = SimMeasurementsSP.grid(inputData - vis);
            pfitsio::write2d((Image<t_real>)residual.real(), residual_fits);
            SNR(i) = utilities::snr_(sky.real(),EModel.real());
            solverTime(i) = std::difftime(std::time(NULL), startT) ;//((endT-startT)/double(CLOCKS_PER_SEC));
            std::cout << "\n\n   [Gsparse:"<<energyG<<"    Run:"<<i<<"]    SNR  " <<SNR(i) << " \nCompt time for PADMM " << solverTime(i) <<  "\n\n" ;
            fflush(stdout);
        }
    
        #pragma omp critical (res)
        {
          /* Saving Results */
          // Test specifications

          std::string const fileSpecRes = output_filename(outputdir+inSkyName+"G.EgIdx"+eg+".Wfrac"+wfrac+"TestSpecRes.txt");
          std::ofstream TestSpecRes;
          TestSpecRes.open(fileSpecRes);
          TestSpecRes << "#sky Nxo Nyo vis Nvis lambda Wfrac UpRatio OvFactor Nxtot Nytot energyC energyG sparsityG normG\n ";
          TestSpecRes <<inSkyName<<" "<<heightOr<<" "<<widthOr<<" "<<inVisName<<" "<<uv_data.u.size()<<" "<<lambda<<" "<<upsampleRatio<<" "<<overSample//
          <<" "<<floor(heightUp * overSample)<<" "<<floor(widthUp * overSample)<<" "<<1-energyC<<" "<<1-energyG<<" "<<SimMeasurementsSP.norm;
          TestSpecRes << "\n#################################\n#L0 norm of each row of the sparse G matrix \n"<<l0RowsGmat;
          TestSpecRes.close();
  
          //SNR
          std::string const fileSNRLog = output_filename(outputdir+inSkyName+"W."+wfrac+"_SNR.txt");
          std::ofstream SNRLog;
      
          SNRLog.open(fileSNRLog, std::ios::app);
            for (t_int i = 0; i < runi; ++i){
                SNRLog<<eg<<" "<< SNR(i) << "\n";
            }
          SNRLog.close();
        
          // Solver compt. time
          std::string const fileSolverTLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_CTime.txt");
          std::ofstream CTLog;

          CTLog.open(fileSolverTLog, std::ios::app);
          for (t_int i = 0; i < runi; ++i){
               CTLog<<eg<<" "<< solverTime(i) << "\n";  // CTLog<<<< endl;td::fixed <<std::setprecision(10)<<
          }
          CTLog.close();
        
          // G - sparsity
          std::string const fileGspLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_Gsp.txt");
          std::ofstream GSLog;
     
          GSLog.open(fileGspLog, std::ios::app);
          GSLog<<eg << " " <<sparsityG<< std::endl;
          GSLog.close();
        } 
      }
    }
    
  

  return 0; 

}



