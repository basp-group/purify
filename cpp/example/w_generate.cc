//example of data with w effect - output of convolution is tested in MATLAB 
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <stdio.h>
#include "purify/config.h"
#include <array>
#include <ctime>
#include <memory>
#include <random>
#include <boost/math/special_functions/erf.hpp>
#include "purify/MeasurementOperator.h"
#include "purify/directories.h"
#include "purify/logging.h"
#include "purify/pfitsio.h"
#include "purify/types.h"
#include "purify/utilities.h"



int main( int nargs, char const** args ){
  using namespace purify;
  using namespace purify::notinstalled;
  


 /* Inputs of the program */
  std::string inSkyName=args[1];
  std::string inVisName=args[2];
  std::string outputdir=args[3];
  std::string wfrac =args[4];
  t_real inWFRAC=std::stod(args[4]);


  std::cout<< "**--------------------------------------------------------------**\n";
  std::cout<< "      generating random w components\n";
  std::cout<< "      Sky Model image: "<<inSkyName<<"\n";
  std::cout<< "      UVW coverage: "<<inVisName<<"\n";
  std::cout<< "      WRAC " << wfrac <<"\n";
  std::cout<< "      Result saved in : "<<outputdir<<"\n";
  std::cout<< "**---------------------------------------------------------------**\n";

  //input files
  std::string const fitsfile = image_filename(inSkyName+".fits");
  std::cout<<fitsfile<<"\n";
  std::string const vis_file = inVisName;
  
  //consts
  // std::cout<<"\nIt s a minor issue - but just make sure to update the freq of the uv coverage adopted \n";
  const t_real C = constant::c;
  const t_real freq0 = 1000e6;
  const t_real lambda = C/freq0; // wavelength 21 cm 
  const t_real  arcsecTrad = constant::pi / (180 * 60 *60) ; //factor of conversion from arcsec to radian
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
  t_int heightOr = skyOr.rows();
  t_int widthOr = skyOr.cols();
  std::cout<< "\n"<<  "Original test image  of size " << heightOr<< " x "<<widthOr<< " pixels. \n";
  // Read visibilities | uvw coverages from files 
  bool read_w = true; // read w from uv coverage
  uv_data = utilities::read_uvw(vis_file,lambda);//_sim( vis_file, lambda, read_w); // uvw are  from now  on in units of lambda
  std::cout<< "\n Vis file  read"<<" \n";
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
  
  std::string const fileWvalues = output_filename(outputdir+"w_orig.txt");
  std::ofstream Wvals;
  std::cout<<fileWvalues<<"\n";
  Wvals.open(fileWvalues, std::ios::app);
  for (t_int t = 0; t < uv_data.u.size(); ++t){
       Wvals<< uv_data.w(t)<< "\n";
  }
  Wvals.close();
   
  std::string const fileUvalues = output_filename(outputdir+"u_orig.txt");
  std::ofstream Uvals;
  std::cout<<fileUvalues<<"\n";
  Uvals.open(fileUvalues, std::ios::app);
  for (t_int t = 0; t < uv_data.u.size(); ++t){
       Uvals<< uv_data.u(t)<< "\n";
  }
  Uvals.close();
   

  std::string const fileVvalues = output_filename(outputdir+"v_orig.txt");
  std::ofstream Vvals;
  std::cout<<fileVvalues<<"\n";
  Vvals.open(fileVvalues, std::ios::app);
  for (t_int t = 0; t < uv_data.u.size(); ++t){
       Vvals<< uv_data.v(t)<< "\n";
  }
  Vvals.close();
   
  if (inWFRAC >0){
      std::cout<< "Generating random  w components.. ";
      std::random_device rand;
      std::mt19937 gen(rand());
      std::uniform_real_distribution<t_real> dist(-1, 1);
      auto generator = std::bind(dist, gen);
      for (int i = 0; i < u.size(); ++i){
           uv_data.w(i) = inWFRAC*wlimit*dist(gen);
      }
      w_max = (uv_data.w.cwiseAbs()).maxCoeff();
      
  } 
  else{
    if (inWFRAC==-1)   std::cout << "Keeping  original w components ";
  }     


 std::string const fileWvaluesg = output_filename(outputdir+"w_GEN.txt");
 std::ofstream Wvalsg;
 std::cout<<fileWvaluesg<<"\n";
 Wvalsg.open(fileWvaluesg, std::ios::app);
 for (t_int t = 0; t < uv_data.u.size(); ++t){
       Wvalsg<< uv_data.w(t)<< "\n";
  }
  Wvalsg.close();
// std::string const fileSNRLog = output_filename(outputdir+inSkyName+"W."+wfrac+"_SNR.txt");
//           std::ofstream SNRLog;
      
//           SNRLog.open(fileSNRLog, std::ios::app);
//             for (t_int i = 0; i < runi; ++i){
//                 SNRLog<<eg<<" "<< SNR(i) << "\n";
//             }
//           SNRLog.close();
 //std::string const fileGspLog = output_filename(outputdir+inSkyName+".W"+wfrac+"_Gsp.txt");
 //          std::ofstream GSLog;   
 //          GSLog.open(fileGspLog, std::ios::app);
 //          GSLog<<eg << " " <<sparsityG<< std::endl;
 //          GSLog.close();
  std::cout<<"bon! \n\n";

  return 0;    



}



