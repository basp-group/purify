#include "purify/config.h"
#include "purify/logging.h"
#include "purify/utilities.h"
#include "purify/wproj_utilities.h"
#include "purify/MeasurementOperator.h"
#include "purify/FFTOperator.h"
#include <Eigen/Sparse>
#include <time.h>
#include <omp.h>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_AUTOMATIC_RESIZING
namespace purify {
  namespace wproj_utilities {
  
  t_real pi = constant::pi;
  
  Matrix<t_complex> generate_chirp(const t_real & w_rate, const t_real &cell_x, const t_real & cell_y, const t_int & x_size, const t_int & y_size){
      
        // return chirp image fourier transform for w component
        // w:: w-term in units of lambda
        // celly:: size of y pixel in arcseconds
        // cellx:: size of x pixel in arcseconds
        // x_size:: number of pixels along x-axis
        // y_size:: number of pixels along y-axis
      
          const t_real theta_FoV_L = cell_x * x_size;
          const t_real theta_FoV_M = cell_y * y_size;

          const t_real L = 2 * std::sin(pi / 180.* theta_FoV_L / (60. * 60.) * 0.5);
          const t_real M = 2 * std::sin(pi / 180.* theta_FoV_M / (60. * 60.) * 0.5);

          const t_real delt_x = L / x_size;
          const t_real delt_y = M / y_size;
          Image<t_complex> chirp = Image<t_complex>::Zero(y_size, x_size);
          t_complex I(0, 1);
          t_real nz=((t_real) y_size*x_size);

          
          #pragma omp parallel for collapse(2)     
          for (t_int l = 0; l < x_size; ++l){
              for (t_int m = 0; m < y_size; ++m) {
                  t_real x = (l + 0.5 - x_size * 0.5) * delt_x;
                  t_real y = (m + 0.5 - y_size * 0.5) * delt_y;
                  t_complex val =  std::exp(- 2 * pi * I * w_rate * (std::sqrt(1 - x*x - y*y) - 1)) * std::exp(- 2 * pi * I * (l * 0.5 + m * 0.5))/nz;                
                  if (std::abs(val) >1e-14){
                    chirp(m, l) = val;
                  }
              }
          }
          return chirp; 
  }    

  Eigen::SparseVector<t_complex> create_chirp_row(const t_real & w_rate, const t_real &cell_x, const t_real & cell_y,const t_real & ftsizev, const t_real & ftsizeu, const t_real& energy_fraction){
    t_int fft_flag = (FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    auto fftop_ = purify::FFTOperator().fftw_flag(fft_flag);
    const t_int Npix = ftsizeu * ftsizev;
    auto chirp_image = wproj_utilities::generate_chirp(w_rate, cell_x, cell_y, ftsizeu, ftsizev); 
    Matrix<t_complex> rowC;
    #pragma omp critical (fft)
    rowC = fftop_.forward(chirp_image);
    rowC.resize(Npix,1);
    Eigen::SparseVector<t_complex> chirp_row(ftsizeu*ftsizev);
    chirp_row = sparsify_row_values(rowC, energy_fraction);
  
    return chirp_row;
  }

  Eigen::SparseVector<t_complex> row_wise_convolution(const Eigen::SparseVector<t_complex> &Grid,  Eigen::SparseVector<t_complex>& Chirp, const t_int& Nx, const t_int& Ny){
      
      Eigen::SparseVector<t_complex> output_row(Nx*Ny);
      t_int row_support = 0 ;    
      const t_int nx2 = Nx/2;
      const t_int ny2 = Ny/2;
      // PURIFY_DEBUG("\n Inside CONV ");

      // #pragma omp parallel for collapse(2) shared(Grid)
        for(t_int i = 0; i < Nx; ++i){     
          for(t_int j = 0; j < Ny; ++j){ 
        
            t_complex temp (0.0,0.0);
             
            for (Eigen::SparseVector<t_complex>::InnerIterator pix(Grid); pix; ++pix){                   
              Vector<t_int> image_row_col = utilities::ind2sub(pix.index(), Nx, Ny);  
              t_int ii = image_row_col(0); 
              t_int jj = image_row_col(1);
   
              t_int  i_fftshift, j_fftshift  ;   
              if(ii <  nx2)   i_fftshift = ii + nx2; 
              else{   i_fftshift = ii - nx2;  }
              if(jj <  ny2)   j_fftshift = jj + ny2;
              else{   j_fftshift = jj - ny2;  }         
                
              t_int  oldpixi = nx2 - i + i_fftshift;                 
              if (oldpixi >= 0 and oldpixi < Nx){ 
                t_int  oldpixj = ny2 - j + j_fftshift; 
                if (oldpixj >= 0 and oldpixj < Ny){
                t_int chirp_pos  =  oldpixi * Ny + oldpixj   ;                             
                  t_complex val = pix.value() * Chirp.coeffRef(chirp_pos);              
                  if (std::abs(val) > 1e-14)
                    temp +=  val;                   
                } 
              }             
            }

            if(std::abs(temp) > 1e-14){
              t_int iii,jjj;
              if(i >= nx2)   iii = i - nx2;  else{   iii = i + nx2;   }
              if(j >= ny2)   jjj = j - ny2;  else{   jjj = j + ny2;   } 
              t_int pos = utilities::sub2ind(iii,jjj,Nx,Ny); 
              output_row.insert(pos)= temp; 
         
              // std::cout<<"<<<<< CONV: //"<<pos<<"// "<<std::abs(temp)<<"// -->\n";fflush(stdout); 
            }            
          }  
        }
         std::cout<<"<<<<< CONV: //"<<output_row.nonZeros()<<"// -->\n";fflush(stdout); 
        return output_row;  
  }   
  
  Sparse<t_complex> wprojection_matrix(const Sparse<t_complex> &Grid, const t_int& Nx, const t_int& Ny,const Vector<t_real> & w_components, const t_real &cell_x, const t_real &cell_y, const t_real& energy_fraction_chirp,const t_real& energy_fraction_wproj){
           
        typedef Eigen::Triplet<t_complex> T;
        std::vector<T> tripletList;
        t_int Npix = Nx*Ny;
        t_int Nvis = w_components.size();

        t_real chirp_en = 1;
        t_real wproj_en = 1;

        if (energy_fraction_chirp <1) PURIFY_HIGH_LOG("Hard-thresholding of the Chirp kernels ");
        if (energy_fraction_wproj <1) PURIFY_HIGH_LOG("Hard-thresholding of the rows of G ");
        
        tripletList.reserve(floor(Npix*Nvis*0.2)); 
       // #pragma omp parallel for 
        for(t_int m = 0;  m < Grid.outerSize(); ++m){ 
            PURIFY_HIGH_LOG("\nCURRENT WPROJ - Kernel index [{}]",m);
            Eigen::SparseVector<t_complex> chirp(Npix);
            chirp =  create_chirp_row(w_components(m),cell_x, cell_y, Nx, Ny,energy_fraction_chirp);            
            t_int chirp_size = chirp.nonZeros();
            PURIFY_DEBUG("Number of nonzeros entries in CHIRP :[{}]",chirp_size);
            Eigen::SparseVector<t_complex> G_bis(Npix);  
            for (Sparse<t_complex>::InnerIterator pix(Grid,m); pix; ++pix){
                     G_bis.coeffRef(pix.index()) = pix.value() ;
                     std::cout<<"<<<<< INIT: ||"<<pix.index()<<"|| "<<std::abs(pix.value())<<" || <--\n";fflush(stdout);             
            }
            auto row = row_wise_convolution(G_bis,chirp, Nx,Ny); 

            std::cout<<"\nCONV is DONE with success\n";fflush(stdout);
            Eigen::SparseVector<t_real> absRow = row.cwiseAbs();
            wproj_utilities::sparsify_row_sparse(absRow, energy_fraction_wproj);
            PURIFY_DEBUG("Number of nonzeros entries in sparseRow :[{}]",absRow.nonZeros());
            for (Eigen::SparseVector<t_real>::InnerIterator itr(absRow); itr; ++itr){
               std::cout<<"<<<<< STORE: ||"<<itr.index()<<"|| "<<std::abs(itr.value())<<" || <--\n";fflush(stdout);
              // #pragma omp critical (load1)  
                 tripletList.push_back(T(m,itr.index(),row.coeffRef(itr.index())));  
            }
            // }
        }        
        Sparse<t_complex> Gmat(Nvis, Npix);
        Gmat.setFromTriplets(tripletList.begin(), tripletList.end());
        PURIFY_DEBUG("\n \nBuilding the rows of G.. DONE!\n");
        return Gmat;  
  }   

  void sparsify_row_sparse(Eigen::SparseVector<t_real> &row, const t_real &energy){
         /*
          Takes in a row of G and returns indexes of coeff to keep in the row sparse version 
          energy:: how much energy - in l2 sens - to keep after hard-thresholding 
         */
          
          

          
          if ( energy <1){
            t_real tau = 0.5;
          t_real old_tau = -1;
          t_int niters = 100;
          // const Vector<t_real> abs_row = row;
          const t_real abs_row_total_energy = (row.cwiseProduct(row)).sum();
          t_real min_tau = 0;
          t_real max_tau = 0.5;
          t_int rowLength = row.size();
          t_real abs_row_max = 0;

          for (Eigen::SparseVector<t_real>::InnerIterator itr(row); itr; ++itr){
                      if  (itr.value() > abs_row_max)
                          abs_row_max =itr.value() ;                    
          }
          /* calculating threshold  */
          t_real energy_sum = 0;
          t_real tau__=0;
          for (t_int i = 0; i < niters; ++i){            
              energy_sum = 0;    
              tau__ =   tau *  abs_row_max;    
              // #pragma omp parallel for reduction(+:energy_sum)  
              for (Eigen::SparseVector<t_real>::InnerIterator itr(row); itr; ++itr){
                      if  (itr.value() > tau__)
                          energy_sum +=  itr.value() * itr.value() ;                    
                  }
              energy_sum= energy_sum/abs_row_total_energy;      
              std::cout<<"WPROJ - ROW looping : ["<<energy_sum<<"] tau: ["<< tau__<<"] "<<i<<"\n";fflush(stdout);                                     
              if ( (std::abs(tau - old_tau)/std::abs(old_tau) < 1e-6) and  (energy_sum>=energy)  and (std::abs(energy_sum/energy - 1) <0.001)){
                  // std::cout<<"WPROJ - ROW energy: ["<<energy_sum<<"] tau: ["<< tau<<"] "<<i<<"\n";fflush(stdout);
                 break;        
              }  
              else{
                    old_tau = tau;
                    if (energy_sum > energy)   min_tau = tau; 
                    else{  max_tau = tau; }
                    tau = (max_tau - min_tau) * 0.5 + min_tau;
              }
              if (i == niters-1)                 
                tau = min_tau;           
          }   
          /* performing clipping */ 
          t_real tau_n = tau * abs_row_max;
          row.prune(tau_n,0);
                    std::cout<<"\nWPROJ - ROW energy DONE ["<<tau_n<<"] before exit tau ["<<tau<<"]\n";fflush(stdout);

        }



          
  }
  Eigen::SparseVector<t_complex> sparsify_row_values(const Vector<t_complex>& row, const t_real& energy){
         /*
          Takes in a row of G and returns indexes of coeff to keep in the row sparse version 
          energy:: how much energy - in l2 sens - to keep after hard-thresholding 
         */
          
          t_real tau = 0.5;
          t_real old_tau = -1;
          t_int niters = 100;
          const Vector<t_real> abs_row = row.cwiseAbs();
          const t_real abs_row_max = abs_row.maxCoeff();
          t_real abs_row_total_energy = (abs_row.array() * abs_row.array()).sum();
          t_real min_tau = 0;
          t_real max_tau = 0.5;
          t_int rowLength = row.size();

          Eigen::SparseVector<t_complex> output_row(row.size());
          if ( energy == 1){ 
            #pragma omp parallel for 
            for (t_int i = 0; i < row.size(); ++i){
              if (abs_row(i) >1e-14)
                 output_row.coeffRef(i)=row(i);
            }
            return output_row;
          }  
          /* calculating threshold  */
          t_real energy_sum = 0;
          for (t_int i = 0; i < niters; ++i){            
              energy_sum = 0;    
              t_real tau__ =   tau *  abs_row_max;    
              #pragma omp parallel for reduction(+:energy_sum)   
              for (t_int j = 0; j < row.size(); ++j){
                      if  (abs_row(j) > tau__)
                          energy_sum +=  abs_row(j) * abs_row(j) ;                    
                  }
              energy_sum= energy_sum/abs_row_total_energy;      
              std::cout<<"WPROJ - ROW looping : ["<<energy_sum<<"] tau: ["<< tau__<<"] "<<i<<"\n";fflush(stdout);                                     
              if ( (std::abs(tau - old_tau)/std::abs(old_tau) < 1e-6) and  (energy_sum>=energy)  and (std::abs(energy_sum/energy - 1) <0.001)){
                  // std::cout<<"WPROJ - ROW energy: ["<<energy_sum<<"] tau: ["<< tau<<"] "<<i<<"\n";fflush(stdout);
                 break;        

              }  
              else{
                    old_tau = tau;
                    if (energy_sum > energy)   min_tau = tau; 
                    else{  max_tau = tau; }
                    tau = (max_tau - min_tau) * 0.5 + min_tau;
              }
              if (i == niters-1)                 
                tau = min_tau;           
          }   
          /* performing clipping */ 
          // std::cout<<"\nWPROJ - ROW energy DONE \n";fflush(stdout);
          t_real tau_n = std::max(tau * abs_row_max,1e-14);
          // #pragma omp parallel for     
          for (t_int j = 0; j < rowLength; ++j){
            if (abs_row(j)> tau_n){ 
              std::cout<<"WPROJ - ROW value: ["<<std::abs(row(j))<<"] tau: ["<< tau_n<<"] "<<j<<"\n";fflush(stdout);    
              output_row.coeffRef(j)=row(j);                  
            }                        
          }
          std::cout<<"\nWPROJ - ROW energy DONE  before exit tau ["<<tau_n<<"]\n";fflush(stdout);

          return output_row;
  }
           
  Eigen::SparseVector<t_int> sparsify_row_index(Vector<t_complex>& row, const t_real& energy){
         /*
          Takes in a row of G and returns indexes of coeff to keep in the row sparse version 
          energy:: how much energy - in l2 sens - to keep after hard-thresholding 
         */
          
          t_real tau = 0.5;
          t_real old_tau = -1;
          t_int niters = 200;

          Vector<t_real> abs_row = row.cwiseAbs();
          t_real abs_row_max = abs_row.maxCoeff();
          t_real abs_row_total_energy = (abs_row.array() * abs_row.array()).sum();
          t_real min_tau = 0;
          t_real max_tau = 0.5;
          t_int rowLength = abs_row.size();
          Eigen::SparseVector<t_int> output_row(abs_row.size());
          if ( energy == 1){ 
            PURIFY_DEBUG("row energy 1");
            for (t_int i = 0; i < rowLength; ++i){
                if (std::abs(row(i)) >1e-14)
                  output_row.coeffRef(i)=i;
            }          
            return output_row;
          }  
          /* calculating threshold  */
          t_real energy_sum = 0;
          for (t_int i = 0; i < niters; ++i){            
              energy_sum = 0;           
              #pragma omp parallel for  reduction(+:energy_sum)   
                  for (t_int i = 0; i < rowLength; ++i){
                      if  (abs_row(i)/abs_row_max > tau)
                          energy_sum +=  abs_row(i) * abs_row(i) ;                    
                  }
              energy_sum= energy_sum/abs_row_total_energy;                                           
              if ( (std::abs(tau - old_tau)/std::abs(old_tau) < 1e-3) and  (energy_sum>=energy)  and (std::abs(energy_sum/energy - 1) <0.001)){
                  PURIFY_DEBUG("CHIRP - ROW Energy:{}",energy_sum);
                 break;        
              }  
              else{
                    old_tau = tau;
                    if (energy_sum > energy)   min_tau = tau; 
                    else{  max_tau = tau; }
                    tau = (max_tau - min_tau) * 0.5 + min_tau;
              }
              if (i == niters-1)                 
                tau = min_tau;           
          }
         
          t_int sf =0;
          t_real valmax =0;
          /* performing clipping */ 
          #pragma omp parallel for          
          for (t_int i = 0; i < rowLength; ++i){
            if (abs_row(i)/abs_row_max > tau){
            
              // #pragma omp atomic
                // sf++;
                output_row.coeffRef(i)=i;

            }                        
          }
          PURIFY_DEBUG("\nRow after hard-thresholding: SUPPORT:{}",sf); 
          return output_row;
  }
    
  t_real sparsity_sp(const Sparse<t_complex> & Gmat){

      const t_int Nvis = Gmat.rows();
      const t_int Npix = Gmat.cols();
      t_real val = Gmat.nonZeros()/(t_real)(Nvis * Npix);
    
      PURIFY_DEBUG(" Sparsity perc:  {}", val);
      return val; 
  }

  t_real sparsity_im(const Image<t_complex> & Cmat){
      /*
              returs  nber on non zero elts/ total nber of elts in Image   
      */
      const t_int Nvis = Cmat.rows();
      const t_int Npix = Cmat.cols();
      t_real sparsity=0; 
      for(t_int m = 0; m < Nvis; m++){
        for (t_int n = 0; n < Npix; n++){
                if (std::abs(Cmat(m,n)) > 1e-16)
                            sparsity++;
        }
      }
      t_real val =(sparsity/(t_real)(Nvis * Npix)) ;
      PURIFY_DEBUG(" Sparsity perc:  {}", val);
      return val;
  }

  t_real snr_metric(const Image<t_real> &model, const Image<t_real> &solution){
      /*
        Returns SNR of the estimated model image 
      */
        t_real nm= model.matrix().norm();
        t_real ndiff = (model - solution).matrix().norm();
        t_real val = 20 * std::log10(nm/ndiff);
        return val;
  }
  t_real mr_metric(const Image<t_real> &model, const Image<t_real> &solution){
      /*
        Returns SNR of the estimated model image 
      */
        t_int Npix = model.rows() * model.cols();
        Image<t_real>  model_ = model.array()+1e-10;
        Image<t_real>  solution_ = solution.array()+1e-10;
        Image<t_real> model_sol = model_.matrix().cwiseQuotient(solution_.matrix());
        Image<t_real> sol_model = solution_.matrix().cwiseQuotient(model_.matrix());
        Image<t_real> min_ratio = sol_model.matrix().cwiseMin(model_sol.matrix());
        t_real val = min_ratio.array().sum()/Npix;
        return val;
  }

}
}
 