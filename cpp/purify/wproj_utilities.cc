#include "purify/config.h"
#include "purify/logging.h"
#include "purify/utilities.h"
#include "purify/wproj_utilities.h"
#include "purify/MeasurementOperator.h"
#include "purify/FFTOperator.h"
#include <Eigen/Sparse>
#include <time.h>
#include <omp.h>
#include <Unsupported/Eigen/SparseExtra>

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_AUTOMATIC_RESIZING
namespace purify {
  namespace wproj_utilities {
    t_int fft_flag = (FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    auto fftop_ = purify::FFTOperator().fftw_flag(fft_flag);
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

          const t_real delt_x = 1*L / x_size;
          const t_real delt_y = 1*M / y_size;
          Image<t_complex> chirp (y_size, x_size);
          t_complex I(0, 1);
          t_real nz=((t_real) y_size*x_size*1.0);

          #pragma omp parallel for collapse(2)     
          for (t_int l = 0; l < x_size; ++l){
              for (t_int m = 0; m < y_size; ++m) {
                  t_real x = (l + 0.5 - x_size * 0.5) * delt_x;
                  t_real y = (m + 0.5 - y_size * 0.5) * delt_y;
                  t_complex val =  (std::exp(-2 * pi * I * w_rate * (std::sqrt(1-x*x - y*y)-1))) * std::exp(- 2 * pi * I * (l * 0.5 + m * 0.5))/nz;                                 
                  chirp(l,m) = val;
                  
              }
          }

          return chirp; 
  }    

  Eigen::SparseVector<t_complex> create_chirp_row(const t_real & w_rate, const t_real &cell_x, const t_real & cell_y,const t_real & ftsizev, const t_real & ftsizeu, const t_real& energy_fraction){
    // t_int fft_flag = (FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    // auto fftop_ = purify::FFTOperator().fftw_flag(fft_flag);
    const t_int Npix = ftsizeu * ftsizev;
    auto chirp_image = wproj_utilities::generate_chirp(w_rate, cell_x, cell_y, ftsizeu, ftsizev); 
    // saveMarket(chirp_image,"./outputs/chirp_image.txt");
    Matrix<t_complex> rowC;
    #pragma omp critical (fft)
    rowC = fftop_.forward(chirp_image);
    // saveMarket(rowC,"./outputs/chirp_Fourier.txt");
    rowC.resize(Npix,1);
    Vector<t_real> absRow = rowC.cwiseAbs();
    const t_real max_modulus_chirp  = absRow.maxCoeff();
    Eigen::SparseVector<t_real> rowSparse=absRow.sparseView(1e-14,1);
    sparsify_row_sparse_dense(rowSparse,max_modulus_chirp, energy_fraction);
    Eigen::SparseVector<t_complex> chirp_row(Npix);
    chirp_row.reserve(rowSparse.nonZeros());
    for (Eigen::SparseVector<t_real>::InnerIterator itr(rowSparse); itr; ++itr){
        // t_complex val = rowC(itr.index());
        chirp_row.coeffRef(itr.index())= rowC(itr.index());
    }  
    // saveMarket(chirp_row,"./outputs/chirpF_sparse.txt");

    return chirp_row;
  }

  Eigen::SparseVector<t_complex> row_wise_convolution( Eigen::SparseVector<t_complex> &Grid_,  Eigen::SparseVector<t_complex> &chirp_,  const t_int &Nx, const t_int &Ny){
      
      if (chirp_.nonZeros() ==1)
         return Grid_;
      Eigen::SparseVector<t_complex> output_row(Nx*Ny);
      output_row.reserve(Nx*Ny);
      t_int Nx2= Nx/2;
      t_int Ny2 = Ny/2;
      #pragma omp parallel for collapse(2)
        for(t_int i = 0; i < Nx; ++i){     
          for(t_int j = 0; j < Ny; ++j){ 
            Eigen::SparseVector<t_complex> Chirp = chirp_;
            Eigen::SparseVector<t_complex> Grid = Grid_;
            t_complex temp (0.0,0.0);
             
            for (Eigen::SparseVector<t_complex>::InnerIterator pix(Grid); pix; ++pix){                   
              Vector<t_int> image_row_col = utilities::ind2sub(pix.index(), Nx, Ny);  
              t_int ii = image_row_col(0); 
              t_int jj = image_row_col(1);
              t_int  i_fftshift, j_fftshift  ;   
              if(ii <  Nx2)   i_fftshift = ii + Nx2; 
              else{   i_fftshift = ii - Nx2;  }
              if(jj <  Ny2)   j_fftshift = jj + Ny2;
              else{   j_fftshift = jj - Ny2;  }  

              t_int  oldpixi = Nx2 - i + i_fftshift; 
              t_int  oldpixj = Ny2 - j + j_fftshift;    

              if ((oldpixi >= 0 and oldpixi < Nx) and (oldpixj >= 0 and oldpixj < Ny)){
                t_int chirp_pos  =  oldpixi * Ny + oldpixj   ;                             
                t_complex val = pix.value() * Chirp.coeffRef(chirp_pos);              
                // if (std::abs(val) > 1e-18)
                    temp = temp+ val;                   
              }                          
            }
            if(std::abs(temp) > 1e-18){
              t_int iii,jjj;

              if(i >= Nx2)   iii = i - Nx2;  else{   iii = i + Nx2;   }
              if(j >= Ny2)   jjj = j - Ny2;  else{   jjj = j + Ny2;   } 
              t_int pos = utilities::sub2ind(iii,jjj,Nx,Ny); 
              output_row.coeffRef(pos)= temp;       
            }            
          }  
        }
        std::cout<<"\noutput_row  1: "<<output_row.nonZeros()<<" elements";fflush(stdout);
        return output_row;  
  } 

  Sparse<t_complex> wprojection_matrix(const Sparse<t_complex> &Grid, const t_int& Nx, const t_int& Ny,const Vector<t_real> & w_components, const t_real &cell_x, const t_real &cell_y, const t_real& energy_fraction_chirp,const t_real& energy_fraction_wproj){
           
        typedef Eigen::Triplet<t_complex> T;
        std::vector<T> tripletList;
        const t_int Npix = Nx*Ny;
        const t_int Nvis = w_components.size();

        PURIFY_HIGH_LOG("Hard-thresholding of the Chirp kernels: energy [{}] ",energy_fraction_chirp);
        PURIFY_HIGH_LOG("Hard-thresholding of the rows of G: energy [{}]  ",energy_fraction_wproj);
        
        tripletList.reserve(floor(Npix*Nvis*0.2)); 
        #pragma omp parallel for 
        for(t_int m = 0;  m < Grid.outerSize(); ++m){ 
            PURIFY_DEBUG("CURRENT WPROJ - Kernel index [{}]",m);

            Eigen::SparseVector<t_complex> chirp(Npix);
            chirp =  create_chirp_row(w_components(m),cell_x, cell_y, Nx, Ny,energy_fraction_chirp);
            // saveMarket(chirp,"./outputs/chirp"+std::to_string(m)+".txt"); 
                    
            Eigen::SparseVector<t_complex> G_bis = Grid.row(m);  
            // saveMarket(G_bis,"./outputs/G"+std::to_string(m)+".txt"); 

            auto row = row_wise_convolution(G_bis,chirp,Nx,Ny); 
            // saveMarket(row,"./outputs/GF"+std::to_string(m)+".txt");  

            Eigen::SparseVector<t_real> absRow = row.cwiseAbs();
            sparsify_row_sparse(absRow, energy_fraction_wproj);
            // PURIFY_DEBUG("Number of nonzeros entries in chirp :[{}]",chirp.nonZeros());   
            // PURIFY_DEBUG("Number of nonzeros entries in sparseRow :[{}]",absRow.nonZeros());
           
            for (Eigen::SparseVector<t_real>::InnerIterator itr(absRow); itr; ++itr){
                #pragma omp critical (load1)  
                     tripletList.push_back(T(m,itr.index(),row.coeffRef(itr.index())));  
            }
        }       
        Sparse<t_complex> Gmat(Nvis, Npix);
        Gmat.setFromTriplets(tripletList.begin(), tripletList.end());
        PURIFY_DEBUG("\nBuilding the rows of G.. DONE!\n");
        return Gmat;  
  }   

  void sparsify_row_sparse(Eigen::SparseVector<t_real> &row, const t_real &energy){
        /*
          Takes in a row of G and returns indexes of coeff to keep in the row sparse version 
          energy:: how much energy - in l2 sens - to keep after hard-thresholding 
        */
        if ( energy <1){
          t_real tau = 0.5;
          t_real old_tau = 1;
          t_int niters = 200;
          const t_real abs_row_total_energy = (row.cwiseProduct(row)).sum();
          t_real min_tau = 0.0;
          t_real max_tau = 1;
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
              for (Eigen::SparseVector<t_real>::InnerIterator itr(row); itr; ++itr){
                      if  (itr.value() > tau__)
                          energy_sum +=  itr.value() * itr.value() ;                    
                  }
              energy_sum= energy_sum/abs_row_total_energy;      
              if ( (std::abs(tau - old_tau)/std::abs(old_tau) < 1e-6) or  ((energy_sum>=energy)  and ((energy_sum/energy - 1) <0.001))){
                 break;        
              }  
              
              old_tau = tau;         
              if (energy_sum > energy) {
                      min_tau = tau; 
                     
                    }
              if (energy_sum < energy)  
                      max_tau = tau;                
              tau = (max_tau + min_tau)*0.5;
              
              if (i == niters-1)   { 
                tau = old_tau;      
                }     
          }   
          /* performing clipping */ 
          t_real tau_n = std::max(tau * abs_row_max,1e-10);
          row.prune(tau_n,1);
        }         
  }
  void sparsify_row_sparse_dense(Eigen::SparseVector<t_real> &row, const t_real &abs_row_max, const t_real &energy){
        /*
          Takes in a row of G and returns indexes of coeff to keep in the row sparse version 
          energy:: how much energy - in l2 sens - to keep after hard-thresholding 
        */
        if (energy <1.0){
          t_real tau = 0.5;
          t_real old_tau = 1;
          t_int niters = 100; 
          t_real min_tau = 0;
          t_real max_tau = 1;
          const t_real abs_row_total_energy = (row.cwiseProduct(row)).sum();    
          t_int rowLength = row.size();

          /* calculating threshold  */
          t_real energy_sum = 0;
          t_real tau__=0;
          for (t_int i = 0; i < niters; ++i){            
              energy_sum = 0;    
              tau__ =   tau *  abs_row_max;    
              for (Eigen::SparseVector<t_real>::InnerIterator itr(row); itr; ++itr){
                      if  (itr.value() > tau__)
                          energy_sum +=  itr.value() * itr.value() ;                    
              }
              energy_sum= energy_sum/abs_row_total_energy;      
              if ( (std::abs(tau - old_tau)/std::abs(old_tau) < 1e-6) or  ( (energy_sum>=energy)  and (std::abs(energy_sum/energy - 1) <0.001) )){
                 break;        
              }  
              else{
                     old_tau = tau;
                    if (energy_sum > energy)   {
                        min_tau = tau;
                       }
                    else{  max_tau = tau; }
                    tau = (max_tau + min_tau) * 0.5 ;
              }
              if (i == niters-1)                 
                tau = min_tau;           
          }   
          /* performing clipping */ 
          t_real tau_n = std::max(tau * abs_row_max,1e-10);
          row.prune(tau_n,1);
        }        
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

  Eigen::SparseVector<t_complex> generate_vect(const t_int & x_size, const t_int & y_size,const t_real &sigma,const t_real &mean){
      
        // t_real sigma = 3;
        t_int W = x_size;
        Matrix<t_complex> chirp = Image<t_complex>::Zero(y_size, x_size);
        // t_real mean = 0;
        t_complex I(0, 1);
        t_real step = W/2;

        // t_complex sum = 0.0; // For accumulating the kernel values
        for (int x = 0; x < W; ++x){
            for (int y = 0; y < W; ++y){
              t_int xx = x -step;
              xx = xx * xx;
              t_int yy = y - step;
              yy = yy * yy;
              t_complex val =  exp( -0.5 * (xx/(sigma*sigma) + yy/(sigma*sigma)) );//* std::exp(- 2 * pi * I * (x * 0.5 + y * 0.5));
              if (std::abs(val) >1e-5 )      
                chirp(x,y) = val;
              else chirp(x,y) =0.0;
            }
        }
      
          chirp.resize(x_size*y_size,1);
          Eigen::SparseVector<t_complex> chirp_ = chirp.sparseView(1e-5,1);
          return chirp_; 
  }    

   t_real upsample_ratio_sim(const utilities::vis_params& uv_vis, const t_real& L, const t_real& M, const t_int& x_size, const t_int& y_size, const t_int& multipleOf){
            /*
              returns the upsampling (in Fourier domain) ratio
            */    
            const Vector<t_real> & u = uv_vis.u.cwiseAbs();
            const Vector<t_real> & v = uv_vis.v.cwiseAbs();
            const Vector<t_real> & w = uv_vis.w.cwiseAbs();
            Vector<t_real> uvdist =  (u.array() * u.array() + v.array() * v.array()).sqrt();
            t_real bandwidth = 2 * uvdist.maxCoeff();
            // std::cout<<"\nDEBUG: bandwidth "<<bandwidth;
            Vector<t_real> bandwidth_up_vector = 2 * ( uvdist + w * L * 0.5);
            t_real bandwidth_up = bandwidth_up_vector.maxCoeff();
            t_real ratio = bandwidth_up / bandwidth;
            // std::cout<<"\nDEBUG: bandwidthUP "<<bandwidth_up;

            // std::cout<<"DEBUG:Initially calculated Upsampling ratio:"<<ratio <<"\n";

          //sets up dimensions for new image - even size
          t_int new_x = floor(x_size * ratio)+utilities::mod(floor(x_size * ratio),2);
          t_int new_y = floor(y_size * ratio)+utilities::mod(floor(y_size * ratio),2);     
          if (utilities::mod(new_x,multipleOf) !=0)  new_x=multipleOf*floor((t_real)new_x/multipleOf)+ multipleOf;
            if (utilities::mod(new_y,multipleOf) !=0)  new_y=multipleOf*floor((t_real)new_y/multipleOf)+ multipleOf;
          
            // if (mod(new_y,multipleOf) !=0) std::cout <<"\nDEBUG: (in upsample_ratio_sim) ERROR!!!!!!  --- IMAGE SIZE y\n ";
            // if (mod(new_x,multipleOf) !=0) std::cout <<"\nDEBUG: (in upsample_ratio_sim) ERROR!!!!!!  --- IMAGE SIZE x\n ";

            // !!!!!!! assuming same ratio on x and y for now !!!!!!!
            ratio = ( (t_real)new_x)/((t_real)x_size);

            // std::cout<<"DEBUG: (in upsample_ratio_sim) new image size"<< new_x<<" old image size "<< x_size<<"\n"; 

            return ratio;
        }



}
}
 