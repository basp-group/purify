#include "utilities.h"
#include <new>
#include <omp.h>
#include <time.h>
namespace purify {
	namespace utilities {
	utilities::vis_params random_sample_density(const t_int& vis_num, const t_real& mean, const t_real& standard_deviation){
		/*
			Generates a random sampling density for visibility coverage
			vis_num:: number of visibilities
			mean:: mean of distribution
			standard_deviation:: standard deviation of distirbution
		*/
			auto sample = [&mean, &standard_deviation] (t_real x) { 
				std::random_device rd;
				std::mt19937_64 rng(rd());
				t_real output = 4 * standard_deviation + mean;
				static std::normal_distribution<> normal_dist(mean, standard_deviation);
				//ensures that all sample points are within bounds
				while(std::abs(output - mean) > 3 * standard_deviation){
					output = normal_dist(rng);
				}
				if (output != output) std::cout << output << '\n';
				return output;
			 };

			utilities::vis_params uv_vis;
			uv_vis.u = Vector<t_real>::Zero(vis_num).unaryExpr(sample);
			uv_vis.v = Vector<t_real>::Zero(vis_num).unaryExpr(sample);
			uv_vis.w = Vector<t_real>::Zero(vis_num).unaryExpr(sample);
			uv_vis.weights = Vector<t_complex>::Constant(vis_num, 1);
			uv_vis.vis = Vector<t_complex>::Constant(vis_num, 1);
			uv_vis.phase_centre = 0;
			return uv_vis;
	}
	utilities::vis_params read_visibility(const std::string& vis_name, const bool w_term)
	  {
	    /*
	      Reads an csv file with u, v, visibilities and returns the vectors.

	      vis_name:: name of input text file containing [u, v, real(V), imag(V)] (separated by ' ').
	    */
	    std::ifstream temp_file(vis_name);
	    t_int row = 0;
	    std::string line;
	    //counts size of vis file
	    while (std::getline(temp_file, line))
	      ++row;
	    Vector<t_real> utemp = Vector<t_real>::Zero(row);
	    Vector<t_real> vtemp = Vector<t_real>::Zero(row);
	    Vector<t_real> wtemp = Vector<t_real>::Zero(row);
	    Vector<t_complex> vistemp = Vector<t_complex>::Zero(row);
	    Vector<t_complex> weightstemp = Vector<t_complex>::Zero(row);
	    std::ifstream vis_file(vis_name);

	    // reads in vis file
	    row = 0;
	    t_real real;
	    t_real imag;
	    std::string s;
	    std::string  entry;
	    while (vis_file)
	    {
	      if (!std::getline(vis_file, s)) break;
	      std::istringstream ss(s);
	      std::getline(ss, entry, ' ');
	      utemp(row) = std::stod(entry);
	      std::getline(ss, entry, ' ');
	      vtemp(row) = std::stod(entry);
	      
	      if (w_term)
	      {
	      	std::getline(ss, entry, ' ');
	      	wtemp(row) = std::stod(entry);
	      }

	      std::getline(ss, entry, ' ');
	      real = std::stod(entry);
	      std::getline(ss, entry, ' ');
	      imag = std::stod(entry);
	      vistemp(row) = t_complex(real, imag);
	      std::getline(ss, entry, ' ');
	      weightstemp(row) = 1/(std::stod(entry) * std::stod(entry));
	      ++row;
	    }
	    utilities::vis_params uv_vis;
	    uv_vis.u = utemp;
	    uv_vis.v = vtemp; // found that a reflection is needed for the orientation of the gridded image to be correct
	    uv_vis.w = wtemp;
	    uv_vis.vis = vistemp;
	    uv_vis.weights = weightstemp;
		
	    
	    return uv_vis;
	  }

	  void write_visibility(const utilities::vis_params& uv_vis, const std::string & file_name, const bool w_term)
	  {
	  	/*
			writes visibilities to output text file (currently ignores w-component)
			uv_vis:: input uv data
			file_name:: name of output text file
	  	*/
		std::ofstream out(file_name);
		out.precision(13);
		for (t_int i = 0; i < uv_vis.u.size(); ++i)
		{
			out << uv_vis.u(i) << " " << -uv_vis.v(i) << " ";
			if (w_term)
				out << uv_vis.w(i) << " ";
			out << std::real(uv_vis.vis(i)) << " " << std::imag(uv_vis.vis(i)) << " " << 1./std::sqrt(std::real(uv_vis.weights(i))) << std::endl;
		}
		out.close();
	  }

	  utilities::vis_params set_cell_size(const utilities::vis_params& uv_vis, t_real cell_size_u, t_real cell_size_v)
	  {
	      /*
	        Converts the units of visibilities to units of 2 * pi, while scaling for the size of a pixel (cell_size)

	        uv_vis:: visibilities
	        cell_size:: size of a pixel in arcseconds
	      */

	      utilities::vis_params scaled_vis;

	      if (cell_size_u == 0 and cell_size_v == 0)
	      {
	        Vector<t_real> u_dist = uv_vis.u.array() * uv_vis.u.array();
	        t_real max_u = std::sqrt(u_dist.maxCoeff());
	        cell_size_u = (180 * 3600) / max_u / purify_pi / 3 * 1.02; //Calculate cell size if not given one

	        Vector<t_real> v_dist = uv_vis.v.array() * uv_vis.v.array();
	        t_real max_v = std::sqrt(v_dist.maxCoeff());
	        cell_size_v = (180 * 3600) / max_v / purify_pi / 3  * 1.02; //Calculate cell size if not given one
	        std::cout << "PSF has a FWHM of " << cell_size_u * 3 << " x " << cell_size_v * 3 << " arcseconds" << '\n';
	      }
	      if (cell_size_v == 0)
	      {
	        cell_size_v = cell_size_u;
	      }

	      
	      std::cout << "Using a pixel size of " << cell_size_u << " x " << cell_size_v << " arcseconds" << '\n';
	      t_real scale_factor_u = 1;
	      t_real scale_factor_v = 1;
	      if (uv_vis.units.compare("lambda")==0)
	      {
	      	scale_factor_u = 180 * 3600 / cell_size_u / purify_pi;
	      	scale_factor_v = 180 * 3600 / cell_size_v / purify_pi;
	      	scaled_vis.w = uv_vis.w;
	      }
	      if (uv_vis.units.compare("radians")==0)
	      {
	      	scale_factor_u = 180 * 3600 / purify_pi;
	      	scale_factor_v = 180 * 3600 / purify_pi;
	      	scaled_vis.w = uv_vis.w;	      	
	      }
	      scaled_vis.u = uv_vis.u / scale_factor_u * 2 * purify_pi;
	      scaled_vis.v = uv_vis.v / scale_factor_v * 2 * purify_pi;
	      
	      scaled_vis.vis = uv_vis.vis;
	      scaled_vis.weights = uv_vis.weights;
	      scaled_vis.units = "radians";
	      scaled_vis.phase_centre = uv_vis.phase_centre;
	      return scaled_vis;
	  }

	  Vector<t_complex> apply_weights(const Vector<t_complex> visiblities, const Vector<t_complex> weights)
	  {
	    /*
	      Applies weights to visiblities, assuming they are 1/sigma^2.
	    */
	    Vector<t_complex> weighted_vis;
	    weighted_vis = (visiblities.array() * weights.array()).matrix();
	    return weighted_vis;
	  }

	  utilities::vis_params uv_scale(const utilities::vis_params& uv_vis, const t_int& sizex, const t_int& sizey)
	  {
	    /*
	      scales the uv coordinates from being in units of 2 * pi to units of pixels.
	    */
	      utilities::vis_params scaled_vis;
	      scaled_vis.u = uv_vis.u / (2 * purify_pi) * (t_real)sizex;
	      scaled_vis.v = uv_vis.v / (2 * purify_pi) * (t_real)sizey;
	      scaled_vis.vis = uv_vis.vis;
	      scaled_vis.weights = uv_vis.weights;
	      for (t_int i = 0; i < uv_vis.u.size(); ++i)
	      {
	      	scaled_vis.u(i) = utilities::mod(scaled_vis.u(i), sizex);
	      	scaled_vis.v(i) = utilities::mod(scaled_vis.v(i), sizey);
	      }
	      scaled_vis.w = uv_vis.w;
	      scaled_vis.units = "pixels";
	      scaled_vis.phase_centre = uv_vis.phase_centre;
	      return scaled_vis;
	  }

	  utilities::vis_params uv_symmetry(const utilities::vis_params& uv_vis)
	  {
	    /*
	      Adds in reflection of the fourier plane using the condjugate symmetry for a real image.

	      uv_vis:: uv coordinates for the fourier plane
	    */
	    t_int total = uv_vis.u.size();
	    Vector<t_real> utemp(2 * total);
	    Vector<t_real> vtemp(2 * total);
	    Vector<t_real> wtemp(2 * total);
	    Vector<t_complex> vistemp(2 * total);
	    Vector<t_complex> weightstemp(2 * total);
	    
	    for (t_int i = 0; i < uv_vis.u.size(); ++i)
	    {
	      utemp(i) = uv_vis.u(i);
	      vtemp(i) = uv_vis.v(i);
	      wtemp(i) = uv_vis.w(i);
	      vistemp(i) = uv_vis.vis(i);
	      weightstemp(i) = uv_vis.weights(i);
	    }
	    for (t_int i = total; i < 2 * total; ++i)
	    {
	      utemp(i) = -uv_vis.u(i - total);
	      vtemp(i) = -uv_vis.v(i - total);
	      wtemp(i) = uv_vis.w(i - total);
	      vistemp(i) = std::conj(uv_vis.vis(i - total));
	      weightstemp(i) = uv_vis.weights(i - total);
	    }
	    utilities::vis_params conj_vis;
	    conj_vis.u = utemp;
	    conj_vis.v = vtemp;
	    conj_vis.w = wtemp;
	    conj_vis.vis = vistemp;
	    conj_vis.weights = weightstemp;
	    conj_vis.units = uv_vis.units;
	    conj_vis.phase_centre = uv_vis.phase_centre;

	    return conj_vis;
	  }
	
	  t_int sub2ind(const t_int& row, const t_int& col, const t_int& rows, const t_int& cols) 
	  {
	    /*
	      Converts (row, column) of a matrix to a single index. This does the same as the matlab funciton sub2ind, converts subscript to index. 
	      Though order of cols and rows is probably transposed.

	      row:: row of matrix (y)
	      col:: column of matrix (x)
	      cols:: number of columns for matrix
	      rows:: number of rows for matrix
	     */
	    return row * cols + col;
	  }

	  Vector<t_int> ind2sub(const t_int& sub, const t_int& cols, const t_int& rows) 
	  {
	    /*
	      Converts index of a matrix to (row, column). This does the same as the matlab funciton sub2ind, converts subscript to index.
	      
	      sub:: subscript of entry in matrix
	      cols:: number of columns for matrix
	      rows:: number of rows for matrix
	      row:: output row of matrix
	      col:: output column of matrix

	     */
	    Vector<t_int> row_col(2);
	    row_col(1) = sub % cols;
	   	row_col(0) = floor((sub - row_col(1)) / cols);
	   	return row_col;
	  }

	  t_real mod(const t_real& x, const t_real& y) 
	  {
	    /*
	      returns x % y, and warps circularly around y for negative values.
	    */
	      t_real r = std::fmod(x, y);
	      if (r < 0)
	        r = y + r;
	      return r;
	  }


	  Image<t_complex> convolution_operator(const Image<t_complex>& a, const Image<t_complex>& b){
	    /*
	    returns the convolution of images a with images b
	    a:: vector a, which is shifted
	    b:: vector b, which is fixed
	    */

	    //size of a image
	    t_int a_y = a.rows();
	    t_int a_x = a.cols();
	    //size of b image
	    t_int b_y = b.rows();
	    t_int b_x = b.cols();    

	    Image<t_complex> output = Image<t_complex>::Zero(a_y + b_y, a_x + b_x);



	    for (t_int l = 0; l < b.cols(); ++l)
	    {
	      for (t_int k = 0; k < b.rows(); ++k)
	      {
	        output(k, l) = (a * b.block(k, l, a_y, a_x)).sum();
	      }
	    }

	    return output;
	  }

	    Matrix<t_complex> re_sample_ft_grid(const Matrix<t_complex>& input, const t_real& re_sample_ratio){
	    /*
	      up samples image using by zero padding the fft
	      
	      input:: fft to be upsampled, with zero frequency at (0,0) of the matrix.

	    */
	      if (re_sample_ratio == 1)
	      	return input;

	      //sets up dimensions for old image
	      t_int old_x = input.cols();
	      t_int old_y = input.rows();
    
	      t_int old_x_centre_floor = floor(input.cols() * 0.5);
	      t_int old_y_centre_floor = floor(input.rows() * 0.5);
	      //need ceilling in case image is of odd dimension
	      t_int old_x_centre_ceil = ceil(input.cols() * 0.5);
	      t_int old_y_centre_ceil = ceil(input.rows() * 0.5);

	      //sets up dimensions for new image -!!!!! enforce even sized images for now !!!!	      
	      // t_int new_y = floor((t_real)input.rows() * re_sample_ratio);
	      t_int new_x = floor((t_real)input.cols() * re_sample_ratio)+utilities::mod(floor((t_real)input.cols() * re_sample_ratio),2);
	      t_int new_y = floor((t_real)input.rows() * re_sample_ratio)+utilities::mod(floor((t_real)input.rows() * re_sample_ratio),2);

          // std::cout<<"DEBUG: (in re_sample_ft_grid) new image size " <<new_x<< " old image size "<< old_x <<"\n";

          if (new_x == old_x and new_y == old_y)
	      	return input;

	      t_int new_x_centre_floor = floor(new_x * 0.5);
	      t_int new_y_centre_floor = floor(new_y * 0.5);
	      //need ceilling in case image is of odd dimension
	      t_int new_x_centre_ceil = ceil(new_x * 0.5);
	      t_int new_y_centre_ceil = ceil(new_y * 0.5);

	      Matrix<t_complex> output = Matrix<t_complex>::Zero(new_y, new_x);

	      t_int box_dim_x;
	      t_int box_dim_y;


	      //now have to move each quadrant into new grid
	      box_dim_x = std::min(old_x_centre_floor, new_x_centre_floor);
	      box_dim_y = std::min(old_y_centre_floor, new_y_centre_floor);   
	      //(0, 0)
	      output.block(0, 0, box_dim_y, box_dim_x) = input.block(0, 0, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_floor, new_x_centre_floor);
	      box_dim_y = std::min(old_y_centre_ceil, new_y_centre_ceil);   
	      //(0, y0)
	      output.block(new_y - box_dim_y, 0, box_dim_y, box_dim_x) = input.block(old_y - box_dim_y, 0, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_ceil, new_x_centre_ceil);
	      box_dim_y = std::min(old_y_centre_floor, new_y_centre_floor);   
	      //(x0, 0)
	      output.block(0, new_x - box_dim_x, box_dim_y, box_dim_x) = input.block(0, old_y - box_dim_x, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_ceil, new_x_centre_ceil);
	      box_dim_y = std::min(old_y_centre_ceil, new_y_centre_ceil);  
	      //(x0, y0)
	      output.block(new_y - box_dim_y, new_x - box_dim_x, box_dim_y, box_dim_x) = input.block(old_y - box_dim_y, old_y - box_dim_x, box_dim_y, box_dim_x);

	      return output;
	  }
	 /* Image<t_complex> sparsify_chirp(const Image<t_complex>& row, const t_real& energy_fraction){
	      
	      // Takes in fourier transform of chirp, and returns sparsified version
	      //   row:: input fourier transform of chirp
	      //   energy_fraction:: how much energy to keep after sparsifying 
	        
	      //there is probably a way to get eigen to do this without a loop
	      if (energy_fraction == 1) 
	      	return row;
	      t_real tau = 0.5;
	      t_real old_tau = -1;
	      t_int niters = 100;
	      Image<t_real> abs_row = row.cwiseAbs();
	      t_real abs_row_max = abs_row.maxCoeff();
	      t_real abs_row_total_energy = (abs_row * abs_row).sum();

	      t_real min_tau = 0;
	      t_real max_tau = 1;
	      //calculating threshold
	      for (t_int i = 0; i < niters; ++i)
	      {
	        t_real energy_sum = 0;
	        for (t_int i = 0; i < abs_row.size(); ++i)
	        {
	          if (abs_row(i)/abs_row_max > tau)
	          {
	            energy_sum = energy_sum + abs_row(i) * abs_row(i) / abs_row_total_energy;
	          }
	        }

	        old_tau = tau;
	        if (energy_sum >= energy_fraction)
	        {
	          min_tau = tau;
	        }else{
	          max_tau = tau;
	        }
	        tau = (max_tau - min_tau) * 0.5 + min_tau;
	        //std::cout << energy_sum << '\n';
	        if (std::abs(tau - old_tau)/tau < 1e-6 and energy_sum > energy_fraction)
	        {
	        	tau = old_tau;
	        	break;
	        }
	        

	      }

	      Image<t_complex> output_row = Image<t_complex>::Zero(row.rows(), row.cols());
	      t_real final_energy = 0;
	      //performing clipping
	      for (t_int i = 0; i < abs_row.size(); ++i)
	      {
	        if (abs_row(i)/abs_row_max > tau)
	        {
	          output_row(i) = row(i);
	          final_energy = final_energy + abs_row(i) * abs_row(i);
	        }
	      }
	      //std::cout << "Final energy:" << '\n';
	      //std::cout << final_energy / abs_row_total_energy << '\n';
	      return output_row;
	  } 

	   
	  /*Image<t_complex> generate_chirp(const t_real w_term, const t_real cell_x, const t_real cell_y, const t_int x_size, const t_int y_size){
	    
	      // return chirp image fourier transform for w component

	      // w:: w-term in units of lambda
	      // celly:: size of y pixel in arcseconds
	      // cellx:: size of x pixel in arcseconds
	      // x_size:: number of pixels along x-axis
	      // y_size:: number of pixels along y-axis

	    
	    const t_real theta_FoV_L = cell_x * x_size;
	    const t_real theta_FoV_M = cell_y * y_size;

	    const t_real L = 2 * std::sin(purify_pi / 180.* theta_FoV_L / (60. * 60.) * 0.5);
	    const t_real M = 2 * std::sin(purify_pi / 180.* theta_FoV_M / (60. * 60.) * 0.5);

	    const t_real delt_x = L / x_size;
	    const t_real delt_y = M / y_size;

	    Image<t_complex> chirp(y_size, x_size);
	    t_complex I(0, 1);
	    for (t_int l = 0; l < x_size; ++l)
	    {
	      for (t_int m = 0; m < y_size; ++m)
	      {
	        t_real x = (l + 0.5 - x_size * 0.5) * delt_x;
	        t_real y = (m + 0.5 - y_size * 0.5) * delt_y;

	        chirp(m, l) = std::exp(- 2 * purify_pi * I * w_term * (std::sqrt(1 - x*x - y*y) - 1)) * std::exp(- 2 * purify_pi * I * (l * 0.5 + m * 0.5));
	      }
	    }
	    return chirp/chirp.size();
	  } 
    */

	/*Sparse<t_complex> convolution(const Sparse<t_complex> & input_gridding_matrix, const Image<t_complex>& Chirp, const t_int& Nx, const t_int& Ny, const t_int& Nvis){

			const Sparse<t_complex> & Grid = input_gridding_matrix.transpose();
	        std::cout << "Convolving Gridding matrix with Chirp" << std::endl;
	        std::cout << "Nx = " << Nx << " Ny = " << Ny << std::endl;
	        std::cout << Chirp.rows() << " " << Chirp.cols() << std::endl;
	        t_int Npix = Nx * Ny;
	        Sparse<t_complex> newG(Nvis, Npix);
	        Image<t_complex> Gtemp_mat(Nx, Ny);

	        typedef Eigen::Triplet<t_complex> T;
	        std::vector<T> tripletList;
	        
	        t_int sparsity = 0;
	        
	        for(t_int m = 0; m < Nvis; m++){//chirp->M
	        	
	            //loop over every pixels
	            for(t_int i = 0; i < Nx; i++){//nx
	                for(t_int j = 0; j < Ny; j++){ //ny
	                    t_complex  Gtemp0 (0.0,0.0);
	                    t_complex  chirptemp (0.0,0.0);

	                    //only loop over the non-zero gmat elements
	                    for (Eigen::SparseMatrix<t_complex>::InnerIterator pix(Grid,m); pix; ++pix){
	                        //express the column index as two-dimensional indices in image plane
	                        t_int ii, jj, i_fftshift, j_fftshift;

	                        Vector<t_int> image_row_col = ind2sub(pix.index(), Nx, Ny); 
	                        ii = image_row_col(0);
	                        jj = image_row_col(1);

	                        if(ii < Nx/2) i_fftshift = ii + Nx / 2;
	                        if(ii >= Nx/2) i_fftshift = ii - Nx / 2;
	                        if(jj < Ny/2) j_fftshift = jj + Ny/2;
	                        if(jj >= Ny/2) j_fftshift = jj - Ny/2;
	                        t_int oldpixi, oldpixj;
	                        
	                        //translate the chirp matrix for m to center on the pixel (i,j)
	                        //store old pixel indices of Chirp 
	                        oldpixi = Nx/2 - i + i_fftshift;
	                        oldpixj = Ny/2 - j + j_fftshift;
	                       
	                        //index of the chirp which translates to (ii,jj)
	                        t_int chirppixindex = oldpixi * Ny + oldpixj;
	                        //only add if within the overlap between chirp and Gmat
	                        //no circular convolution

	                        if(oldpixi >= 0 && oldpixi < Nx){
	                            if(oldpixj >= 0 && oldpixj < Ny){
							                    chirptemp = Chirp(m, chirppixindex);
	                                Gtemp0 = Gtemp0 + ( pix.value() * chirptemp );

	                            }
	                        }        
	                    }
	                    if (std::abs(Gtemp0) < 1e-13)
	                    	sparsity++;
	                    Gtemp_mat(i,j) = Gtemp0;        
	                }
	            }
	            
	            tripletList.reserve(sparsity);
	            for(t_int i=0; i<Nx; i++){
	                for(t_int j=0; j<Ny; j++){
	                    t_int ii, jj;
	                    if(i >= Nx/2) ii = i - Nx/2;
	                    if(i < Nx/2) ii = i + Nx/2;
	                    if(j >= Ny/2) jj = j - Ny/2;
	                    if(j < Ny/2) jj = j + Ny/2;
	    
	                    if(abs(Gtemp_mat(i, j)) > 1e-13){
	                        tripletList.push_back(T(m,sub2ind(ii,jj,Nx,Ny),Gtemp_mat(i, j)));
	                    }
	                } 
	            }

	        }

	        std::cout<<"---- After convolution  ---- "<<std::endl;

	    newG.setFromTriplets(tripletList.begin(), tripletList.end());

	    
	    return newG;    

	    } */

	    t_real upsample_ratio(const utilities::vis_params& uv_vis, const t_real& cell_x, const t_real& cell_y, const t_int& x_size, const t_int& y_size){
	        /*
	         returns the upsampling (in Fourier domain) ratio
	         */

	        const t_real theta_FoV_L = cell_x * x_size;
	    	const t_real theta_FoV_M = cell_y * y_size;

	    	const t_real L = 2 * std::sin(purify_pi / 180.* theta_FoV_L / (60. * 60.) * 0.5);
	    	const t_real M = 2 * std::sin(purify_pi / 180.* theta_FoV_M / (60. * 60.) * 0.5);

	    	const t_real FoV = std::max(L, M);
	        
	        const Vector<t_real> & w = uv_vis.w.cwiseAbs();
	        const Vector<t_real> uv_dist = (uv_vis.u.array() * uv_vis.u.array() + uv_vis.v.array() * uv_vis.v.array()).sqrt();
	        
	        const Vector<t_real> bandwidth_up_vector = uv_dist + w * FoV * 0.5;
	        
	        const t_real bandwidth_up = bandwidth_up_vector.maxCoeff();

	    	const t_real bandwidth = uv_dist.maxCoeff();

	        const t_real ratio = bandwidth_up / bandwidth;
	        return ratio;
	    }

	    utilities::vis_params whiten_vis(const utilities::vis_params& uv_vis){
	    	/*
				A function that whitens and returns the visibilities.

				vis:: input visibilities
				weights:: this expects weights that are the inverse of the complex variance, they are converted th RMS for whitenning.
	    	*/
				auto output_uv_vis = uv_vis;
	    		output_uv_vis.vis = uv_vis.vis.array() * uv_vis.weights.array().cwiseAbs().sqrt();
	    		return output_uv_vis;
	    }
	    t_real calculate_l2_radius(const Vector<t_complex> & y, const t_real& sigma){
	    	/*
				Calculates the epsilon, the radius of the l2_ball in sopt
				y:: vector for the l2 ball
	    	*/
			if (sigma == 0)
			{
				return std::sqrt(y.size() + 2 * std::sqrt(y.size())) * standard_deviation(y);
			}
	    	return std::sqrt(y.size() + 2 * std::sqrt(y.size())) * sigma;
	    }
	   	t_real SNR_to_standard_deviation(const Vector<t_complex>& y0, const t_real& SNR){
	   		/*
			Returns value of noise rms given a measurement vector and signal to noise ratio
			y0:: complex valued vector before noise added
			SNR:: signal to noise ratio

			This calculation follows Carrillo et al. (2014), PURIFY a new approach to radio interferometric imaging
	   		*/
	    	return y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(SNR / 20.0));
	    }

		Vector<t_complex> add_noise(const Vector<t_complex>& y0, 
				const t_complex& mean, const t_real& standard_deviation){
			/*
				Adds complex valued gaussian noise to vector
				y0:: vector beore noise
				mean:: complex valued mean of noise
				standard_deviation:: rms of noise
			*/
			auto sample = [&mean, &standard_deviation] (t_complex x) { 
				std::random_device rd_real;
				std::random_device rd_imag;
				std::mt19937_64 rng_real(rd_real());
				std::mt19937_64 rng_imag(rd_imag());
				static std::normal_distribution<t_real> normal_dist_real(std::real(mean), standard_deviation / std::sqrt(2));
				static std::normal_distribution<t_real> normal_dist_imag(std::imag(mean), standard_deviation / std::sqrt(2));
				t_complex I(0,1.);
				return normal_dist_real(rng_real) + I * normal_dist_imag(rng_imag);
			 };

			auto noise = Vector<t_complex>::Zero(y0.size()).unaryExpr(sample);

			return y0 + noise;
		}
		
		bool file_exists(const std::string& name) {
			/*
				Checks if a file exists
				name:: path of file checked for existance.
				returns true if exists and false if not exists
			*/
  			struct stat buffer;   
  			return (stat (name.c_str(), &buffer) == 0); 

		}	


        /* w - related utilities  */	    

	   

        utilities::vis_params read_uvw_sim(const std::string& vis_name, const t_real& lambda, bool w_term){
          /*
            Reads an csv file with u, v,  w coordinates in meter - for simulation purposes
            vis_name:: name of input text file containing [u, v, w] (separated by ' ') in units of wavelength \lambda.
          */

            std::ifstream temp_file(vis_name);
            t_int row = 0;
            std::string line;
            //counts size of vis file
            while (std::getline(temp_file, line))
            ++row;
            Vector<t_real> utemp(row);
            Vector<t_real> vtemp(row);
            Vector<t_real> wtemp(row);
            Vector<t_complex> vistemp(row);
            Vector<t_complex> weightstemp(row);
            std::ifstream vis_file(vis_name);

            // reads in vis file
            row = 0;
            std::string s;
            std::string  entry;
            while (vis_file){
                if (!std::getline(vis_file, s)) break;
                std::istringstream ss(s);
                std::getline(ss, entry, ' ');
                utemp(row) = std::stod(entry); 
                std::getline(ss, entry, ' ');
                vtemp(row) = std::stod(entry);
                if (w_term){ 
                	std::getline(ss, entry, ' ');
                	wtemp(row) = std::stod(entry); 
                }
                ++row;
            }
            utilities::vis_params uv_vis;
            uv_vis.u = utemp / lambda ;
            uv_vis.v = vtemp / lambda ; 
            uv_vis.weights = weightstemp.array()+1; // weights = 1  - for now -
            uv_vis.vis  = vistemp;
            if (w_term) uv_vis.w = wtemp.array() / lambda;
            uv_vis.units = "lambda";
            return uv_vis;
        }
        utilities::vis_params set_cell_size_sim(const utilities::vis_params& uv_vis, t_real bandwidthX, t_real bandwidthY){
          /*
             Converts the units of visibilities to units of 2 * pi, while scaling for the size of a pixel (cell_size)
             uv_vis:: visibilities
             cell_size:: size of a pixel in arcseconds
          */

           utilities::vis_params scaled_vis = uv_vis;
           scaled_vis.u = ( uv_vis.u / (bandwidthX) ) * purify_pi;
           scaled_vis.v = ( uv_vis.v / (bandwidthY) ) * purify_pi;
           return scaled_vis;
        }
        t_real upsample_ratio_sim(const utilities::vis_params& uv_vis, const t_real& L, const t_real& M, const t_int& x_size, const t_int& y_size, const t_int& multipleOf){
            /*
              returns the upsampling (in Fourier domain) ratio
            */    
            const Vector<t_real> & u = uv_vis.u.cwiseAbs();
            const Vector<t_real> & v = uv_vis.v.cwiseAbs();
            const Vector<t_real> & w = uv_vis.w.cwiseAbs();
            Vector<t_real> uvdist = (u.array() * u.array() + v.array() * v.array()).sqrt();
            t_real bandwidth = 2 * uvdist.maxCoeff();
            Vector<t_real> bandwidth_up_vector = 2 * ( uvdist + w * L * 0.5);
            t_real bandwidth_up = bandwidth_up_vector.maxCoeff();
            t_real ratio = bandwidth_up / bandwidth;

            //std::cout<<"DEBUG:Initially calculated Upsampling ratio:"<<ratio <<"\n";

	        //sets up dimensions for new image - even size
	        t_int new_x = floor(x_size * ratio)+mod(floor(x_size * ratio),2);
	        t_int new_y = floor(y_size * ratio)+mod(floor(y_size * ratio),2);     
	        if (mod(new_x,multipleOf) !=0)  new_x=multipleOf*floor(new_x/multipleOf)+ multipleOf;
            if (mod(new_y,multipleOf) !=0)  new_y=multipleOf*floor(new_y/multipleOf)+ multipleOf;
          
            // if (mod(new_y,multipleOf) !=0) std::cout <<"\nDEBUG: (in upsample_ratio_sim) ERROR!!!!!!  --- IMAGE SIZE y\n ";
            // if (mod(new_x,multipleOf) !=0) std::cout <<"\nDEBUG: (in upsample_ratio_sim) ERROR!!!!!!  --- IMAGE SIZE x\n ";

            // !!!!!!! assuming same ratio on x and y for now !!!!!!!
            ratio = ( (t_real)new_x)/((t_real)x_size);

            // std::cout<<"DEBUG: (in upsample_ratio_sim) new image size"<< new_x<<" old image size "<< x_size<<"\n"; 

            return ratio;
        }

        Matrix<t_complex> re_sample_ft_grid_sim(const Matrix<t_complex>& input, const t_real& re_sample_ratio, const t_int& multipleOf){
	        /*
	         up samples image using by zero padding the fft	      
	          input:: fft to be upsampled, with zero frequency at (0,0) of the matrix.
	          the final size should multiple of 2^level, level is the wavelet decomp. scale 
	        */
	      if (re_sample_ratio == 1)
	      	return input;

	      //sets up dimensions for old image
	      t_int old_x = input.cols();
	      t_int old_y = input.rows();

	      t_int old_x_centre_floor = floor(input.cols() * 0.5);
	      t_int old_y_centre_floor = floor(input.rows() * 0.5);
	      //need ceilling in case image is of odd dimension
	      t_int old_x_centre_ceil = ceil(input.cols() * 0.5);
	      t_int old_y_centre_ceil = ceil(input.rows() * 0.5);

	      //sets up dimensions for new image
	      t_int new_x = floor(input.cols() * re_sample_ratio)+mod(floor(input.cols() * re_sample_ratio),2);
	      t_int new_y = floor(input.rows() * re_sample_ratio)+mod(floor(input.cols() * re_sample_ratio),2);
	     
	      if (mod(new_x,multipleOf) !=0)  new_x=multipleOf*floor(new_x/multipleOf)+ multipleOf;
          if (mod(new_y,multipleOf) !=0)  new_y=multipleOf*floor(new_y/multipleOf)+ multipleOf;
          
          if (new_x == old_x and new_y == old_y)
	      	return input;

	      t_int new_x_centre_floor = floor(new_x * 0.5);
	      t_int new_y_centre_floor = floor(new_y * 0.5);
	      //need ceilling in case image is of odd dimension
	      t_int new_x_centre_ceil = ceil(new_x * 0.5);
	      t_int new_y_centre_ceil = ceil(new_y * 0.5);

	      Matrix<t_complex> output = Matrix<t_complex>::Zero(new_y, new_x);

	      t_int box_dim_x;
	      t_int box_dim_y;

	      //now have to move each quadrant into new grid
	      box_dim_x = std::min(old_x_centre_floor, new_x_centre_floor);
	      box_dim_y = std::min(old_y_centre_floor, new_y_centre_floor);   
	      //(0, 0)
	      output.block(0, 0, box_dim_y, box_dim_x) = input.block(0, 0, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_floor, new_x_centre_floor);
	      box_dim_y = std::min(old_y_centre_ceil, new_y_centre_ceil);   
	      //(0, y0)
	      output.block(new_y - box_dim_y, 0, box_dim_y, box_dim_x) = input.block(old_y - box_dim_y, 0, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_ceil, new_x_centre_ceil);
	      box_dim_y = std::min(old_y_centre_floor, new_y_centre_floor);   
	      //(x0, 0)
	      output.block(0, new_x - box_dim_x, box_dim_y, box_dim_x) = input.block(0, old_y - box_dim_x, box_dim_y, box_dim_x);

	      box_dim_x = std::min(old_x_centre_ceil, new_x_centre_ceil);
	      box_dim_y = std::min(old_y_centre_ceil, new_y_centre_ceil);  
	      //(x0, y0)
	      output.block(new_y - box_dim_y, new_x - box_dim_x, box_dim_y, box_dim_x) = input.block(old_y - box_dim_y, old_y - box_dim_x, box_dim_y, box_dim_x);
          // std::cout<<"\nsize of the upsampled FFT (" << output.rows() << " , " <<output.cols() << ")\n";
	      return output;

	    }

        Matrix<t_complex> generate_chirp(const t_real w_term, const t_real cell_x, const t_real cell_y, const t_int x_size, const t_int y_size){
	    
	      // return chirp image fourier transform for w component

	      // w:: w-term in units of lambda
	      // celly:: size of y pixel in arcseconds
	      // cellx:: size of x pixel in arcseconds
	      // x_size:: number of pixels along x-axis
	      // y_size:: number of pixels along y-axis
	    
	        const t_real theta_FoV_L = cell_x * x_size;
	        const t_real theta_FoV_M = cell_y * y_size;

	        const t_real L = 2 * std::sin(purify_pi / 180.* theta_FoV_L / (60. * 60.) * 0.5);
	        const t_real M = 2 * std::sin(purify_pi / 180.* theta_FoV_M / (60. * 60.) * 0.5);

	        const t_real delt_x = L / x_size;
	        const t_real delt_y = M / y_size;
	        Image<t_complex> chirp=Image<t_complex>::Zero(y_size, x_size);
	        t_complex I(0, 1);
	        t_real nz=1/((t_real) y_size*x_size);
             // std::cout<<nz<<"*";
             // fflush(stdout);
	           #pragma omp parallel for schedule(dynamic) num_threads(2)
	            for (t_int l = 0; l < x_size; ++l){
	                for (t_int m = 0; m < y_size; ++m) {

	                        t_real x = (l + 0.5 - x_size * 0.5) * delt_x;
	                        t_real y = (m + 0.5 - y_size * 0.5) * delt_y;
	                        chirp(m, l) =  nz * std::exp(- 2 * purify_pi * I * w_term * (std::sqrt(1 - x*x - y*y) - 1)) * std::exp(- 2 * purify_pi * I * (l * 0.5 + m * 0.5));
	                }
	            }

            fflush(stdout);
	        return chirp;

	    }    

        Sparse<t_complex> convolution(const Sparse<t_complex> & input_gridding_matrix, const Image<t_complex>& Chirp, const t_int& Nx, const t_int& Ny, const t_int& Nvis)
	    {
           
			const Sparse<t_complex> & Grid = input_gridding_matrix.transpose();
	        typedef Eigen::Triplet<t_complex> T;
	        std::vector<T> tripletList;
            t_int sparsity = 1;



	        std::cout << "Convolving Gridding matrix with the Chirpmatrix.." << std::endl;
	        std::cout << "Nx = " << Nx << " Ny = " << Ny << std::endl;
	        std::cout << Chirp.rows() << " " << Chirp.cols() << std::endl;
          	
          	tripletList.reserve(floor(Nx*Ny*Nvis*0.5));

            clock_t timeS=clock();
		    	
		        #pragma omp parallel for num_threads(12) schedule(dynamic)
		        for(t_int m = 0; m < Nvis; m++){    
		        	
		        	
                    if (utilities::mod(m,100) ==0){
                            std::printf(".");
                            fflush(stdout); 
                    } 
                    
		            if (utilities::mod(m,500) ==0) {
                          std::cout << " > "<<m<<" < ";                   
		            }
                    
                    Vector<t_complex> chirpRow = Chirp.row(m);

                    
                    #pragma omp parallel for 
	                for(t_int i = 0; i < Nx; i++){ 
	                    for(t_int j = 0; j < Ny; j++){ 
	              	        t_complex  Gtemp0 (0.0,0.0);            	          
	              	        for (Eigen::SparseMatrix<t_complex>::InnerIterator pix(Grid,m); pix; ++pix){  
	                    	            //only loop over the non-zero gmat elements
	                    	            //express the column index as two-dimensional indices in image plane
	                                    t_int  i_fftshift, j_fftshift;
	                                    Vector<t_int> image_row_col = ind2sub(pix.index(), Nx, Ny); 
	                                    t_int ii  = image_row_col(0); /* */  t_int jj = image_row_col(1);
	                                    if(ii <  Nx/2)   i_fftshift = ii + Nx/2; /* */  else{   i_fftshift = ii - Nx/2;  }
	                                    if(jj <  Ny/2)  j_fftshift = jj + Ny/2;	 /* */  else{    j_fftshift = jj - Ny/2;  }	                                        
	                                    //translate the chirp matrix for m to center on the pixel (i,j) //store old pixel indices of Chirp 
	                                    t_int  oldpixi = Nx/2 - i + i_fftshift; /* */  t_int  oldpixj = Ny/2 - j + j_fftshift;	                                           
	                                    t_int chirppixindex = oldpixi * Ny + oldpixj;//index of the chirp which translates to (ii,jj)
	                                    //only add if within the overlap between chirp and Gmat //no circular convolution
	                                    if((oldpixi >= 0 && oldpixi < Nx) and (oldpixj >= 0 && oldpixj < Ny) and (std::abs(chirpRow(chirppixindex)) >1e-14)) 
	  	                                        Gtemp0 += pix.value() * chirpRow(chirppixindex);                            
                            }	                                
	                        if(std::abs(Gtemp0) > 1e-14){
	                                    t_int iii, jjj,coord;
	                                    if(i >= Nx/2)   iii = i - Nx/2; /* */  else{   iii = i + Nx/2;   }
	                                    if(j >= Ny/2)   jjj = j - Ny/2; /* */  else{   jjj = j + Ny/2;   }
	                                    coord=sub2ind(iii,jjj,Nx,Ny);
	                                    #pragma omp critical (load)
	                                    {
	                                        tripletList.push_back(T(m,coord,Gtemp0));                                       
	                                    }
	                        }	                                      	          	                    	          	                                                            
                        } 
		            }                          
		        }


		  	std::cout << "\nDone!" ;
		  	fflush(stdout); 
            t_int Npix = Nx * Ny;	
            
 
            clock_t timeE=clock()-timeS;
            t_real ctime = timeE/double(CLOCKS_PER_SEC);
            std::cout << "\nTime to build G: " << ctime << " sc.\n";
            fflush(stdout); 
            Sparse<t_complex> Gmat(Nvis, Npix);
            Gmat.setFromTriplets(tripletList.begin(), tripletList.end());

            return Gmat;  
        }      
        
        Matrix<t_complex> sparsify_chirp(const Matrix<t_complex>& row, const t_real& energy){
	        /*
	           Takes in fourier transform of chirp, and returns sparsified version
	           row:: input fourier transform of chirp
	           energy_fraction:: how much energy to keep after sparsifying 
	        */
	        //there is probably a way to get eigen to do this without a loop
	       
	        if ( energy == 1)   return row;

	        t_real tau = 0.5;
	        t_real old_tau = -1;
	        t_int niters = 1000;
	        Image<t_real> abs_row = row.cwiseAbs();
	        t_real abs_row_max = abs_row.maxCoeff();
	        t_real abs_row_total_energy = (abs_row * abs_row).sum();
            auto rowLength=abs_row.size();
	        t_real min_tau = 0;
	        t_real max_tau = 1;
	        
	        /* calculating threshold */
	        for (t_int i = 0; i <niters; ++i){
	            t_real energy_sum = 0; 
                #pragma parallel omp for  reduction(+:energy_sum) num_threads(2) 
	                for (t_int i = 0; i < rowLength; ++i){
	                    if (abs_row(i)/abs_row_max > tau){
	                    	t_real add = abs_row(i) * abs_row(i);
	                        energy_sum += abs_row(i) * abs_row(i) ;
	                    }
	                }
	            
	            energy_sum= energy_sum/abs_row_total_energy;
	            	        
	            if ( (std::abs(energy_sum - energy)/energy < 0.001) and (std::abs(tau - old_tau)/old_tau < 1e-4)){
	                  // std::cout << ".EC: "<< 1-(energy_sum)<<".";
	             	break;
	            }
	            else{
	            old_tau = tau;
	            if (energy_sum > energy)  min_tau = tau;
                else{ max_tau = tau; }
	            tau = (max_tau - min_tau) * 0.5 + min_tau;  
	            }     
	        }
            
            /* performing clipping */
	        Image<t_complex> output_row = Image<t_complex>::Zero(row.rows(), row.cols());
	        #pragma omp parallel for num_threads(2) 
	            for (t_int i = 0; i < rowLength; ++i){
	                if (abs_row(i)/abs_row_max >tau)
	                    output_row(i) = row(i);
	            }
	        abs_row.resize(0,0);


	        return output_row;

	    }
	    Vector<t_complex> sparsify_row(const Vector<t_complex>& row, const t_real& energy){
	       /*
	        Takes in a row of G and returns sparsified version row
	        energy:: how much energy - in l2 sens - to keep after sparsifying 
	       */
	        
	      

	       t_real tau = 0.5;
	       t_real old_tau = -1;
	       t_int niters = 1000;
	       Vector<t_real> abs_row = row.cwiseAbs();
	       t_real abs_row_max = abs_row.maxCoeff();
	       t_real abs_row_total_energy = (abs_row.array() * abs_row.array()).sum();
	       t_real min_tau = 0;
	       t_real max_tau = 1;
	       t_int rowLength = abs_row.size();
	       Vector<t_complex> output_row = Vector<t_complex>::Zero(rowLength,1);

	        if ( energy == 1) 
	        { 
	        	for (t_int i = 0; i < rowLength; ++i){
	                output_row(i) = row(i); 
	            } 
	            return output_row;
	        }	    	
	        /* calculating threshold  */
	        for (t_int i = 0; i < niters; ++i)
	        {
	           
	            t_real energy_sum = 0;	    
        
	            #pragma omp parallel for  reduction(+:energy_sum)  num_threads(2)            
	                for (t_int i = 0; i < rowLength; ++i){
	                    if (abs_row(i)/abs_row_max > tau)
	                        energy_sum +=  abs_row(i) * abs_row(i) ;                    
	                }

	            energy_sum= energy_sum/abs_row_total_energy;	                
	                       
	            if ( (std::abs(tau - old_tau)/old_tau < 1e-4) and (std::abs(energy_sum/energy - 1) <0.001)  ){
	        	       // std::cout << ".EG: "<<1-(energy_sum) <<".";
	        	    break;
	            }
	            else{
	              old_tau = tau;
	              if (energy_sum > energy)   min_tau = tau; 
	              else{  max_tau = tau; }
	              tau = (max_tau - min_tau) * 0.5 + min_tau;
	            }
	        }

            /* performing clipping */
	        
	        #pragma omp parallel for num_threads(2) 
	        for (t_int i = 0; i < rowLength; ++i){
	                if (abs_row(i)/abs_row_max > tau)
	                   output_row(i) = row(i);            
	        }	    
	        return output_row;
	    }


        Sparse<t_complex> sparsify_rows_matrix( const Sparse<t_complex> & G, const t_real& energy){ 
            
            t_int timeS=clock();
            const t_int Nvis = G.rows();
            const t_int Npix = G.cols();
            
            typedef Eigen::Triplet<t_complex> T;
	        std::vector<T> tripletList;

            std::cout << "\nSparsification of the G matrix ..\n";
            std::cout<< "Loosing "<< (1 -energy)  << " of the total energy on each row of G\n";

            t_int sparsity=0;
            t_int reserve=0;

            /* Sparsification of each row of G */
            
            
            t_int nthreads= omp_get_num_threads();  
                
            // reserve=sparsity+2*Npix;
            // tripletList.reserve(reserve);   
            t_int chunksize =200;                
                
 
                #pragma omp parallel for  num_threads(3) schedule(static,chunksize)
                for (t_int m = 0; m < Nvis; ++m){  
                	if (m ==0) { 
                        std::cout<<"Approx. G in\n";
                        fflush(stdout);
                    }

                    if (utilities::mod(m,500) ==0) {
                    	std::cout << " -- "<<m;  fflush(stdout); 
                    }

                    Vector<t_complex> srow= sparsify_row(G.row(m),energy);               
                     #pragma omp critical (allocate)
                    { 
                        reserve=sparsity+3*Npix;
                    	tripletList.reserve(reserve); 
                    }   
                    #pragma omp parallel for 
                    for  (t_int i = 0; i < Npix; ++i){
	                	if (std::abs(srow(i)) >1e-14){

	                	    #pragma omp critical (load)
	                	  	{
	                	    	    tripletList.push_back(T(m,i,srow(i))); 
	                	            sparsity+=1;
	                        }                          
	                    }
                    }   
                                         
                }
          
            std::cout<<"Done!\n";
                    fflush(stdout); 

	        t_int timeE=clock();
	        Sparse<t_complex> SparseG(Nvis, Npix);
	        SparseG.setFromTriplets(tripletList.begin(), tripletList.end());
	        
	        std::cout << "\nTime to sparsify G " << (timeE-timeS)/double(CLOCKS_PER_SEC) << " sc.\n";
	                fflush(stdout); 


            return SparseG; 
	    }

       

        t_real sparsity_sp(const Sparse<t_complex> & Gmat){
        	 /*
        	    returs nber on non zero elts/ total nber of elts in Sparse Matrix  
        	 */
	        const Sparse<t_complex> & Grid = Gmat.transpose();
            const t_int Nvis = Gmat.rows();
            const t_int Npix = Gmat.cols();
            t_real sparsity=0;   
            for(t_int m = 0; m < Nvis; m++){
	            for (Eigen::SparseMatrix<t_complex>::InnerIterator pix(Grid,m); pix; ++pix){
	                    	sparsity++;
	            }
	        }
            t_real val =(sparsity/(t_real)(Nvis * Npix));
            std::cout<<" Sparsity perc: " << val << "\n";
            fflush(stdout); 
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
                            if (std::abs(Cmat(m,n)) > 1e-14)
	                    	    sparsity++;
	            }
	        }
            t_real val =(sparsity/(t_real)(Nvis * Npix)) ;
            std::cout<<" Sparsity perc: " << val << "\n";
            fflush(stdout); 
	        return val;
 
	    }

	    Vector<t_int> l0_row_matrix(const Sparse<t_complex> & Gmat){
	    	/*
               returns a vector of l_0 norms of  
                the rows of  a sparse Matrix Gmat  
  	    	*/
	        const Sparse<t_complex> & Grid = Gmat.transpose();
            const t_int Nvis = Gmat.rows();
            const t_int Npix = Gmat.cols();
            Vector<t_int> rowSp = Vector<t_int>::Zero(Nvis);
            
            #pragma omp parallel for schedule(dynamic)
                for(t_int m = 0; m < Nvis; m++)
                {
                 	t_int nberpix=0;
	                for (Eigen::SparseMatrix<t_complex>::InnerIterator pix(Grid,m); pix; ++pix)
	                {
	                    	nberpix++;
	                }
	                rowSp(m)=nberpix;
	            }
	       
	        return rowSp;
	    }
		t_real snr_(const Image<t_real> &model, const Image<t_real> &solution){
			/*
				Returns SNR of the estimated model image 
			*/
	    	t_real nm= model.matrix().norm();
	    	t_real ndiff = (model - solution).matrix().norm();
	    	t_real val = 20 * std::log10(nm/ndiff);
	    	return val;
	    }	 

	}
}