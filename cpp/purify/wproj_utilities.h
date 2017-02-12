#ifndef PURIFY_wproj_utilities_H
#define PURIFY_wproj_utilities_H

#include "purify/config.h"
#include <fstream>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include "purify/FFTOperator.h"
#include "purify/types.h"
#include <time.h>
#include <omp.h>

namespace purify {
  namespace wproj_utilities {
    Matrix<t_complex> generate_chirp(const t_real & w_rate, const t_real &cell_x, const t_real & cell_y, const t_int & x_size, const t_int & y_size);
    Eigen::SparseVector<t_complex> create_chirp_row(const t_real & w_rate, const t_real &cell_x, const t_real & cell_y,const t_real & ftsizev, const t_real & ftsizeu, const t_real& energy_fraction);
    
    void sparsify_row_sparse(Eigen::SparseVector<t_real> &row,  const t_real &energy);
    void sparsify_row_sparse_dense(Eigen::SparseVector<t_real> &row, const t_real &abs_row_max, const t_real &energy);

    Eigen::SparseVector<t_complex> row_wise_convolution( Eigen::SparseVector<t_complex> &Grid,  Eigen::SparseVector<t_complex> &Chirp,  const t_int &Nx,  const t_int &Ny);
    Sparse<t_complex> wprojection_matrix(const Sparse<t_complex> &Grid, const t_int& Nx, const t_int& Ny,const Vector<t_real> & w_components, const t_real &cell_x, const t_real &cell_y, const t_real& energy_fraction_chirp,const t_real& energy_fraction_wproj);
    
    t_real snr_metric(const Image<t_real> &model, const Image<t_real> &solution);
    t_real mr_metric(const Image<t_real> &model, const Image<t_real> &solution);

    t_real sparsity_sp(const Sparse<t_complex> & Gmat);
    t_real sparsity_im(const Image<t_complex> & Cmat);
  }
}

#endif
