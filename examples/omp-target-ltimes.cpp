//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <stddef.h>
#include <stdio.h>



#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"


#include <cstdio>
#include <cmath>


using namespace RAJA;


RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");


void runLTimesRajaKernel(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

	using namespace RAJA::statement;

  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }

  int hid = omp_get_initial_device();
  int did = omp_get_default_device();

  // create device memory
  double *d_ell, *d_phi, *d_psi;
  d_ell = static_cast<double*>(omp_target_alloc(sizeof(double) * ell_data.size(), did));
  d_phi = static_cast<double*>(omp_target_alloc(sizeof(double) * phi_data.size(), did));
  d_psi = static_cast<double*>(omp_target_alloc(sizeof(double) * psi_data.size(), did));

  // Copy to device
  omp_target_memcpy(
      &ell_data[0],
      d_ell,
      sizeof(double) * ell_data.size(),
      0,0, hid, did);
  omp_target_memcpy(
      &phi_data[0],
      d_phi,
      sizeof(double) * phi_data.size(),
      0,0,hid,did);
  omp_target_memcpy(
      &psi_data[0],
      d_psi,
      sizeof(double) * psi_data.size(),
      0,0,hid,did);


  // create views on data
  std::array<RAJA::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));



  using Pol = RAJA::KernelPolicy<
      Collapse<omp_target_parallel_collapse_exec,
        ArgList<0, 1, 2>,
        For<3, RAJA::seq_exec, Lambda<0>>>>;

  RAJA::Timer timer;
  timer.start();

  auto segments =  RAJA::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
      TypedRangeSegment<IDirection>(0, num_directions),
      TypedRangeSegment<IGroup>(0, num_groups),
      TypedRangeSegment<IZone>(0, num_zones));


  kernel<Pol>(

      segments,

      // Lambda_CalcPhi
      [=] (IMoment m, IDirection d, IGroup g, IZone z) {
        phi(m, g, z) += ell(m, d) * psi(d, g, z);
      });



  timer.stop();
  printf("LTimes took %lf seconds using RAJA::kernel\n",
      timer.elapsed());


  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }

}


int main(){

  bool debug = true;

  int m = 25;
  int d = 80;
  int g = 32;
  int z = 32*1024;

  printf("m=%d, d=%d, g=%d, z=%d\n", m, d, g, z);

  runLTimesRajaKernel(debug, m, d, g, z);

  return 0;
}


