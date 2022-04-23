/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef TEST_CACHED_VIEW_HPP
#define TEST_CACHED_VIEW_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>


#include <string>
#include <iostream>

#include <RDMA_Interface.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using DeviceSpace_t = Kokkos::CudaSpace;
using HostSpace_t = Kokkos::HostSpace;

using RemoteTraits = Kokkos::RemoteSpaces_MemoryTraitsFlags;

// This unit test covers our use-case in CGSOLVE
// We do not test for puts as we do not have that capability in RACERlib yet.
template <class Data_t> void test_cached_view1D(int dim0, int league_size, int team_s ) {
  int myRank;
  int numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  MPI_Barrier(MPI_COMM_WORLD);


  using ViewHost_1D_t =
      Kokkos::View<Data_t *, HostSpace_t>;
  using ViewDevice_1D_t =
      Kokkos::View<Data_t *, DeviceSpace_t>;                   
  using ViewRemote_1D_t =
      Kokkos::View<Data_t *, RemoteSpace_t,
                   Kokkos::MemoryTraits<RemoteTraits::Cached>>;

  ViewRemote_1D_t v_r = ViewRemote_1D_t("RemoteView", dim0);
  ViewDevice_1D_t v_d = ViewDevice_1D_t(v_r.data(),v_r.extent(0));
  ViewDevice_1D_t v_d_out_1 = ViewDevice_1D_t("DataView", v_d.extent(0));
  ViewDevice_1D_t v_d_out_2 = ViewDevice_1D_t("DataView", v_d.extent(0));
  ViewDevice_1D_t v_d_out_3 = ViewDevice_1D_t("DataView", v_d.extent(0));
  ViewHost_1D_t v_h = ViewHost_1D_t("HostView", v_d.extent(0));

  printf("Conf: %i, %i, %i, %i\n", dim0, league_size, team_s, v_r.extent(0));

  int num_teams = league_size;
  int num_teams_adjusted = num_teams - 2;
  int team_size = team_s;
  int thread_vector_length = 1;
  int nextRank = (myRank + 1) % numRanks;

  // Uniformly distributed as per documentation. 
  // Note: Adding dim0 mod numRanks to the last rank would be an error (segfault)
  // Note: if dim0 < numRanks, the kernels will perform 0 work

  auto local_range =
      Kokkos::Experimental::get_range(dim0,myRank);
   auto remote_range =
      Kokkos::Experimental::get_range(dim0,nextRank);

  auto size_local_rank = local_range.second - local_range.first + 1;
  auto size_next_rank = remote_range.second - remote_range.first + 1;

  size_t size_per_team = size_next_rank / num_teams_adjusted;
  size_t size_per_team_mod = size_next_rank % num_teams_adjusted;
     
  auto policy = Kokkos::TeamPolicy<>
       (num_teams, team_size, thread_vector_length);
  using team_t = Kokkos::TeamPolicy<>::member_type;

 Kokkos::parallel_for("Init", size_local_rank, KOKKOS_LAMBDA(const int i){
   size_t actual_size = (myRank == numRanks -1) ? size_next_rank : size_local_rank;
   v_d(i) = myRank * actual_size + i;
   printf("%i, %f\n", i, (double)myRank * actual_size + i);
  });

  Kokkos::fence(); 
  RemoteSpace_t().fence();
  Kokkos::Experimental::remote_parallel_for(
    "Increment", policy, KOKKOS_LAMBDA(const team_t& team) {

    size_t start = team.league_rank() * size_per_team;
    size_t block = team.league_rank() == team.league_size()-1 ? 
                                   size_per_team + size_per_team_mod : size_per_team;

    size_t actual_size = (myRank == numRanks -1) ? size_next_rank : size_local_rank;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,block),
        [&] (const size_t i) {
        size_t index = nextRank * actual_size + start + i;  
        double num = v_r(index);
        printf("Received idx:%li, local idx:%li, data:%f\n", index, start+i, num);          
        assert((int)num == index);
      /*  v_d_out_1(start+i) = v_r(index);
        v_d_out_2(start+i) = v_r(index);    
        v_d_out_3(start+i) = v_r(index);  */
      });
    }, v_r);

  Kokkos::fence();
  RemoteSpace_t().fence();

/*  size_t actual_size = (myRank == numRanks -1) ? size_next_rank : size_local_rank;

  Kokkos::deep_copy(v_h, v_d_out_1);
  for (int i = 0; i < size_next_rank; ++i)
    ASSERT_EQ(v_h(i), nextRank * actual_size + i);

  Kokkos::deep_copy(v_h, v_d_out_2);
  for (int i = 0; i <size_next_rank; ++i)
    ASSERT_EQ(v_h(i), nextRank * actual_size + i);

  Kokkos::deep_copy(v_h, v_d_out_3);
  for (int i = 0; i < size_next_rank; ++i)
    ASSERT_EQ(v_h(i), nextRank * actual_size + i);*/
}

TEST(TEST_CATEGORY, test_cached_view) {
  char * var;
  var = std::getenv("IS");
  int size = var == nullptr ? 64 : std::stoi(var);
  var = std::getenv("LS");
  int league_size = var == nullptr ? 3 : std::stoi(var);
  var = std::getenv("TS");
  int team_size = var == nullptr ? 1 : std::stoi(var);

  test_cached_view1D<double>(size,league_size, team_size); 
  //Do not repeat tests here - the ipc mem alloc might fail (to be fixed)
}

#endif /* TEST_CACHED_VIEW_HPP */
