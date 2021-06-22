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

#ifndef RACERLIB_INTERFACE
#define RACERLIB_INTERFACE

#include <Kokkos_Core.hpp>
#include <RDMA_Engine.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

template <typename T>
struct Engine;
#define RACERLIB_SUCCESS 1

// Todo: template this on Feature for generic Engine feature support
template <typename data_type>
struct Engine {

void put(void *comm_id, void *allocation, data_type &value, int PE, int offset);
data_type get(void *comm_id, void *allocation, int PE, int offset);

// Call this at View memory allocation (allocation record)
int start(void *comm_id, void *allocation_id);
// Call this at View memory deallocation (~allocation record);
int stop(void *comm_id, void *allocation_id);
// Call this on fence. We need to make sure that at sychronization points,
// caches are empty
int flush(void *comm_id, void *allocation);
// Call this on Kokkos initialize.
int init(void *comm_id); // set communicator reference, return RACERLIB_STATUS
// Call this on kokkos finalize
int finalize(
    void *comm_id); // finalize communicator instance, return RECERLIB_STATUS


  RdmaScatterGatherEngine *sge;
  RdmaScatterGatherWorker *sgw;
  std::set<RdmaScatterGatherEngine *> sges;
  
  Engine();
  void allocate_device_component(void *p, MPI_Comm comm);
  void allocate_host_component();
  // Dealloc all for now.
  void deallocate_device_component();
  void deallocate_host_component();
  RdmaScatterGatherWorker *get_worker() const;
  RdmaScatterGatherEngine *get_engine() const;
  ~Engine();
  void fence();
};

} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos

#endif //RACERLIB_INTERFACE