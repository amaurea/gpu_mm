#include "../include/gpu_mm.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif

template<typename T>
__global__ void tod2pickup_kernel(T * pickup, T * tod, T * x, int nsamp, int nx) {
    extern __shared__ char shared[];
    T * work = (T*) shared;

    // Zero work so we can accumulate into it
    for(int ix = threadIdx.x; ix < nx; ix += blockDim.x)
      work[ix] = 0;
    __syncthreads();
    // Do the accumulation with bilinear interpolation
    int det = blockIdx.x;
    T * trow = tod + (size_t)det*nsamp;
    T * prow = pickup + (size_t)det*nx;
    for(int i = threadIdx.x; i < nsamp; i += blockDim.x) {
      int ix = int(x[i]);
      T xrel = x[i]-ix;
      atomicAdd(&work[ix  ], trow[i]*(1-xrel));
      atomicAdd(&work[ix+1], trow[i]*xrel);
    }
    __syncthreads();
    // Accumulate into output array
    for(int ix = threadIdx.x; ix < nx; ix += blockDim.x)
      prow[ix] += work[ix];
}

template<typename T>
__global__ void pickup2tod_kernel(T * pickup, T * tod, T * x, int nsamp, int nx) {
    // Do the bilinear interpolation
    int det = blockIdx.x;
    T * trow = tod + (size_t)det*nsamp;
    T * prow = pickup + (size_t)det*nx;
    for(int i = threadIdx.x; i < nsamp; i += blockDim.x) {
      int ix = int(x[i]);
      T xrel = x[i]-ix;
      trow[i] += prow[ix]*(1-xrel) + prow[ix+1]*xrel;
    }
}

template<typename T>
extern void launch_tod2pickup(
    ksgpu::Array<T> &pickup,    // (ndet,nx)
    const ksgpu::Array<T> &tod, // (ndet,nsamp)
    const ksgpu::Array<T> &x)   // (nsamp). Assumed to be in range [0,nx-1] (yes, exclusive end)
{
    // The existing check functions didn't quite cover this case, so I just did it
    // directly. Can factor out later if necessary
    xassert(pickup.ndim == 2);
    xassert(tod.ndim == 2);
    xassert(x.ndim == 1);
    int nx    = pickup.shape[1];
    int ndet  = pickup.shape[0];
    int nsamp = x.shape[0];
    xassert(tod.shape[0] == ndet);
    xassert(tod.shape[1] == nsamp);
    xassert(tod.is_fully_contiguous());
    xassert(pickup.is_fully_contiguous());
    xassert(x.is_fully_contiguous());
    // Shared memory we need
    int shmem_nbytes = nx*sizeof(T);

    tod2pickup_kernel<T> <<< ndet, 256, shmem_nbytes>>> (pickup.data, tod.data, x.data, nsamp, nx);
    CUDA_PEEK("tod2pickup kernel launch");
}

template<typename T>
extern void launch_pickup2tod(
    const ksgpu::Array<T> &pickup, // (ndet,nx)
    ksgpu::Array<T> &tod,          // (ndet,nsamp)
    const ksgpu::Array<T> &x)      // (nsamp). Assumed to be in range [0,nx-1] (yes, exclusive end)
{
    // The existing check functions didn't quite cover this case, so I just did it
    // directly. Can factor out later if necessary
    xassert(pickup.ndim == 2);
    xassert(tod.ndim == 2);
    xassert(x.ndim == 1);
    int nx    = pickup.shape[1];
    int ndet  = pickup.shape[0];
    int nsamp = x.shape[0];
    xassert(tod.shape[0] == ndet);
    xassert(tod.shape[1] == nsamp);
    xassert(tod.is_fully_contiguous());
    xassert(pickup.is_fully_contiguous());
    xassert(x.is_fully_contiguous());

    pickup2tod_kernel<T> <<< ndet, 256>>> (pickup.data, tod.data, x.data, nsamp, nx);
    CUDA_PEEK("pickup2tod kernel launch");
}

#define INSTANTIATE(T) \
    template void launch_tod2pickup( \
        ksgpu::Array<T> &pickup, \
        const ksgpu::Array<T> &tod, \
        const ksgpu::Array<T> &x); \
    template void launch_pickup2tod( \
        const ksgpu::Array<T> &pickup, \
        ksgpu::Array<T> &tod, \
        const ksgpu::Array<T> &x);

INSTANTIATE(float);
INSTANTIATE(double);

}  // namespace gpu_mm

