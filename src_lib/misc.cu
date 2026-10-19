//nvcc -o libgpu_point.so gpu_point.cu -shared -lcublas -Xcompiler -fPIC -lgomp

// Mistakes here all by Sigurd, don't blame Kendrick

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/gpu_mm.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif

template<typename T> __global__ void pair_rot_kernel(T * tod, const int32_t * pairs, const T * coss, const T * sins, int32_t bsize, int inv, int32_t nsamp, int32_t npair, int32_t nblock) {
    int32_t pair = blockIdx.x;
    int32_t det1 = pairs[2*pair+0], det2 = pairs[2*pair+1];
    T * d1 = tod + (size_t)det1*nsamp;
    T * d2 = tod + (size_t)det2*nsamp;
    const T * cs = coss+ (size_t)pair*nblock;
    const T * ss = sins+ (size_t)pair*nblock;
    for(int32_t si = threadIdx.x; si < nsamp; si += blockDim.x) {
        int32_t block = si/bsize;
        if(block >= nblock) block = nblock-1;
        T c = cs[block], s = ss[block];
        if(inv) s = -s;
        T diff =  c*d1[si] + s*d2[si];
        T sum  = -s*d1[si] + c*d2[si];
        d1[si] = diff;
        d2[si] = sum;
    }
}

template<typename T>
extern void pair_rot(
    ksgpu::Array<T> &tod,                // (ndet,nsamp)
    const ksgpu::Array<int32_t> & pairs, // (npair,2)
    const ksgpu::Array<T> & coss,        // (npair,nblock)
    const ksgpu::Array<T> & sins,        // (npair,nblock)
    int32_t bsize,
    int inv)
{
    // The existing check functions didn't quite cover this case, so I just did it
    // directly. Can factor out later if necessary
    xassert(tod  .ndim == 2);
    xassert(pairs.ndim == 2);
    xassert(coss .ndim == 2);
    xassert(sins .ndim == 2);
    int32_t ndet = tod.shape[0];
    int32_t nsamp= tod.shape[1];
    int32_t npair = pairs.shape[0];
    int32_t nblock= coss.shape[1];
    xassert(coss.shape[0] == npair);
    xassert(sins.shape[0] == npair);
    xassert(tod  .is_fully_contiguous());
    xassert(pairs.is_fully_contiguous());
    xassert(coss .is_fully_contiguous());
    xassert(sins .is_fully_contiguous());
    pair_rot_kernel<T> <<<npair, 256>>> (tod.data, pairs.data, coss.data, sins.data, bsize, inv, nsamp, npair, nblock);
    CUDA_PEEK("pair_rot kernel launch");
}

// This should be safe for npair < max uint16 and nbin < max uint32,
// and pretty well performing as long as nbin*npair >>> bsize
template<typename T> __global__ void pair_cov_kernel(const T * tod, const int32_t * pairs, T * covs, int32_t bsize, int32_t nsamp, int32_t npair, int32_t nbin) {
    int32_t pair  = blockIdx.y;
    int32_t det1 = pairs[2*pair+0], det2 = pairs[2*pair+1];
    const T * d1 = tod + (size_t)det1*nsamp;
    const T * d2 = tod + (size_t)det2*nsamp;

    // Each thread does one bin. A block of threads does blockDim consecutive bins.
    // Have gridDim.x such blocks, which together process gridDim.x*blockDim.x bins

    for(int32_t bin = blockIdx.x*blockDim.x + threadIdx.x; bin < nbin; bin += gridDim.x*blockDim.x) {
        int32_t s1 = bin*bsize;
        int32_t s2 = bin < nbin-1 ? (bin+1)*bsize : nsamp;
        T cov = 0;
        for(int32_t si = s1; si < s2; si++)
            cov += d1[si]*d2[si];
        cov /= s2-s1;
        covs[pair*nbin+bin] = cov;
    }
}

template<typename T>
extern void pair_cov(
    const ksgpu::Array<T> &tod,          // (ndet,nsamp)
    const ksgpu::Array<int32_t> & pairs, // (npair,2)
    ksgpu::Array<T> & covs,              // (npair,nblock)
    int32_t bsize)
{
    // The existing check functions didn't quite cover this case, so I just did it
    // directly. Can factor out later if necessary
    xassert(tod  .ndim == 2);
    xassert(pairs.ndim == 2);
    xassert(covs .ndim == 2);
    int32_t ndet  = tod.shape[0];
    int32_t nsamp = tod.shape[1];
    int32_t npair = pairs.shape[0];
    int32_t nbin  = covs.shape[1];
    xassert(covs.shape[0] == npair);
    xassert(tod  .is_fully_contiguous());
    xassert(pairs.is_fully_contiguous());
    xassert(covs .is_fully_contiguous());
    int32_t block_size = 64;
    int32_t nblock = (nbin+block_size-1)/block_size;
    dim3 grid; grid.y = npair; grid.x = nblock;
    pair_cov_kernel<T> <<<grid, block_size>>> (tod.data, pairs.data, covs.data, bsize, nsamp, npair, nbin);
    CUDA_PEEK("pair_cov kernel launch");
}

template<typename T> __global__ void eigh2_kernel(const T * as, const T * bs, const T * cs, T * E1s, T * E2s, T * Vxs, T * Vys, int64_t nsamp) {
    int64_t off    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for(int64_t i = off; i < nsamp; i += stride) {
        T a = as[i], b = bs[i], c = cs[i];
        // Avoid overflow
        T e_norm = max(a, c);
        a /= e_norm; b /= e_norm; c /= e_norm;
        T trace = a+c;
        T diff  = a-c;
        T core  = sqrt(diff*diff+4*b*b);
        T E1    = (trace-core)/2;
        T E2    = (trace+core)/2;
        T ae    = a-E1;
        T x, y;
        if(abs(ae) >= abs(b)) { x = -b/ae; y = 1; }
        else                  { x = 1; y = -ae/b; }
        T v_norm = sqrt(x*x+y*y);
        E1s[i] = E1*e_norm;
        E2s[i] = E2*e_norm;
        Vxs[i] = x/v_norm;
        Vys[i] = y/v_norm;
    }
}

// Efficient batched 2x2 eigendecomposition of [[a,b],[b,c]]. E1 and E2 are the small and
// large eigenvalues respectively. Vx and Vy are the components of the first eigenvector.
// Unlimited batch sizes unlike cupy.linalg.eigh, and allocates no memory.
template<typename T>
extern void eigh2(
        const ksgpu::Array<T> & a, const ksgpu::Array<T> & b, const ksgpu::Array<T> & c,
        ksgpu::Array<T> & E1, ksgpu::Array<T> & E2, ksgpu::Array<T> & Vx, ksgpu::Array<T> & Vy) {
    xassert(a.ndim == 1 && b.ndim == 1 && c.ndim == 1 && E1.ndim == 1 && E2.ndim == 1 && Vx.ndim == 1 && Vy.ndim == 1);
    int64_t nsamp = a.shape[0];
    xassert(b.shape[0] == nsamp && c.shape[0] == nsamp && E1.shape[0] == nsamp && E2.shape[0] == nsamp && Vx.shape[0] == nsamp && Vy.shape[0] == nsamp);
    xassert(a.is_fully_contiguous());
    xassert(b.is_fully_contiguous());
    xassert(c.is_fully_contiguous());
    xassert(E1.is_fully_contiguous());
    xassert(E2.is_fully_contiguous());
    xassert(Vx.is_fully_contiguous());
    xassert(Vy.is_fully_contiguous());
    // Find how many streaming multiprocessors we have to get a good block size
    int current_device, num_sms;
    cudaGetDevice(&current_device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, current_device);
    int64_t bsize  = 256;
    int64_t nblock = min((nsamp+bsize-1)/bsize, num_sms*32L);
    eigh2_kernel<T> <<<nblock,bsize>>> (a.data, b.data, c.data, E1.data, E2.data, Vx.data, Vy.data, nsamp);
    CUDA_PEEK("eigh2 kernel launch");
}

#define INSTANTIATE(T) \
template void pair_rot( \
    ksgpu::Array<T> &tod, \
    const ksgpu::Array<int32_t> & pairs, \
    const ksgpu::Array<T> & coss, \
    const ksgpu::Array<T> & sins, \
    int32_t bsize, \
    int inv); \
template void pair_cov( \
    const ksgpu::Array<T> &tod, \
    const ksgpu::Array<int32_t> & pairs, \
    ksgpu::Array<T> & covs, \
    int32_t bsize); \
template void eigh2( \
    const ksgpu::Array<T> & a, \
    const ksgpu::Array<T> & b, \
    const ksgpu::Array<T> & c, \
    ksgpu::Array<T> & E1, \
    ksgpu::Array<T> & E2, \
    ksgpu::Array<T> & Vx, \
    ksgpu::Array<T> & Vy);

INSTANTIATE(float);
INSTANTIATE(double);

}

__global__ void clip_kernel(float * arr, long n, float vmin, float vmax)
{
    long off  = blockIdx.x*blockDim.x+threadIdx.x;
    long step = blockDim.x*gridDim.x;
    for(long i = off; i < n; i+= step)
        arr[i] = fmaxf(vmin, fminf(vmax, arr[i]));
}

extern "C" void clip(float * arr, long n, float vmin, float vmax) {
    clip_kernel<<<128,128>>>(arr, n, vmin, vmax);
}
