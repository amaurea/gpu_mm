#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ void analyze_ypix(int &iy0, int &iy1, T &dy, T ypix, int nypix)
{
    iy0 = int(ypix);
    iy0 = (iy0 >= 0) ? iy0 : 0;
    iy0 = (iy0 <= nypix-2) ? iy0 : (nypix-2);
    iy1 = iy0 + 1;
    dy = ypix - iy0;
}


template<typename T>
__device__ void analyze_xpix(int &ix0, int &ix1, T &dx, T xpix, int nxpix)
{
    // Wrap around
    xpix = (xpix >= 0) ? xpix : (xpix + nxpix);
    xpix = (xpix <= nxpix) ? xpix : (xpix - nxpix);

    ix0 = int(xpix);
    ix0 = (ix0 >= 0) ? ix0 : 0;
    ix0 = (ix0 <= nxpix-1) ? ix0 : (nxpix-1);
    ix1 = (ix0 < (nxpix-1)) ? (ix0+1) : 0;
    dx = xpix - ix0;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ void add_tqu(T *map, int npix, T t, T q, T u, T w)
{
    atomicAdd(map, w*t);
    atomicAdd(map + npix, w*q);
    atomicAdd(map + 2*npix, w*u);
}


template<typename T>
__global__ void simple_tod2map_kernel(T *map, const T *tod, const T *xpointing, uint nsamp, int nypix, int nxpix, uint nsamp_per_block)
{
    static constexpr T one = 1;
    static constexpr T two = 2;
    
    int npix = nypix * nxpix;
    uint s0 = blockIdx.x * nsamp_per_block;
    uint s1 = min(nsamp, s0 + nsamp_per_block);
    
    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];
	T alpha = xpointing[s + 2*nsamp];
	T t = tod[s];

	// FIXME add 'status' argument, and calls to range_check_{xpix,ypix}().
	normalize_xpix(xpix, nxpix);   // defined in gpu_mm2_internals.hpp
	
	int iy0, iy1, ix0, ix1;
	quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp

	T dy = ypix - iy0;
	T dx = xpix - ix0;
	
	T q, u;	
	dtype<T>::xsincos(two*alpha, &q, &u);
	q *= t;
	u *= t;

	add_tqu(map + iy0*nypix + ix0, npix, t, q, u, (one-dy) * (one-dx));
	add_tqu(map + iy0*nypix + ix1, npix, t, q, u, (one-dy) * (dx));
	add_tqu(map + iy1*nypix + ix0, npix, t, q, u, (dy) * (one-dx));
	add_tqu(map + iy1*nypix + ix1, npix, t, q, u, (dy) * (dx));
    }
}


template<typename T>
void launch_simple_tod2map(Array<T> &map, const Array<T> &tod, const Array<T> &xpointing)
{
    long nsamp, nypix, nxpix;
    
    check_map_and_init_npix(map, nypix, nxpix, "launch_simple_tod2map", true);  // on_gpu=true
    check_tod_and_init_nsamp(tod, nsamp, "launch_simple_tod2map", true);        // on_gpu=true
    check_xpointing(xpointing, nsamp, "launch_simple_tod2map", true);           // on_gpu=true

    int nthreads_per_block = 128;
    int nsamp_per_block = 1024;
    int nblocks = (nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    simple_tod2map_kernel <<< nblocks, nthreads_per_block >>>
	(map.data, tod.data, xpointing.data, nsamp, nypix, nxpix, nsamp_per_block);

    CUDA_PEEK("simple_tod2map kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_simple_tod2map(Array<T> &map, const Array<T> &tod, const Array<T> &xpointing)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
