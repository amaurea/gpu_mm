#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
__global__ void unplanned_map2tod_kernel(
    T *tod,
    const T *lmap,
    const T *xpointing,
    const long *cell_offsets,
    long nsamp,
    uint nsamp_per_block,
    int nypix,
    int nxpix,
    int nycells,
    int nxcells,
    long ystride,
    long polstride,
    bool periodic_xcoord,
    bool partial_pixelization)
{
    // 'map_evaluator' and 'pixel_locator' are defined in gpu_mm_internals.hpp.
    map_evaluator<T,true> mev(lmap, cell_offsets, nycells, nxcells, ystride, polstride, partial_pixelization);
    pixel_locator<T> px(nypix, nxpix, periodic_xcoord);
    
    const long s0 = blockIdx.x * long(nsamp_per_block);
    const long s1 = min(nsamp, s0 + long(nsamp_per_block));
    uint err = 0;   // FIXME currently ignored

    for (long s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];
	T alpha = xpointing[s + 2*nsamp];

	T sin_2a, cos_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);

	px.locate(ypix, xpix, err);
	tod[s] = mev.eval(px, cos_2a, sin_2a, err);
    }
}


template<typename T>
void launch_unplanned_map2tod(
    Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "unplanned_map2tod", true);            // on_gpu = true
    check_local_map(local_map, local_pixelization, "unplanned_map2tod", true);  // on_gpu = true
    check_xpointing(xpointing, nsamp, "unplanned_map2tod", true);               // on_gpu = true

    int nthreads_per_block = 128;
    int nsamp_per_block = 1024;
    int nblocks = (nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    unplanned_map2tod_kernel <<< nblocks, nthreads_per_block >>>
	(tod.data,
	 local_map.data,
	 xpointing.data,
	 local_pixelization.cell_offsets_gpu.data,
	 nsamp,
	 nsamp_per_block,
	 local_pixelization.nycells << 6,   // FIXME nypix
	 local_pixelization.nxcells << 6,   // FIXME nxpix
	 local_pixelization.nycells,
	 local_pixelization.nxcells,
	 local_pixelization.ystride,
	 local_pixelization.polstride,
	 false,   // FIXME periodic_xcoord
	 false);  // FIXME partial_pixelization

    CUDA_PEEK("unplanned_map2tod kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_unplanned_map2tod(Array<T> &tod, const Array<T> &lmap, const Array<T> &xpointing, const LocalPixelization &lpix)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
