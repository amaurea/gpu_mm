#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


static void _check_map2tod_args(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing, int &ndet, int &nt, int &ndec, int &nra)
{
    xassert(tod.ndim == 2);
    xassert(tod.is_fully_contiguous());
    
    xassert(map.ndim == 3);
    xassert(map.shape[0] == 3);
    xassert(map.is_fully_contiguous());
    
    xassert(xpointing.ndim == 3);
    xassert(xpointing.shape[0] == 3);
    xassert(xpointing.shape[1] == tod.shape[0]);
    xassert(xpointing.shape[2] == tod.shape[1]);
    xassert(xpointing.is_fully_contiguous());

    ndet = tod.shape[0];
    nt = tod.shape[1];
    ndec = map.shape[1];
    nra = map.shape[2];
     
    xassert(ndet > 0);
    xassert(nt > 0);
    xassert(ndec > 0);
    xassert(nra > 0);

    xassert((nt % 32) == 0);
    xassert((ndec % 64) == 0);
    xassert((nra % 64) == 0);    
}


void reference_map2tod(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    xassert(tod.on_host());
    xassert(map.on_host());
    xassert(xpointing.on_host());

    int ndet, nt, ndec, nra;
    _check_map2tod_args(tod, map, xpointing, ndet, nt, ndec, nra);
    
    long nsamp = long(ndet) * long(nt);
    long npix = long(ndec) * long(nra);

    for (long s = 0; s < nsamp; s++) {
	float px_dec = xpointing.data[s];
	float px_ra = xpointing.data[s + nsamp];
	float alpha = xpointing.data[s + 2*nsamp];
	float cos_2a = cosf(2*alpha);
	float sin_2a = sinf(2*alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	
	xassert(idec >= 0);
	xassert(idec < ndec-1);
	xassert(ira >= 0);
	xassert(ira < nra-1);
	
	long ipix = idec*nra + ira;
	float out = 0.0;
	
	float w = (1.0-ddec) * (1.0-dra);
	out += w * map.data[ipix];
	out += w * cos_2a * map.data[ipix+npix];
	out += w * sin_2a * map.data[ipix+2*npix];

	w = (1.0-ddec) * (dra);
	out += w * map.data[ipix + 1];
	out += w * cos_2a * map.data[ipix+npix + 1];
	out += w * sin_2a * map.data[ipix+2*npix + 1];
	
	w = ddec * (1.0 - dra);
	out += w * map.data[ipix + nra];
	out += w * cos_2a * map.data[ipix+npix + nra];
	out += w * sin_2a * map.data[ipix+2*npix + nra];
	
	w = ddec * dra;
	out += w * map.data[ipix + nra+1];
	out += w * cos_2a * map.data[ipix+npix + nra+1];
	out += w * sin_2a * map.data[ipix+2*npix + nra+1];

	tod.data[s] = out;
    }
}


// -------------------------------------------------------------------------------------------------


__global__ void old_map2tod_kernel(float *tod, const float *map, const float *xpointing,
				   int ndet, int nt, int ndec, int nra, int nt_per_block)
{
    // Number of blocks should be: ceil(ns / nt_per_block)
    long ns = long(ndet) * long(nt);
    long s0 = long(blockIdx.x) * long(nt_per_block);

    tod += s0;
    xpointing += s0;
    
    int n = min(ns-s0, long(nt_per_block));
    int npix = ndec * nra;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
	float px_dec = xpointing[i];
	float px_ra = xpointing[i + ns];
	float alpha = xpointing[i + 2*ns];
	float cos_2a = cosf(2.0f * alpha);
	float sin_2a = sinf(2.0f * alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);

	// assert(idec >= 0);
	// assert(idec < ndec-1);
	// assert(ira >= 0);
	// assert(ira < nra-1);
	    
	int ipix = idec*nra + ira;
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	float out = 0.0f;

	float w = (1.0f - ddec) * (1.0f - dra);
	out += w * map[ipix];
	out += w * cos_2a * map[ipix+npix];
	out += w * sin_2a * map[ipix+2*npix];

	w = (1.0f - ddec) * (dra);
	out += w * map[ipix + 1];
	out += w * cos_2a * map[ipix+npix + 1];
	out += w * sin_2a * map[ipix+2*npix + 1];
	
	w = ddec * (1.0f - dra);
	out += w * map[ipix + nra];
	out += w * cos_2a * map[ipix+npix + nra];
	out += w * sin_2a * map[ipix+2*npix + nra];
	
	w = ddec * dra;
	out += w * map[ipix + nra+1];
	out += w * cos_2a * map[ipix+npix + nra+1];
	out += w * sin_2a * map[ipix+2*npix + nra+1];

	tod[i] = out;
    }
}


void launch_old_map2tod(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    static constexpr int nt_per_block = 16384;
    static constexpr int nthreads_per_block = 512;

    xassert(tod.on_gpu());
    xassert(map.on_gpu());
    xassert(xpointing.on_gpu());
	
    int ndet, nt, ndec, nra;
    _check_map2tod_args(tod, map, xpointing, ndet, nt, ndec, nra);

    long nblocks = (long(ndet) * long(nt) + nt_per_block - 1) / nt_per_block;
    xassert(nblocks < (1L << 31));

    old_map2tod_kernel<<< nblocks, nthreads_per_block >>>
	(tod.data, map.data, xpointing.data, ndet, nt, ndec, nra, nt_per_block);

    CUDA_PEEK("old_map2tod_kernel");
}


}  // namespace gpu_mm
