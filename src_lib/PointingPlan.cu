#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>
#include <gputils/constexpr_functions.hpp>   // constexpr_is_log2()
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


struct cell_analysis
{
    uint icell;
    uint amask;
    int na;
    
    __device__ cell_analysis(int iycell, int ixcell)
    {
	icell = (iycell << 10) | ixcell;
	bool valid = (iycell >= 0) && (ixcell >= 0);
	
	int laneId = threadIdx.x & 31;
	uint lmask = (1U << laneId) - 1;   // all lanes lower than current lane
	uint mmask = __match_any_sync(ALL_LANES, icell);  // all matching lanes
	bool is_lowest = ((mmask & lmask) == 0);
	
	amask = __ballot_sync(ALL_LANES, valid && is_lowest);
	na = __popc(amask);	
    }
};


// absorb_mt(): Helper function for plan_kerne().
// The number of arguments is awkwardly large!

__device__ __forceinline__ void
absorb_mt(ulong *plan_mt, int *shmem,        // pointers
	  ulong &mt_local, int &nmt_local,   // per-warp ring buffer
	  uint icell, uint amask, int na,    // map cells to absorb
	  uint s, bool mflag, int na_prev,   // additional data needed to construct mt_new
	  int nmt_max, uint &err)            // error testing and reporting
{
    if (na == 0)
	return;
    
    // Block dims are (W,32), so threadIdx.x is the laneId.
    int laneId = threadIdx.x;
    
    // Logical laneId (relative to current value of nmt_local, wrapped around)
    int llid = (laneId + 32 - nmt_local) & 31;
    
    // Permute 'icell' so that llid=N contains the N-th active icell
    uint src_lane = __fns(amask, 0, llid+1);
    icell = __shfl_sync(ALL_LANES, icell, src_lane & 31);  // FIXME do I need "& 31"?

    // Reminder: mt bit layout is
    //   Low 10 bits = Global xcell index
    //   Next 10 bits = Global ycell index
    //   Next 26 bits = Primary TOD cache line index
    //   Next bit = mflag (does cache line overlap multiple map cells?)
    //   Next bit = zflag (mflag && first appearance of cache line)
    
    bool zflag = mflag && (na_prev == 0) && (llid == 0);

    // Construct mt_new from icell, s, mflag, zflag.
    uint mt20 = (s >> 5);
    mt20 |= (mflag ? (1U << 26) : 0);
    mt20 |= (zflag ? (1U << 27) : 0);
    
    ulong mt_new = icell | (ulong(mt20) << 20);

    // Extend ring buffer.
    // If nmt_local is >32, then it "wraps around" from mt_local to mt_new.
    
    mt_local = (laneId < nmt_local) ? mt_local : mt_new;
    nmt_local += na;

    if (nmt_local < 32)
	return;

    // If we get here, we've accumulated 32 values of 'mt_local'.
    // These values can now be written to global memory.

    // Output array index (in 'plan_mt' array)
    int nout = 0;
    if (laneId == 0)
	nout = atomicAdd(shmem, 32);
    nout = __shfl_sync(ALL_LANES, nout, 0);  // broadcast from lane 0 to all lanes
    nout += laneId;

    if (nout < nmt_max)
	plan_mt[nout] = mt_local;

    nmt_local -= 32;
    mt_local = mt_new;
    err = (nout < nmt_max) ? err : (err | errflag_inconsistent_nmt);
}


template<typename T, int W>
__global__ void plan_kernel(ulong *plan_mt, const T *xpointing, uint *nmt_cumsum, uint nsamp, uint nsamp_per_block, int nypix, int nxpix, uint *errp)
{
    // Assumed for convienience in shared memory logic
    static_assert(W <= 30);

    // FIXME can be removed soon
    assert(blockDim.x == 32);
    assert(blockDim.y == W);
		      
    // Block dims are (W,32)
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    // Shared memory layout:
    //   int    nmt_counter       running total of 'plan_mt' values written by this block
    //   int    sid_counter       running total of secondary cache lines for this block (FIXME no longer used)
    //   int    nmt_local[W]      used once at end of kernel
    
    __shared__ int shmem[32];  // only need (W+2) elts, but convenient to pad to 32
    
    // Zero shared memory
    if (warpId == 0)
	shmem[laneId] = 0;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min(nsamp, (b+1) * nsamp_per_block);
    
    // Range of nmt values to be written by this threadblock.
    uint mt_out0 = b ? nmt_cumsum[b-1] : 0;
    uint mt_out1 = nmt_cumsum[b];
    int nmt_max = mt_out1 - mt_out0;
    
    // Shift output pointer 'plan_mt'.
    // FIXME some day, consider implementing cache-aligned IO as optimization
    plan_mt += mt_out0;
    
    // (mt_local, nmt_local) act as a per-warp ring buffer.
    // The value of nmt_local is the same on all threads in the warp.
    ulong mt_local = 0;
    int nmt_local = 0;
    uint err = 0;

    for (uint s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	// cell_enumerator is defined in gpu_mm_internals.hpp
	cell_enumerator cells(ypix, xpix, nypix, nxpix, err);

	cell_analysis ca0(cells.iy0, cells.ix0);
	cell_analysis ca1(cells.iy0, cells.ix1);
	cell_analysis ca2(cells.iy1, cells.ix0);
	cell_analysis ca3(cells.iy1, cells.ix1);

	bool mflag = ((ca0.na + ca1.na + ca2.na + ca3.na) > 1);

	absorb_mt(plan_mt, shmem,        // pointers
		  mt_local, nmt_local,   // per-warp ring buffer
		  ca0.icell, ca0.amask, ca0.na,   // map cells to absorb
		  s, mflag, 0,           // additional data needed to construct mt_new
		  nmt_max, err);         // error testing and reporting
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  ca1.icell, ca1.amask, ca1.na,
		  s, mflag, ca0.na,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  ca2.icell, ca2.amask, ca2.na,
		  s, mflag, ca0.na + ca1.na,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  ca3.icell, ca3.amask, ca3.na,
		  s, mflag, ca0.na + ca1.na + ca2.na,
		  nmt_max, err);
    }
    
    if (laneId == 0)
	shmem[warpId+2] = nmt_local;

    __syncthreads();

    // FIXME logic here could be optimized -- align IO on cache lines,
    // use fewer warp shuffles to reduce.
    
    int shmem_remote = shmem[laneId];
    
    int nout = __shfl_sync(ALL_LANES, shmem_remote, 0);    // nmt_counter
    for (int w = 0; w < warpId; w++)
	nout += __shfl_sync(ALL_LANES, shmem_remote, w+2);  // value of 'nmt_local' on warp w
    nout += laneId;

    if ((laneId < nmt_local) && (nout < nmt_max))
	plan_mt[nout] = mt_local;

    bool fail = (warpId == (W-1)) && (laneId == 0) && ((nout + nmt_local) != nmt_max);
    err = fail ? (err | errflag_inconsistent_nmt) : err;

    errp[b] = err;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu,
			   const Array<unsigned char> &buf_, const Array<unsigned char> &tmp_buf) :
    nsamp(preplan.nsamp),
    nypix(preplan.nypix),
    nxpix(preplan.nxpix),
    pp(preplan),
    buf(buf_)
{
    check_buffer(buf, preplan.plan_nbytes, "PointingPlan constructor", "buf");
    check_buffer(tmp_buf, preplan.plan_constructor_tmp_nbytes, "PointingPlan constructor", "tmp_buf");
    check_xpointing(xpointing_gpu, preplan.nsamp, "PointingPlan constructor", true);   // on_gpu=true

    long max_nblocks = max(preplan.planner_nblocks, preplan.pointing_nblocks);
    long mt_nbytes = align128(preplan.plan_nmt * sizeof(ulong));
    long err_nbytes = align128(max_nblocks * sizeof(uint));
    size_t cub_nbytes = pp.cub_nbytes;
    
    xassert(preplan.plan_nbytes == mt_nbytes + err_nbytes);
    xassert(preplan.plan_constructor_tmp_nbytes == mt_nbytes + align128(cub_nbytes));

    this->plan_mt = (ulong *) (buf.data);
    this->err_gpu = (uint *) (buf.data + mt_nbytes);

    ulong *unsorted_mt = (ulong *) (tmp_buf.data);
    void *cub_tmp = (void *) (tmp_buf.data + mt_nbytes);

    // Number of warps in plan_kernel.
    constexpr int W = 4;

    plan_kernel<T,W> <<< pp.planner_nblocks, {32,W} >>>
	(unsorted_mt,             // ulong *plan_mt,
	 xpointing_gpu.data,      // const T *xpointing,
	 pp.nmt_cumsum.data,      // uint *nmt_cumsum,
	 pp.nsamp,                // uint nsamp,
	 pp.ncl_per_threadblock << 5,   // uint nsamp_per_block (FIXME 32-bit overflow)
	 pp.nypix,                // int nypix,
	 pp.nxpix,                // int nxpix,
	 this->err_gpu);          // uint *errp)

    CUDA_PEEK("plan_kernel launch");

    CUDA_CALL(cub::DeviceRadixSort::SortKeys(
        cub_tmp,         // void *d_temp_storage
	cub_nbytes,      // size_t &temp_storage_bytes
	unsorted_mt,     // const KeyT *d_keys_in
	this->plan_mt,   // KeyT *d_keys_out
	pp.plan_nmt,     // NumItemsT num_items
	0,               // int begin_bit = 0
	20               // int end_bit = sizeof(KeyT) * 8
	// cudaStream_t stream = 0
    ));
    
    check_gpu_errflags(this->err_gpu, pp.planner_nblocks, "PointingPlan constructor");
}


// This constructor allocates GPU memory (rather than using externally managed GPU memory)
template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu) :
    PointingPlan(preplan, xpointing_gpu,
		 Array<unsigned char>({preplan.plan_nbytes}, af_gpu), 
		 Array<unsigned char>({preplan.plan_constructor_tmp_nbytes}, af_gpu))
{ }


template<typename T>
void PointingPlan::map2tod(
    Array<T> &tod,
    const Array<T> &local_map,
    const Array<T> &xpointing,
    const LocalPixelization &lpix,
    bool allow_outlier_pixels,
    bool debug) const
{
    check_tod(tod, nsamp, "PointingPlan::map2tod", true);                 // on_gpu=true
    check_xpointing(xpointing, nsamp, "PointingPlan::map2tod", true);     // on_gpu=true

    launch_map2tod(
        tod, local_map, xpointing, lpix, this->plan_mt, this->err_gpu,
	this->pp.plan_nmt, this->pp.nmt_per_threadblock, this->pp.pointing_nblocks,
	allow_outlier_pixels, debug
    );
}


template<typename T>
void PointingPlan::tod2map(
    Array<T> &local_map,
    const Array<T> &tod,
    const Array<T> &xpointing,
    const LocalPixelization &lpix,
    bool allow_outlier_pixels,
    bool debug) const
{
    check_tod(tod, nsamp, "PointingPlan::tod2map", true);                 // on_gpu=true
    check_xpointing(xpointing, nsamp, "PointingPlan::tod2map", true);     // on_gpu=true

    launch_tod2map(
	local_map, tod, xpointing, lpix, this->plan_mt, this->err_gpu,
	this->pp.plan_nmt, this->pp.nmt_per_threadblock, this->pp.pointing_nblocks,
	allow_outlier_pixels, debug
    );
}


// Only used in unit tests
Array<ulong> PointingPlan::get_plan_mt(bool gpu) const
{
    int aflags = gpu ? af_gpu : af_rhost;
    cudaMemcpyKind direction = gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    
    Array<ulong> ret({pp.plan_nmt}, aflags);
    CUDA_CALL(cudaMemcpy(ret.data, this->plan_mt, pp.plan_nmt * sizeof(ulong), direction));
    return ret;
}


string PointingPlan::str() const
{
    // FIXME reduce cut-and-paste with PointingPrePlan::str()
    stringstream ss;
    
    ss << "PointingPlan("
       << "nsamp=" << nsamp
       << ", nypix=" << nypix
       << ", nxpix=" << nxpix
       << ", plan_nbytes=" << pp.plan_nbytes << " (" << nbytes_to_str(pp.plan_nbytes) << ")"
       << ", tmp_nbytes=" << pp.plan_constructor_tmp_nbytes << " (" << nbytes_to_str(pp.plan_constructor_tmp_nbytes) << ")"
       << ", overhead=" << pp.overhead
       << ", nmt_per_threadblock=" << pp.nmt_per_threadblock
       << ", pointing_nblocks=" << pp.pointing_nblocks
       << ")";

    return ss.str();
}


// -------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const gputils::Array<T> &xpointing_gpu, \
	const gputils::Array<unsigned char> &buf, \
	const gputils::Array<unsigned char> &tmp_buf); \
    \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const gputils::Array<T> &xpointing_gpu); \
    \
    template void PointingPlan::map2tod( \
	gputils::Array<T> &tod, \
	const gputils::Array<T> &local_map, \
	const gputils::Array<T> &xpointing, \
	const LocalPixelization &lpix, \
	bool allow_outlier_pixels, \
	bool debug) const; \
    \
    template void PointingPlan::tod2map( \
	gputils::Array<T> &local_map, \
	const gputils::Array<T> &tod, \
	const gputils::Array<T> &xpointing, \
	const LocalPixelization &lpix, \
	bool allow_outlier_pixels, \
	bool debug) const


INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
