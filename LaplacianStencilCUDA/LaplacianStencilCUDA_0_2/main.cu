// 3D Laplacian stencil on the GPU, with an explicit host/device memory model
// (cudaMalloc + cudaMemcpy -- no unified memory) and a CPU reference for
// correctness checking.
//
// To run in Google Colab:
//   1. Runtime -> Change runtime type -> T4 GPU.
//   2. In a code cell:   !pip install nvcc4jupyter
//   3. In a code cell:   %load_ext nvcc4jupyter
//   4. In a new cell, put  %%cuda  on the first line and paste the rest of
//      this file below it. Running the cell invokes nvcc to build the
//      program for the attached GPU and streams its stdout back into the
//      notebook.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaEvent_t, etc.

#define XDIM 512
#define YDIM 512
#define ZDIM 512

// CUDA API calls and kernel launches often return asynchronously and fail
// silently -- errors from a bad kernel launch surface only on a later,
// unrelated call, and synchronous return codes (cudaError_t) are easy to
// drop on the floor. Wrapping every call site in CUDA_CHECK converts a
// silent failure into an immediate, line-tagged abort, so problems don't
// masquerade as correctness bugs downstream.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error " << cudaGetErrorString(err_)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------- CPU wall-clock timer ------------------------------------------
struct CpuTimer
{
    using clock_t = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock_t> mStart, mStop;

    void Start() { mStart = clock_t::now(); }
    void Stop(const std::string& msg)
    {
        mStop = clock_t::now();
        std::chrono::duration<double, std::milli> dt = mStop - mStart;
        std::cout << "[" << msg << dt.count() << " ms]" << std::endl;
    }
};

// ---------- GPU event-based timer -----------------------------------------
// Some CUDA calls (kernel launches in particular) return control to the
// host immediately after invocation, while the actual work continues
// asynchronously on the GPU. A plain wall-clock timer started on the host
// would therefore only capture the cost of issuing the launch, not the
// time the GPU spent doing the work. CUDA events let us record timestamps
// attached to the GPU-side work itself; calling cudaEventSynchronize on
// the stop event blocks the host until the GPU has actually finished,
// giving a well-defined elapsed time measured to completion.
struct GpuTimer
{
    cudaEvent_t mStart, mStop;

    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&mStart));
        CUDA_CHECK(cudaEventCreate(&mStop));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mStop);
    }

    void Start() { CUDA_CHECK(cudaEventRecord(mStart, 0)); }
    void Stop(const std::string& msg)
    {
        CUDA_CHECK(cudaEventRecord(mStop, 0));
        CUDA_CHECK(cudaEventSynchronize(mStop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
        std::cout << "[" << msg << ms << " ms]" << std::endl;
    }
};

// ---------- CPU reference kernel ------------------------------------------
void ComputeLaplacianCPU(const float (& __restrict__ u )[XDIM][YDIM][ZDIM],
                               float (& __restrict__ Lu)[XDIM][YDIM][ZDIM])
{
    for (int i = 1; i < XDIM - 1; i++)
    for (int j = 1; j < YDIM - 1; j++)
    for (int k = 1; k < ZDIM - 1; k++)
        Lu[i][j][k] =
            -6.f * u[i][j][k]
            + u[i + 1][j][k] + u[i - 1][j][k]
            + u[i][j + 1][k] + u[i][j - 1][k]
            + u[i][j][k + 1] + u[i][j][k - 1];
}

// ---------- GPU kernel ----------------------------------------------------
__global__ void LaplacianKernel(const float (& __restrict__ u )[XDIM][YDIM][ZDIM],
                                      float (& __restrict__ Lu)[XDIM][YDIM][ZDIM])
{
    // CUDA linearizes the 3D threadIdx into a single lane index using
    // the formula below: incrementing threadIdx.x bumps the linear ID by
    // 1, incrementing threadIdx.y bumps it by blockDim.x, and incrementing
    // threadIdx.z bumps it by blockDim.x * blockDim.y. The hardware then
    // packs every 32 consecutive lanes of that linear ID into one warp.
    auto tID = threadIdx.x
             + threadIdx.y * blockDim.x
             + threadIdx.z * blockDim.x * blockDim.y;

    // For our 8x8x8 block (512 threads = 9 bits, 3 bits per dim), peel
    // off each thread-local coordinate as a 3-bit field. Because k is
    // routed to the LOW bits of the linear thread ID, stepping through
    // successive thread IDs increments k by 1 -- and since k is also the
    // innermost (stride-1) dimension in memory, the two stride-1 axes
    // line up: every 8 consecutive thread IDs land on 8 consecutive
    // `float`s. A warp is actually 32 lanes, not 8, and spans 4 such
    // 8-wide groups (one per value of threadIdx.y), but within each
    // group the hardware can coalesce the loads into a single 32-byte
    // memory transaction instead of 8 separate ones. That is the
    // "coalescing" win we are buying here.
    int ti = (tID >> 6) & 0x7;
    int tj = (tID >> 3) & 0x7;
    int tk = (tID     ) & 0x7;

    int i = blockIdx.z * blockDim.z + ti;
    int j = blockIdx.y * blockDim.y + tj;
    int k = blockIdx.x * blockDim.x + tk;

    // The specific shifts and masks above (>> 3, >> 6, & 0x7) are tied
    // to our 8x8x8 block shape -- each dimension is 2^3, so every
    // coordinate is a 3-bit field of the linear thread ID. For other
    // power-of-2 block shapes the widths change accordingly. If the
    // block dimensions were *not* powers of two, drop the linearization
    // and just write
    //     int i = threadIdx.z, j = threadIdx.y, k = threadIdx.x;
    // (with the same blockIdx*blockDim offsets). The coalesced
    // innermost-CUDA-dim -> innermost-memory-dim mapping is preserved.

    // With threadIdx.z -> i, every lane of a warp shares the same `i`, so
    // this first check is warp-uniform (no divergence). The j and k
    // checks can still diverge but only on blocks that sit on the domain
    // boundary face.
    if (i < 1 || i >= XDIM - 1) return;
    if (j < 1 || j >= YDIM - 1) return;
    if (k < 1 || k >= ZDIM - 1) return;

    Lu[i][j][k] =
        -6.f * u[i][j][k]
        + u[i + 1][j][k] + u[i - 1][j][k]
        + u[i][j + 1][k] + u[i][j - 1][k]
        + u[i][j][k + 1] + u[i][j][k - 1];
}

int main()
{
    using array_t = float (&)[XDIM][YDIM][ZDIM];
    const std::size_t N     = std::size_t(XDIM) * YDIM * ZDIM;
    const std::size_t bytes = N * sizeof(float);

    // Host allocations. The ...Raw pointers own the flat storage; the
    // un-suffixed names are array-shaped views over the same memory.
    float* h_uRaw     = new float[N];
    float* h_LuRaw    = new float[N];  // receives the D2H-copied GPU result
    float* h_LuRefRaw = new float[N];  // CPU reference output, for validation

    array_t h_u     = reinterpret_cast<array_t>(*h_uRaw);
    array_t h_Lu    = reinterpret_cast<array_t>(*h_LuRaw);
    array_t h_LuRef = reinterpret_cast<array_t>(*h_LuRefRaw);

    std::memset(h_uRaw,     0, bytes);
    std::memset(h_LuRaw,    0, bytes);
    std::memset(h_LuRefRaw, 0, bytes);

    // Populate the interior of u with random floats in [-1, 1]. The boundary
    // shell stays zero (from memset above), which is what makes the i+/-1 /
    // j+/-1 / k+/-1 neighbor reads safe without any edge checks.
    {
        CpuTimer t;
        t.Start();
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (int i = 1; i < XDIM - 1; i++)
        for (int j = 1; j < YDIM - 1; j++)
        for (int k = 1; k < ZDIM - 1; k++)
            h_u[i][j][k] = dist(rng);
        t.Stop("RNG fill of u  : ");
    }

    // Device allocations, with matching array-shaped views. d_u / d_Lu hold
    // device addresses -- only the kernel may actually dereference them.
    float *d_uRaw = nullptr, *d_LuRaw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_uRaw,  bytes));
    CUDA_CHECK(cudaMalloc(&d_LuRaw, bytes));
    CUDA_CHECK(cudaMemset(d_LuRaw, 0, bytes));

    array_t d_u  = reinterpret_cast<array_t>(*d_uRaw);
    array_t d_Lu = reinterpret_cast<array_t>(*d_LuRaw);

    // Host -> Device copy of input.
    {
        GpuTimer t;
        t.Start();
        CUDA_CHECK(cudaMemcpy(d_uRaw, h_uRaw, bytes, cudaMemcpyHostToDevice));
        t.Stop("H2D copy of u  : ");
    }

    // Kernel launch. 3D grid, one thread per output voxel, global memory
    // only -- this variant fixes the thread->memory mapping so warp lanes
    // read and write contiguous `float`s in memory (coalesced). Shared-
    // memory tiling is the subject of a follow-up variant.
    dim3 block(8, 8, 8);
    // Grid dims match the kernel's mapping: .x drives k, .y drives j,
    // .z drives i. With XDIM = YDIM = ZDIM this is a 64x64x64 grid either
    // way; the order matters in general.
    dim3 grid((ZDIM + block.x - 1) / block.x,
              (YDIM + block.y - 1) / block.y,
              (XDIM + block.z - 1) / block.z);
    {
        GpuTimer t;
        t.Start();
        LaplacianKernel<<<grid, block>>>(d_u, d_Lu);
        t.Stop("GPU kernel     : ");
        CUDA_CHECK(cudaGetLastError());
    }

    // Device -> Host copy of output.
    {
        GpuTimer t;
        t.Start();
        CUDA_CHECK(cudaMemcpy(h_LuRaw, d_LuRaw, bytes, cudaMemcpyDeviceToHost));
        t.Stop("D2H copy of Lu : ");
    }

    // CPU reference.
    {
        CpuTimer t;
        t.Start();
        ComputeLaplacianCPU(h_u, h_LuRef);
        t.Stop("CPU reference  : ");
    }

    // Correctness: max absolute difference over the interior.
    float maxErr = 0.f;
    for (int i = 1; i < XDIM - 1; i++)
    for (int j = 1; j < YDIM - 1; j++)
    for (int k = 1; k < ZDIM - 1; k++) {
        float d = std::fabs(h_Lu[i][j][k] - h_LuRef[i][j][k]);
        if (d > maxErr) maxErr = d;
    }
    std::cout << "Max absolute error (GPU vs CPU) : " << maxErr << std::endl;

    // Repeat timing: now that correctness is established, launch the kernel
    // 10 more times back-to-back. The first run typically absorbs one-time
    // costs (context init, module load, first-touch paging on the device),
    // so the later runs give a cleaner steady-state figure.
    std::cout << std::endl;
    for (int run = 1; run <= 10; run++) {
        std::cout << "Running kernel iteration " << std::setw(2) << run << " ";
        GpuTimer t;
        t.Start();
        LaplacianKernel<<<grid, block>>>(d_u, d_Lu);
        t.Stop("Elapsed time : ");
        CUDA_CHECK(cudaGetLastError());
    }

    cudaFree(d_uRaw);
    cudaFree(d_LuRaw);
    delete[] h_uRaw;
    delete[] h_LuRaw;
    delete[] h_LuRefRaw;
    return 0;
}
