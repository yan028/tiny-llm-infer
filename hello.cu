#include <cstdio>
__global__ void hi() { printf("hello from thread %d\n", threadIdx.x); }
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s | sm_%d%d | %.1f GB | max threads/block %d\n",
           p.name, p.major, p.minor, p.totalGlobalMem / 1e9, p.maxThreadsPerBlock);
    hi<<<1, 4>>>(); cudaDeviceSynchronize();
}
