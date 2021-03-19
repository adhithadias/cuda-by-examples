
#include "cudaLib.cuh"

#include <math.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, int scale, int size) {
	//	Insert GPU SAXPY kernel code here
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i <size) y[i] = scale*x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";
	
	float *x = (float *) malloc(vectorSize * sizeof(float));
	float *y = (float *) malloc(vectorSize * sizeof(float));
	float *z = (float *) malloc(vectorSize * sizeof(float));
	
	float scale = 2;	
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	
	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif
	 
	float *d_x, *d_y;
	cudaMalloc((void**)&d_x, vectorSize * sizeof(float));
	cudaMalloc((void**)&d_y, vectorSize * sizeof(float));
	
	cudaMemcpy(d_x, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

	saxpy_gpu<<<(vectorSize+255)/256, 256>>>(d_x, d_y, scale, vectorSize);


	cudaMemcpy(z, d_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);
	
	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", z[i]);
		}
		printf(" ... }\n");
	#endif
	
	int errorCount = verifyVector(x, y, z, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";
	
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	free(z);

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<pSumSize) {
		curandState_t rng;
		curand_init(clock64(), i, 0, &rng);
		float randNum1 = curand_uniform(&rng);

		curand_init(clock64(), i, 0, &rng);
		float randNum2 = curand_uniform(&rng);
		// Get a new random value
		if (int(randNum1 * randNum1 + randNum2 * randNum2) == 0.0) {
			pSums[i] = 1;
		} else {
			pSums[i] = 0;
		}
		
	}

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0.0f;

	int blocks = 1024;
	int threads = 256;
	int vectorSize = blocks*threads;

	uint64_t *pSums = (uint64_t *)malloc(vectorSize*sizeof(uint64_t));
	uint64_t *pSums_d;
	cudaMalloc((void **)&pSums_d, vectorSize*sizeof(uint64_t));


	generatePoints<<<blocks, threads>>>(pSums_d, vectorSize, 1024);

	cudaMemcpy(pSums, pSums_d, vectorSize*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t hitCount = 0;
	for (uint64_t idx = 0; idx < vectorSize; ++idx) {
		if ( pSums[idx] == 1 ) {
			++hitCount;
		}
	}

	cudaFree(pSums_d);
	free(pSums);

	approxPi = ((double) hitCount / vectorSize);
	approxPi = approxPi * 4.0f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";	

	return approxPi;
}
