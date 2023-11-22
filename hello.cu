#include <stdio.h>
#include <stdlib.h>
 
// __global__ keyword specifies a device kernel function
__global__ void kernelA(){
    printf("Hello, from the GPU!\n");
}
 
void MatrixInit(float *M, int n, int p){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            M[i*p + j] = (float)rand()/(float)RAND_MAX;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            printf("%f ", M[i*p + j]);
        }
        printf("\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
}
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < p){
        Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float sum = 0;
            for(int k = 0; k < n; k++){
                sum += M1[i*n + k] * M2[k*n + j];
            }
            Mout[i*n + j] = sum;
        }
    }
}


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < n){
        Mout[i*n + j] = M1[i*n + j] + M2[i*n + j];
    }
}


int main()
{
    printf("Hello, from the CPU!\n");
     
    // Set which device should be used
    // The code will default to 0 if not called though
    cudaSetDevice(0);
 
    // Call a device function from the host: a kernel launch
    // Which will print from the device
    kernelA <<<1,1>>>();


    // This call waits for all of the submitted GPU work to complete
    
    
    cudaDeviceSynchronize();
 
   return 0;
}
