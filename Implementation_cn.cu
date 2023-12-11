#include <stdio.h>
#include <stdlib.h>
#include <time.h>







void MatrixPrint(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%f\t", M[i * p + j]);
    }
    printf("\n");
}


__global__ void matrixConvolution(float* A, float* B, float* C,  int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int num_layer = q;

/*    if (row < m && col < p && num_layer < q) {
        float sum = 0.0f;
        
        for (int z = 0; z < q; z++) {    
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * p + col];
            }
            C[row * p + col] = sum;
        }
        conv[num_layer] = C;
    }

*/
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }


}

void convolution(float* A, float* B, float* C, int m, int n, int p) {
    float* d_A;
    float* d_B;
    float* d_C;

    int size_A = m * n * sizeof(float);
    int size_B = n * p * sizeof(float);
    int size_C = m * p * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixConvolution<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int m = 5;
    int n = 5;
    int p = 28;


    float A[m * n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 , 27, 28};
    float B[n * p] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    float C[m * p];
    printf("A = ");
    MatrixPrint(A, m, n);
    printf("B = ");
    MatrixPrint(B, n, m);
    convolution(A, B, C, m, n, p);
    printf("C = ");
    MatrixPrint(C, m, p);
    return 0;
}