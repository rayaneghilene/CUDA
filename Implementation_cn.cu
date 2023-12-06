#include <stdio.h>
#include <stdlib.h>
#include <time.h>


////// 1) Une matrice float data de taille 1x32x32 initialisé à 0 qui prendra les valeurs de l'image d'entrée.


#include <stdio.h>



void MatrixPrint(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%f\t", M[i * p + j]);
    }
    printf("\n");
}


__global__ void matrixConvolution(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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

    float A[m * n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
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





////// 2) Une matrice float C1_data de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. 
////// C1 correspond aux données après la première Convolution.




////// 3) Une matrice float S1_data de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du sous-échantillonnage. 
////// S1 correspond aux données après le premier Sous-échantillonnage.







////// 4) Une matrice float C1_kernel de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.

