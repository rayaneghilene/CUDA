#include <stdio.h>
#include <stdlib.h>
#include <time.h>



/////////// PARTIE 1 - PRISE EN MAIN DE CUDA : Multiplication de matrices ///////////




////////////////////////////////////// MATRIX INIT //////////////////////////////////////

void MatrixInit(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
}

////////////////////////////////////// MATRIX PRINT //////////////////////////////////////


void MatrixPrint(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%f\t", M[i * p + j]);
    
    }
    printf("\n");
}

////////////////////////////////////// MATRIX ADD //////////////////////////////////////


void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
}

////////////////////////////////////// MATRIX MULTIPLICATION //////////////////////////////////////


void MatrixMult(float *M1, float *M2, float *Mout, int n){
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (k = 0; k < n; k++)
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
        }
    
}

////////////////////////////////////// MATRIX ADD GPU //////////////////////////////////////



__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
}


////////////////////////////////////// MATRIX MULTIPLICATION GPU //////////////////////////////////////

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;
    Mout[i * n + j] = 0;
    for (k = 0; k < n; k++)
        Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
}



int CPU_test() {
    // MatrixAdd test
    int n = 1500, p = 1500;
    float *M1, *M2, *Mout;

    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

 
    Mout = (float *)malloc(n * p * sizeof(float));
    
    
    // Measure execution time for MatrixAdd
    clock_t start_time_add = clock();
    MatrixAdd(M1, M2, Mout, n, p);
    clock_t end_time_add = clock();

    double execution_time_add = ((double)end_time_add - start_time_add) / CLOCKS_PER_SEC;
    printf("Execution time for MatrixAdd: %f seconds\n", execution_time_add);


    free(M1);
    free(M2);
    free(Mout);

    // MatrixMult test
    n = 1500;
    float *M3, *M4, *Mout2;

    M3 = (float *)malloc(n * n * sizeof(float));
    M4 = (float *)malloc(n * n * sizeof(float));

    MatrixInit(M3, n, n);
    MatrixInit(M4, n, n);

    Mout2 = (float *)malloc(n * n * sizeof(float));

    // Measure execution time for MatrixMult
    clock_t start_time_mult = clock();
    MatrixMult(M3, M4, Mout2, n);
    clock_t end_time_mult = clock();

    double execution_time_mult = ((double)end_time_mult - start_time_mult) / CLOCKS_PER_SEC;
    printf("Execution time for MatrixMult: %f seconds\n", execution_time_mult);

    free(M3);
    free(M4);
    free(Mout2);

    return 0;
}



 
int GPUtest() {
    //cudaMatrixAdd test
    int n = 100, p = 100;
    float *M1, *M2, *Mout;
    float *d_M1, *d_M2, *d_Mout;

    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    Mout = (float *)malloc(n * p * sizeof(float));

    cudaMalloc((void **)&d_M1, n * p * sizeof(float));
    cudaMalloc((void **)&d_M2, n * p * sizeof(float));
    cudaMalloc((void **)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_time_add = clock();
    cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout, n, p);
    clock_t end_time_add = clock();

    double execution_time_add = ((double)end_time_add - start_time_add) / CLOCKS_PER_SEC;
    printf("Execution time for MatrixAdd: %f seconds\n", execution_time_add);


    cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);

    //MatrixPrint(Mout, n, p);

    free(M1);
    free(M2);
    free(Mout);

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);



    // cudaMatrixMult test

    n = 100;
    float *M3, *M4, *Mout2;
    float *d_M3, *d_M4, *d_Mout2;

    M3 = (float *)malloc(n * n * sizeof(float));
    M4 = (float *)malloc(n * n * sizeof(float));

    MatrixInit(M3, n, n);
    MatrixInit(M4, n, n);

    Mout2 = (float *)malloc(n * n * sizeof(float));

    cudaMalloc((void **)&d_M3, n * n * sizeof(float));
    cudaMalloc((void **)&d_M4, n * n * sizeof(float));
    cudaMalloc((void **)&d_Mout2, n * n * sizeof(float));

    cudaMemcpy(d_M3, M3, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M4, M4, n * n * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_time_mult = clock();
    cudaMatrixMult<<<n, n>>>(d_M3, d_M4, d_Mout2, n);
    clock_t end_time_mult = clock();

    double execution_time_mult = ((double)end_time_mult - start_time_mult) / CLOCKS_PER_SEC;
    printf("Execution time for MatrixMult: %f seconds\n", execution_time_mult);

    cudaMemcpy(Mout2, d_Mout2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    //MatrixPrint(M3, n, n);
    //MatrixPrint(M4, n, n);

    //MatrixPrint(Mout2, n, n);

    free(M3);
    free(M4);
    free(Mout2);

    cudaFree(d_M3);
    cudaFree(d_M4);
    cudaFree(d_Mout2);

    return 0;

}


/////////////////////// Layer 1 - Génération des données de test //////////////////////////
////////////////////////////////////// TEST RAW DATA //////////////////////////////////////


int test_raw_data() {

///////// init the raw_data matrix with random values between 0 and 1 ///////////
    
    int n = 32;
    float* raw_data = (float*)malloc(n * n * sizeof(float));
    float *d_M_raw;
    MatrixInit(raw_data, n, n);   

    cudaMalloc((void **)&d_M_raw, n * n * sizeof(float));
    cudaMemcpy(d_M_raw, raw_data, n * n * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(raw_data, n, n);

    free(raw_data);
    cudaFree(d_M_raw);

///////// init the C1_data matrix with random values 0  /////////////////////////
     
    int c = 28;
    float* C1_data = (float*)malloc(6* c * c * sizeof(float));
    float *d_M_C1;
    int i, j, y;
    
    for (y = 0; y < 6; y++)
        for (i = 0; i < c; i++)
            for (j = 0; j < c; j++)
                C1_data[i * c + j] = static_cast<float>(0) ;


    cudaMalloc((void **)&d_M_C1, 6 * c * c * sizeof(float));
    cudaMemcpy(d_M_C1, C1_data, 6 * c * c * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(C1_data, 6*c, 6*c);

    free(C1_data);
    cudaFree(d_M_C1);

///////// init the S1_data matrix with random values 0  /////////////////////////
 
    int d = 14;
    float* S1_data = (float*)malloc(d * d * sizeof(float));
    float *d_M_S1;

    for (y = 0; y < 6; y++)
        for (i = 0; i < d; i++)
            for (j = 0; j < d; j++)
                C1_data[i * d + j] = static_cast<float>(0) ;

    cudaMalloc((void **)&d_M_S1, 6 * d * d * sizeof(float));
    cudaMemcpy(d_M_S1, S1_data, 6 * d * d * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(S1_data, 6*d, 6*d);

    free(S1_data);
    cudaFree(d_M_S1);


///////// init the C1_kernel matrix with random values 0  /////////////////////////
 
    int r = 5;
    float* C1_kernel = (float*)malloc(r * r * sizeof(float));
    float *d_M_C1_kernel;

    for (y = 0; y < 6; y++)
        for (i = 0; i < r; i++)
            for (j = 0; j < r; j++)
                C1_kernel[i * r + j] = (static_cast<float>(rand()) / RAND_MAX) * 2  ;

    cudaMalloc((void **)&d_M_C1_kernel, 6 * r * r * sizeof(float));
    cudaMemcpy(d_M_C1_kernel, C1_kernel, 6 * r * r * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(C1_kernel, 6*r, 6*r);

    free(C1_kernel);
    cudaFree(d_M_C1_kernel);



//////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Layer 2 - Convolution 2D ////////////////////////////





    return 0;
}




int main() {
    printf("Execution Time when using CPU\n");
    CPU_test();
    printf("Execution Time when using GPU\n");
    GPUtest();
    
    // test_raw_data();
    // conv_test();
    return 0;
}







