#include <stdio.h>
#include <stdlib.h>
#include <time.h>



/////////// PARTIE 1 - PRISE EN MAIN DE CUDA : Multiplication de matrices ///////////




////////////////////////////////////// MATRIX INIT //////////////////////////////////////

void MatrixInit(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX) /* * 2 - 1*/;
}

////////////////////////////////////// MATRIX PRINT //////////////////////////////////////


void MatrixPrint(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%.2f\t", M[i * p + j]); // Print the matrix element with 2 decimal places
    
    }
    printf("\n");
}


////////////////////////////////////// ELEMENT MATRIX MULTIPLICATION GPU //////////////////////////////////////

__global__ void cudaElementMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    Mout[i * n + j] = M1[i * n + j] * M2[i * n + j];
}

//////////////////////////////////////  MATRIX SUM GPU  //////////////////////////////////////

__global__ void cudaElementMatrixSum(float *M, float *sum, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    atomicAdd(sum, M[i * n + j]);
}


//////////////////////////////////////  MATRIX CONVOLUTION GPU  //////////////////////////////////////

/*
__global__ void cudaMatrixConvolution(float *M, float *Kernel, float *Out,  int n, int p , int r) {
    /*This function takes as input
    * M: the input matrix
    * Kernel: the kernel matrix
    * Out: the output matrix
    * n: the size of the input matrix
    * p: the size of the kernel matrix
    * r: the size of the output matrix
    */

/*
    int i = blockIdx.x;
    int j = threadIdx.x;
    float *Mout;

    int k, l;
    float sum = 0;



    // Matrix Element Multiplication
    Mout[i * n + j] = M[i * n + j] * Kernel[i * n + j];
    // Matrix emelent sum
    atomicAdd(sum, Mout[i * n + j]);
    // assigning variables to the out matrix 
    Out[0] = sum;
*/

/*
    for (k = 0; k < r; k++)
        for (l = 0; l < r; l++)
            sum += M[(i + k) * n + (j + l)] * Kernel[k * r + l];

    Out[i * n + j] = sum;

    
}
*/
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

///////// init the C1_data matrix with 0 values   /////////////////////////
     
    int c = 28;
    float* C1_data = (float*)malloc( c * c * sizeof(float));
    float *d_M_C1;
    int i, j, y;
    
    for (y = 0; y < 1; y++)
        for (i = 0; i < c; i++)
            for (j = 0; j < c; j++)
                C1_data[i * c + j] = static_cast<float>(0) ;


    cudaMalloc((void **)&d_M_C1,  c * c * sizeof(float));
    cudaMemcpy(d_M_C1, C1_data, c * c * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(C1_data, c, c);

    free(C1_data);
    cudaFree(d_M_C1);

///////// init the S1_data matrix with random values 0  /////////////////////////
 
    int d = 14;
    float* S1_data = (float*)malloc( d * d * sizeof(float));
    float *d_M_S1;

    for (y = 0; y < 1; y++)
        for (i = 0; i < d; i++)
            for (j = 0; j < d; j++)
                S1_data[i * d + j] = static_cast<float>(0) ;

    cudaMalloc((void **)&d_M_S1,  d * d * sizeof(float));
    cudaMemcpy(d_M_S1, S1_data,  d * d * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(S1_data, d, d);

    free(S1_data);
    cudaFree(d_M_S1);


///////// init the C1_kernel matrix with random values 0  /////////////////////////
 
    int r = 5;
    float* C1_kernel = (float*)malloc(r * r * sizeof(float));
    float *d_M_C1_kernel;
    
    MatrixInit(C1_kernel, r, r);   

/*
    for (y = 0; y < 6; y++)
        for (i = 0; i < r; i++)
            for (j = 0; j < r; j++)
                C1_kernel[i * r + j] = (static_cast<float>(rand()) / RAND_MAX) * 2  ;

*/
    cudaMalloc((void **)&d_M_C1_kernel,  r * r * sizeof(float));
    cudaMemcpy(d_M_C1_kernel, C1_kernel,  r * r * sizeof(float), cudaMemcpyHostToDevice);
    // MatrixPrint(C1_kernel, r, r);

    free(C1_kernel);
    cudaFree(d_M_C1_kernel);



//////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Layer 2 - Convolution 2D ////////////////////////////



    return 0;
}




int main() {
    /* printf("Execution Time when using CPU\n");
    CPU_test();
    printf("Execution Time when using GPU\n");
    GPUtest();
    */ 

    float *d_M, *d_Out, *d_Kernel;
    float *M, *Out, *Kernel;

    int n = 5;
    int p = n, q = 1;
    int* d_sum;

    M = (float *)malloc(n * n * sizeof(float));
    Kernel = (float *)malloc(p * p * sizeof(float));
    Out = (float *)malloc(q * q * sizeof(float));
    


    MatrixInit(M, n, n);
    MatrixInit(Kernel, p, p);
    MatrixInit(Out, q, q);



    cudaMalloc((void **)&d_M, n * n * sizeof(float));
    cudaMalloc((void **)&d_Kernel, p * p * sizeof(float));
    cudaMalloc((void **)&d_Out, q * q * sizeof(float));




    cudaMemcpy(d_M, M, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kernel, Kernel, p * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Out, Out, q * q * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMatrixConvolution<<<n, p>>>(d_M, d_Kernel, d_Out, n, p, q);


    cudaElementMatrixMult<<n, n>>(d_M, d_Kernel , d_Out, n);
     

     

    cudaMemcpy(M, d_M, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaElementMatrixSum<< n >>(d_Out, d_sum, n);
    cudaMemcpy(Out, d_Out, p * p * sizeof(float), cudaMemcpyDeviceToHost);



    // cudaMemcpy(Out, d_Out, q * q * sizeof(float), cudaMemcpyDeviceToHost);


    // print the name 
    printf("M\n");
    MatrixPrint(M, n, n);


    printf("Kernel\n");
    MatrixPrint(Kernel, p, p);


    printf("Out\n");
    MatrixPrint(Out, q, q);

    free(M);

    cudaFree(d_M);
    cudaFree(d_Kernel);
    cudaFree(d_Out);
    return 0;
}







