#include <stdio.h>
#include <stdlib.h>
#include <time.h>


////// 1) Une matrice float data de taille 1x32x32 initialisé à 0 qui prendra les valeurs de l'image d'entrée.


float* generateRandomMatrix(int rows, int cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX; // Generate random value between 0 and 1
    }
    return matrix;
}

int main() {
    int rows = 32;
    int cols = 32;
    int channels = 1;

    float* raw_data = generateRandomMatrix(rows * cols * channels, 1);
    free(raw_data);
    return 0;
}

int matrix_test(){
    
    int n = 1500, p = 1500;
    float *M1, *M2, *Mout;

    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
}





////// 2) Une matrice float C1_data de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. 
////// C1 correspond aux données après la première Convolution.




////// 3) Une matrice float S1_data de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du sous-échantillonnage. 
////// S1 correspond aux données après le premier Sous-échantillonnage.







////// 4) Une matrice float C1_kernel de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.

