#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
//#include <cuchar>
#include <ctime>
#include <bitset>
#include <openacc.h>
#include <accelmath.h>
#include <cmath>
#include <curand.h>
//#include <random>
//#include <curand_kernel.h>
//#include <math.h>
//using namespace cv;
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;

#define DEPSILON 0.5E-15
/*
extern void run_optimal(double *, double *, double *, double *, int, int, int, int);
extern void run_w2kernel(double *);
extern void testpr();
*/
typedef unsigned char uchar;

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size, int& nrows, int& ncols) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;
        nrows = n_rows;
        ncols = n_cols;
        
        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

uchar* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

/*
void reverseInt(uchar i) {
    //std::bitset<32> x(i);
    //cout << "Bitset: " << x;
    //cout << " | i: " << (unsigned int)(i) << endl;
    //unsigned char c1, c2, c3, c4;
    //c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    //return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    //return i;
}
*/

float** convert_train_dataset_f(uchar **dataset, int num_images, int num_rows, int num_cols) {
    float** _cdata = new float*[num_images];
    
    for(int k = 0; k < num_images; k++) {
        _cdata[k] = new float[num_rows * num_cols];
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                _cdata[k][28*i + j] = (float)(dataset[k][28*i+j]);
            }
        }
    }
    
    return _cdata;
}

double *data2mono_orig(uchar **data, int num_images, int img_size) {
    double *_data = new double[num_images * img_size];
    
    for(int i = 0; i < num_images; i++) {
        for(int k = 0; k < img_size; k++) {
            _data[i * img_size + k] = (double)(data[i][k]);
        }
    }
    
    return _data;
}

double *data2mono(uchar **data, int num_images, int img_size, double dml) {
    double *_data = new double[num_images * img_size];
    
    for(int i = 0; i < num_images; i++) {
        for(int k = 0; k < img_size; k++) {
            _data[i * img_size + k] = dml * ((double)(data[i][k] - 255.0f/2))/255.0f;
        }
    }
    
    return _data;
}

double *labels1hot(uchar *labels, int num_images, int num_labels) {
    double *_labels1hot = new double[num_images * num_labels];
    
    for(int i = 0; i < num_images; i++) {
        for(int l = 0; l < num_labels; l++) {
            if((int)labels[i] == l) {
                _labels1hot[i*num_labels + l] = 1.0f;
            }
            else {
                _labels1hot[i*num_labels + l] = 0.0f;
            }
        }
    }
    
    return _labels1hot;
}

void param2file_al(double *param, string path, string param_name, int num_param ) {
    ofstream out_param;
    out_param.open("NEURAL_NETWORK_TRAINED.xml", ios::out | ios::app);
    
    if( out_param.is_open() ) {
        //out_param << num_param << endl;
        out_param << "<" << param_name << ">" << endl;
        
        for(int i = 0; i < num_param; i++) {
            out_param << "<item>" << param[i] << "</item>" << endl;
            //if(i< num_images-1) { out_labels << endl; }
        }
        
        out_param << "</" << param_name << ">" << endl;
        
    }
    
    out_param.close();
}



// void rngnormalf(int n, float *nr) {
//     //float *a;
//     int i, istat;
//     curandGenerator_t g;
// 
//     //a = (float *) malloc(n*4);
//     #pragma acc parallel loop
//     for (i = 0; i < n; i++)
//         nr[i] = 0.0f;
//     istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
//     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
// 
//     /* Now Normal */
//     printf("Should be normal around 0.0\n");
//     istat = curandGenerateNormal(g, nr, n, 0.0f, 1.0f);
//     if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
//     float mx = 0, mn = 0;
//     
//     //#pragma acc parallel loop reduction(max:mx) reduction(min:mn)
//     for(int i = 0; i < n; i++) {
//         //cout << "i: "<< a[i] << endl;
//         if(nr[i] - mx > 0) mx = nr[i];
//         if(nr[i] - mn < 0) mn = nr[i];
//     }
//     
//     float range = mx - mn;
//     float ml = 0.2f / range;
//     
//     cout << "Mu; " << ml << endl;
//     cout << "Max: " << mx << ", Min: " << mn << endl;
// 
//     //#pragma acc parallel loop
//     for (i = 0; i < n; i++)
//         nr[i] = nr[i] * ml;
//     
//     float mx2 = 0, mn2 = 0;
//     
//     //#pragma acc parallel loop reduction(max:mx2) reduction(min:mn2)
//     for(int i = 0; i < n; i++) {
//         //cout << "i: "<< a[i] << endl;
//         if(nr[i] - mx2 > 0) mx2 = nr[i];
//         if(nr[i] - mn2 < 0) mn2 = nr[i];
//     }
//     cout << "New Max: " << mx2 << ", New Min: " << mn2 << endl;
//     
//     
//     istat = curandDestroyGenerator(g);
// 
// }

void printrr(double *, int, int);

template<class T>
void rngnormal(T *inp, int n, T multiplier) {
    int i, istat;
    curandGenerator_t g;
    int pn = n;
    
    if(pn%2 == 1) pn = pn+1;
    
    float *nr = new float[pn];
    T *nrT = new T[pn];
    
    #pragma acc parallel loop
    for (i = 0; i < pn; i++)
        nr[i] = 0.0f;
    
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
//     cout << "N: " << n << " pn: " << pn << " m: " << multiplier << " | " << inp << endl;
    
    /* Now Normal */
    istat = curandGenerateNormal(g, nr, pn, 0.0f, 1.0f);
    
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
    for(int i = 0; i <pn; i++) {
        nrT[i] = nr[i];
    }
    
    T mx = 0, mn = nrT[0];
    
    #pragma acc parallel loop reduction(max:mx) reduction(min:mn)
    for(int i = 0; i < pn; i++) {
        if(nrT[i] > mx) mx = nrT[i];
        if(nrT[i] < mn) mn = nrT[i];
    }
    
    T range = mx - mn;
    T ml = 2.0f * multiplier / range;
    
//     cout << "Max: " << mx << ", Min: " << mn << ", Range: " << range << ", Ml: " << ml << endl;

    #pragma acc parallel loop
    for (i = 0; i < pn; i++)
        nrT[i] = nrT[i] * ml;
    
    T mx2 = 0, mn2 = 0;
    
    #pragma acc parallel loop reduction(max:mx2) reduction(min:mn2)
    for(int i = 0; i < pn; i++) {
        //cout << "i: "<< a[i] << endl;
        if(nrT[i] > mx2) mx2 = nrT[i];
        if(nrT[i] < mn2) mn2 = nrT[i];
    }
//     cout << "New Max: " << mx2 << ", New Min: " << mn2 << endl;
    
    #pragma acc parallel loop
    for(int i = 0; i < n; i++) {
        inp[i] = nrT[i];
    }
    istat = curandDestroyGenerator(g);

}

void acc_update_self(double *a, int asize) {
    #pragma acc update self(a[:asize])
}

void transpose(double *A, double *A_tr, int NN, int MM) {
    #pragma acc data pcopyin(A[:NN*MM]) pcopyout(A_tr[:NN*MM])
    {
    #pragma acc parallel loop
    for(int n = 0; n < NN; n++) {
        for(int m = 0; m < MM; m++) {
            A_tr[m*NN + n] = A[n*MM + m];
        }
    }
    }
}

template<class T>
void zeros(T *arr, int asize) {
    #pragma acc parallel loop pcopyout(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] = 0.0f;
    }
}

/*
void zerosf(float *arr, int asize) {
    #pragma acc parallel loop pcopyout(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] = 0.0f;
    }
}

void zeros(double *arr, int asize) {
    #pragma acc parallel loop pcopy(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] = 0.0f;
    }
}
*/

void emltp(double *arr, double mltp, int asize) {
    #pragma acc parallel loop pcopy(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] *= mltp;
    }
}

double *makeweights(int N, double multp) {
    double *retw = new double[N];
    
    rngnormal<double>(retw, N, multp);
    
    return retw;
}

double *makebiases(int N, double initl) {
    double *retb = new double[N];
    
    #pragma acc parallel loop pcopyout(retb[:N])
    for(int i = 0; i < N; i++) {
        retb[i] = initl;
    }
    
    return retb;
}

#define BS 32
/*
 * Name: tparallel_matrix_multiply
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the operation C = A * B, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 * }
 *
*/
 void tparallel_matrix_multiply(double  * A, double * B, double * C, int N, int K, int M) {
     
    
    #pragma acc data pcopyin(A[:(N*K)], B[0:K*M]) pcopyout(C[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            double csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            C[i*M + j] = csumd;
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_multiply_ex
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the operation C = A * B, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 * }
 *
*/
 void tparallel_matrix_multiply_ex(double  * A, double * B, double * C, int N, int K, int M) {
     
    
    #pragma acc data pcopyin(A[:(N*K)], B[0:K*M]) pcopyout(C[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            double csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            
            C[i*M + j] = csumd;
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_multiply_mltp
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C' multiplied by a given scalar multiplier 'mlt'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the operation C = A * B * mlt, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 *      mlt: scalar multiplier
 * }
 *
*/
void tparallel_matrix_multiply_mltp(double  * A, double * B, double * C, int N, int K, int M, double mlt) {
    #pragma acc data pcopyin(A[:(N*K)], B[0:K*M]) pcopyout(C[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            double csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            C[i*M + j] = csumd * mlt;
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_multiply_hadamard
 * Description: Performs the matrix multiplication 'A * B' and afterwards a hadamard product '(A * B) * C' and stores the result into matrix 'H'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the second operand in the hadamard product, must be 1-dimensional
 *      H: N x M size matrix containing the results of the hadamard product, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 * }
 *
*/
void tparallel_matrix_multiply_hadamard(double  * A, double * B, double * C, double *H, int N, int K, int M) {
    #pragma acc data pcopyin(A[:(N*K)], B[0:K*M]) pcopyout(C[0:N*M]) pcopyin(H[:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            double csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            H[i*M + j] = csumd * C[i*M + j];
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_hadamard
 * Description: Performs the Hadamard product 'A * B' and stores the result into matrix 'H'
 * Parameters: {
 *      A: N x M size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: N x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      H: N x M size matrix containing the elements of the Hadamard product operation H = A * B, must be 1-dimensional
 *      N: number of rows of matrix A, B, H
 *      M: number of columns of matrix A, B, H
 * }
 *
*/
void tparallel_matrix_hadamard(double  * A, double * B, double *H, int N, int M) {
    #pragma acc data pcopyin(A[:(N*M)], B[0:N*M]) pcopyout(H[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            H[i*M + j] = A[i*M + j] * B[i*M + j];
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_add_row
 * Description: Adds the 1xM vector B horizontally to N rows of matrix A
 * Example:  A         B
 *        -------   -------
 *        |1 2 3|             |2 4  3|
 *        |4 5 6| + |1 2 0| = |5 7  6|
 *        |7 8 9|             |8 10 9|
 * 
 * Parameters: {
 *      A: N x M size matrix containing the elements of the first operand, must be 1-dimensional
 *      B: 1 x M size vector containing the elements of the second operand, must be 1-dimensional
 *      N: number of rows of matrix A
 *      M: number of columns of matrix A, B
 * }
 *
*/
void tparallel_matrix_add_row(double *A, double *B, int N, int M) {
    #pragma acc data pcopy(A[0:N*M]) pcopyin(B[:M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {            
            A[i*M + j] += B[j]; 
        }
    }
    }
}

void tparallel_matrix_add_depthwise(double *A, double *B, int N, int K, int D) {
    #pragma acc data pcopy(A[0:N*K*D]) pcopyin(B[:D])
    {
    
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < K; k++) {
            for(int d = 0; d < D; d++) {
                A[i*K*D + k*D + d] += B[d];
            }
        }
    }
    }
}

/*
 * Name: tparallel_softmax
 * Description: Performs a softmax function on the input and stores the result to the output
 *
 * Parameters: {
 *     softmax_output: N x M size matrix containing the output of the softmax operation
 *     softmax_input: N x M size matrix containing the input to the softmax operation
 *     softmax_output_sum: auxiliary placeholder containing the calculated sum of the softmax denominator
 * }
 *
*/
void tparallel_softmax(double *softmax_output, double *softmax_input, double *softmax_output_sum, int batch_size, int num_outputs) {
    #pragma acc data pcopyin(softmax_input[:batch_size*num_outputs]) pcopyout(softmax_output[:batch_size*num_outputs]) pcopyout(softmax_output_sum[:batch_size])
    {

      
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        softmax_output_sum[i] = 0.0f;
    }
    
     
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        double sumd = 0.0f;
        #pragma acc loop reduction(+:sumd)
        for(int j = 0; j < num_outputs; j++) {
            sumd += exp(softmax_input[i*num_outputs + j]);
//             #pragma acc atomic update
//             softmax_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
        }
        
        softmax_output_sum[i] = sumd;
    }
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_outputs; j++) {            
            #pragma acc atomic write
            softmax_output[i*num_outputs + j] = exp(softmax_input[i*num_outputs + j]) / softmax_output_sum[i];
        }
    }
    }
}

/*
 * Name: tparallel_softmax
 * Description: Performs a softmax function on the input and stores the result to the output
 *
 * Parameters: {
 *     softmax_output: N x M size matrix containing the output of the softmax operation
 *     softmax_input: N x M size matrix containing the input to the softmax operation
 *     softmax_output_sum: auxiliary placeholder containing the calculated sum of the softmax denominator
 * }
 *
*/
void tparallel_softmax_wdrv(double *softmax_output, double *softmax_output_drv, double *softmax_input, double *softmax_output_sum, int batch_size, int num_outputs) {
    #pragma acc data pcopyin(softmax_input[:batch_size*num_outputs]) pcopyout(softmax_output[:batch_size*num_outputs], softmax_output_drv[:batch_size*num_outputs]) pcreate(softmax_output_sum[:batch_size])
    {

      
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        softmax_output_sum[i] = 0.0f;
    }
    
     
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        double sumd = 0.0f;
        #pragma acc loop reduction(+:sumd)
        for(int j = 0; j < num_outputs; j++) {
            sumd += exp(softmax_input[i*num_outputs + j]);
//             #pragma acc atomic update
//             softmax_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
        }
        
        softmax_output_sum[i] = sumd;
    }
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_outputs; j++) {            
            #pragma acc atomic write
            softmax_output[i*num_outputs + j] = exp(softmax_input[i*num_outputs + j]) / softmax_output_sum[i];
             
        }
    }
    }
}

/*
 * Name: tparallel_relu
 * Description: Performs a 'Linear rectified Unit' - ReLU function on the input and stores the result to the output
 *
 * Parameters: {
 *     relu_output: N x M size matrix containing the output of the ReLU operation
*      relu_output_drv: N x M size matrix containing the derivative of the output of the ReLU operation
 *     rl_input: N x M size matrix containing the input to the ReLU operation
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
void tparallel_relu(double *relu_output, double *relu_output_drv, double *rl_input, int batch_size, int num_outputs) {    
    #pragma acc data pcopyin(rl_input[:batch_size*num_outputs]) pcopyout(relu_output[:batch_size*num_outputs], relu_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = rl_input[i*num_outputs + j];
   
            if(val > 0) {
                relu_output[i*num_outputs + j] = val;
                relu_output_drv[i*num_outputs + j] = 1.0f;
            }
            else {
                relu_output[i*num_outputs + j] = 0.0f;
                relu_output_drv[i*num_outputs + j] = 0.0f;
            }
        }
    }
    }
}

/*
 * Name: tparallel_sigmoid
 * Description: Performs a Sigmoid - Logistic function on the input and stores the result to the output
 *
 * Parameters: {
 *     sigm_output: N x M size matrix containing the output of the sigmoid function
*      sigm_output_drv: N x M size matrix containing the derivative of the output of the sigmoid function
 *     sg_input: N x M size matrix containing the input to the sigmoid function
 * }
 * 
 * Example:
 *          |-4 5 2|               |0.0180 0.9933 0.8808|
 *      A = |0 1 -1|, Sigmoid(A) = |0.5000 0.7310 0.2689|
 *          |6 -8 3|               |0.9975 0.0003 0.9526|
*/
void tparallel_sigmoid(double *sigm_output, double *sigm_output_drv, double *sg_input, int batch_size, int num_outputs) {    
    #pragma acc data pcopyin(sg_input[:batch_size*num_outputs]) pcopyout(sigm_output[:batch_size*num_outputs], sigm_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = 1/(1 + exp(-1 * sg_input[i*num_outputs + j]));

            sigm_output[i*num_outputs + j] = val;
            sigm_output_drv[i*num_outputs + j] = val*(1-val);
        }
    }
    }
}

/*
 * Name: tparallel_conv5
 * Description: Performs a convolution on the input 
 *
 * Parameters: {
 *      conv_input: BxCxNxM input matrix where
 *          B: batch size [B]
 *          C: input channels [in_channels]
 *          N: number of rows [in_height]
 *          M: number of columns [in_width]
 *      
 *      conv_filters: DxCxFxF convolution filters where
 *          D: number of output channels [depth]
 *          C: number of input channels [in_channels]
 *          F: filter size width/height [filter_size]
 *      
 *      conv_output: BxDxQxW output matrix where
 *          B: batch_size [B]
 *          D: output channels [depth]
 *          Q: number of rows [out_height]
 *          W: number of columns [out_width]
 *      
 *      stride: number of horizontal/vertical displacement during each convolution step
 * }
 * 
 * Example:
*/
void tparallel_conv5(double *conv_input, double *conv_filters, double *conv_output, int batch_size, int in_channels, int in_height, int in_width, int out_channels , int out_height, int out_width, int filter_size, int stride, bool debug) { 
    
    
    #pragma acc data pcopyin(conv_input[:(batch_size*in_width*in_height*in_channels)], conv_filters[:in_channels*filter_size*filter_size*out_channels]) pcopyout(conv_output[:(batch_size * out_width * out_height * out_channels)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int i = 0; i < batch_size; i++) {
        for(int d = 0; d < out_channels; d++) {
            for(int oh = 0; oh < out_height; oh++) {
                for(int ow = 0; ow < out_width; ow++) {
                    double bdhwsum = 0.0f;
                    
                    #pragma acc loop seq collapse(3) reduction(+:bdhwsum)
                    for(int ch = 0; ch < in_channels; ch++) {
//                         double csum = 0.0f;
                        for(int di = 0; di < filter_size; di++) {
                            for(int dj = 0; dj < filter_size; dj++) {
                                bdhwsum += conv_input[ (i*in_channels + ch)*in_height*in_width + (oh + di)*in_width + ow + dj ] * conv_filters[ (d*in_channels + ch)*filter_size*filter_size + di*filter_size + dj ];
                            }
                        }
//                         sum += csum;
                    }
                    
                    conv_output[(i*out_channels + d)*out_width*out_height + oh*out_width + ow] = bdhwsum;
                }
            }
        }
    }
    
    }
    
}

// Calculate output sizes of a convolution operation depending on input size, desired output size, filter size, stride
void calc_conv_padding(int in_height, int in_width, int filter_size, int stride, int out_height, int out_width, int &padding_left, int &padding_right, int &padding_top, int &padding_bottom) {
        
    int padding_height, padding_width;
    /*
    if(padding) {
        if(in_height%stride == 0) {
            padding_height = max(filter_size - stride, 0);
        }
        else {
            padding_height = max(filter_size - (in_height%stride), 0);
        }

        if(in_width%stride == 0) {
            padding_width = max(filter_size - stride, 0);
        }
        else {
            padding_width = max(filter_size - (in_width%stride), 0);
        }
        
        padding_top = padding_height/2;
        padding_bottom = padding_height - padding_top;
        padding_left = padding_width/2;
        padding_right = padding_width - padding_left;
        
        out_width = ceil((1.0f * in_width)/stride);
        out_height = ceil((1.0f * in_height)/stride);
        
    }
    else {*/
    out_width = ceil( (1.0f * (in_width - filter_size + 1))/stride );
    out_height = ceil( (1.0f * (in_height - filter_size + 1))/stride );
    
    printf("In: %d x %d, f: %d, stride: %d, Out: %d x %d\n", in_width, in_height, filter_size, stride, out_width, out_height);
        
  
}

//TODO review ? find corrent function, in dimensions, stride, same/full/valid padding?
// Calculate output sizes of a convolution operation depending on input size, filter size, stride
void calc_conv_sizes(int in_height, int in_width, int filter_size, int stride, bool padding, int &padding_left, int &padding_right, int &padding_top, int &padding_bottom, int &out_height, int &out_width) {
        
    int padding_height ,padding_width;
    
    if(padding) {
        if(in_height%stride == 0) {
            padding_height = max(filter_size - stride, 0);
        }
        else {
            padding_height = max(filter_size - (in_height%stride), 0);
        }

        if(in_width%stride == 0) {
            padding_width = max(filter_size - stride, 0);
        }
        else {
            padding_width = max(filter_size - (in_width%stride), 0);
        }
        
        padding_top = padding_height/2;
        padding_bottom = padding_height - padding_top;
        padding_left = padding_width/2;
        padding_right = padding_width - padding_left;
        
        out_width = ceil((1.0f * in_width)/stride);
        out_height = ceil((1.0f * in_height)/stride);
        
    }
    else {
        padding_width = 0;
        padding_height = 0;
        padding_top = 0;
        padding_bottom = 0;
        padding_left = 0;
        padding_right = 0;
        
        out_width = ceil( (1.0f * (in_width - filter_size + 1))/stride );
        out_height = ceil( (1.0f * (in_height - filter_size + 1))/stride );
    }
}

void make_batch(double * inputs, double *batch, int batch_size, int input_size, int batch_start) {
    #pragma acc data pcopyin(inputs[(batch_start*input_size):(batch_size*input_size)]) pcopy(batch[:(batch_size*input_size)])
    {
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < input_size; k++) {
            batch[i*input_size + k] = inputs[(i+batch_start)*input_size + k];
        }
    }
        
    }
}

/*
 * Name: pad2D
 * Description: Zero-pads a 2-dimensional array by a specified amount of pixels in each direction 
 *
 * Parameters: {
 *      input: BxDxNxM input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padded_input: BxDxNxM padded input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padding_left: number of pixels to pad to the left
 *      padding_right: number of pixels to pad to the right
 *      padding_top: number of pixels to pad to the top
 *      padding_bottom: number of pixels to pad to the bottom
 *      
 *      N: number of rows of original input
 *      M: number of columns of original input
 *      padded_N: number of rows after padding
 *      padded_M: number of columns after padding
 * }
 * 
 * Example:
*/
void pad2D(double *input, double *padded_input, int B, int C, int N, int M, int padding_left, int padding_right, int padding_top, int padding_bottom) {
    int padded_N = N + padding_top + padding_bottom;
    int padded_M = M + padding_left + padding_right;
    
    #pragma acc data pcopyin(input[:(B*C*M*N)]) pcopy(padded_input[:(B*C*padded_N*padded_M)])
    {
        
    #pragma acc parallel loop collapse(4) 
    for(int b = 0; b < B; b++) {
        for(int c = 0; c < C; c++) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    padded_input[(b*C + c)*padded_N*padded_M + (i+padding_top)*padded_M + j + padding_left] = input[b*C*M*N + c*M*N + i*M + j];
                }
            }
        }
    }
    
    }
}

/*
 * Name: backprop_fc
 * Description: Calculates the error derivatives for weights,biases and propagate to  layer inputs
 *
 * Parameters: {
 *      input: BxDxNxM input matrix where
 *          B: batch size [B]
 *          D: input depth [D]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padded_input: BxDxNxM padded input matrix where
 *          B: batch size [B]
 *          D: input depth [D]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padding_left: number of pixels to pad to the left
 *      padding_right: number of pixels to pad to the right
 *      padding_top: number of pixels to pad to the top
 *      padding_bottom: number of pixels to pad to the bottom
 *      
 *      N: number of rows of original input
 *      M: number of columns of original input
 *      padded_N: number of rows after padding
 *      padded_M: number of columns after padding
 * }
 * 
 * Example: *prev_layer_drv_loss_output, *prev_layer_drv_loss_input, *cur_layer_drv_loss_input, *prev_layer_out, *prev_layer_out_drv 
 *          drv_loss_output_hidden3, drv_loss_input_hidden3, drv_loss_input_output4, train_out_hidden3, train_out_hidden3_drv
*/
void backprop_fc(double *drv_loss_weights, double *drv_loss_biases, double *prev_layer_drv_loss_output, double *prev_layer_drv_loss_input, double *cur_layer_drv_loss_input, double *prev_layer_out, double *prev_layer_out_drv, double *weights, int prev_layer_size, int cur_layer_size, int batch_size, bool debug) {
        
    double *prev_layer_out_transpose = new double[batch_size*prev_layer_size];
    double *weights_transpose = new double[prev_layer_size*cur_layer_size];
    
    #pragma acc data \
        pcopyout(drv_loss_weights[:prev_layer_size*cur_layer_size], drv_loss_biases[:cur_layer_size], prev_layer_drv_loss_input[:batch_size*prev_layer_size], prev_layer_drv_loss_output[:batch_size*prev_layer_size])\
        pcopyin(cur_layer_drv_loss_input[:batch_size*cur_layer_size], prev_layer_out[:batch_size*prev_layer_size], prev_layer_out_drv[:batch_size*prev_layer_size], weights[:prev_layer_size*cur_layer_size]) \
        pcreate(prev_layer_out_transpose[:batch_size*prev_layer_size], weights_transpose[:prev_layer_size*cur_layer_size])
    {
        
    // error drv biases 
    #pragma acc parallel loop
    for(int m = 0; m < cur_layer_size; m++) {
        double sumdb = 0.0f;
        
        #pragma acc loop reduction(+:sumdb)
        for(int i = 0; i < batch_size; i++) {
            sumdb += cur_layer_drv_loss_input[i*cur_layer_size + m];
        }
        
        drv_loss_biases[m] = sumdb;
    }
    
    // error drv weights
    // dE/dW = dE/dInput * dInput/dW
    transpose(prev_layer_out, prev_layer_out_transpose, batch_size, prev_layer_size);
    tparallel_matrix_multiply(prev_layer_out_transpose, cur_layer_drv_loss_input, drv_loss_weights, prev_layer_size, batch_size, cur_layer_size);

    // error drv input layer 0
    transpose(weights, weights_transpose, prev_layer_size, cur_layer_size);
    tparallel_matrix_multiply(cur_layer_drv_loss_input, weights_transpose, prev_layer_drv_loss_output, batch_size, cur_layer_size, prev_layer_size);    
    tparallel_matrix_hadamard(prev_layer_drv_loss_output, prev_layer_out_drv, prev_layer_drv_loss_input, batch_size, prev_layer_size);

    }
    
    free(weights_transpose);
    free(prev_layer_out_transpose);
    
}

void backprop_conv_weights(double *drv_loss_weights, double *drv_loss_biases, double *conv_input, double *cur_layer_drv_loss_input, int batch_size, int in_channels, int in_height, int in_width, int out_channels, int out_width, int out_height, int filter_size, int padded_height, int padded_width, int stride, bool debug ) {
    
    int out_size = out_height * out_width;
    int bs =  batch_size;
    int ich = in_channels;
    int ps = padded_width * padded_height;
    int och = out_channels;
    
    double *drv_loss_weights_transform = new double[(out_channels*in_channels*filter_size*filter_size)];
    double *conv_input_transform = new double[(batch_size * in_channels * ps)];
    double *cur_layer_drv_loss_input_transform = new double[(batch_size * out_channels * out_size)];
    
    #pragma acc data \
        pcopyout(drv_loss_weights[:(out_channels*in_channels*filter_size*filter_size)], drv_loss_biases[:out_channels]) \
        pcopyin(conv_input[:(batch_size * in_channels * ps)]) \
        pcopyin(cur_layer_drv_loss_input[:(batch_size * out_channels * out_size)]) \
        pcopyin(drv_loss_weights_transform[:(out_channels*in_channels*filter_size*filter_size)], conv_input_transform[:(batch_size * in_channels * ps)], cur_layer_drv_loss_input_transform[:(batch_size * out_channels * out_size)])
    {

    // conv in B C S -> conv in transform C B S
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < bs; i++) {
        for(int ch = 0; ch < ich; ch++) {
            for(int s = 0; s < ps; s++) {
                conv_input_transform[(ch*bs+ i)*ps + s] = conv_input[(i*ich + ch)*ps + s];
            }
        }
    }
    
    // delta in  B D S2 -> delta in transform D B S2
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < bs; i++) {
        for(int d = 0; d < och; d++) {
            for(int s = 0; s < out_size; s++) {
                cur_layer_drv_loss_input_transform[(d*bs + i)*out_size + s] = cur_layer_drv_loss_input[(i*och + d)*out_size + s];
            }
        }
    }
    
    tparallel_conv5(conv_input_transform, cur_layer_drv_loss_input_transform, drv_loss_weights_transform, in_channels, batch_size, padded_height, padded_width, out_channels, filter_size, filter_size, out_width, stride, debug);
    
    // delta weights transform D C F1 F2 - > delta weights C D F1 F2
    #pragma acc parallel loop collapse(4)
    for(int ch = 0; ch < ich; ch++) {
        for(int d = 0; d < och; d++) {
            for(int f1 = 0; f1 < filter_size; f1++) {
                for(int f2 = 0; f2 < filter_size; f2++) {
                    drv_loss_weights[(d*ich + ch)*filter_size*filter_size + f1*filter_size + f2] = drv_loss_weights_transform[ (ch*och + d)*filter_size*filter_size + f1*filter_size + f2];
                }
            }
        }
    }
    
    #pragma acc parallel loop
    for(int d = 0; d < out_channels; d++) {
        double sumdb = 0.0f;
        
        #pragma acc loop seq reduction(+:sumdb)
        for(int i = 0; i < batch_size; i++) {
            double sumdbi = 0.0f;
            
            #pragma acc loop reduction(+:sumdbi)
            for(int c = 0; c < out_size; c++) {
                sumdbi += cur_layer_drv_loss_input[ (i*out_channels + d)*out_size + c];
            }
            
            sumdb += sumdbi;
        }
        
        drv_loss_biases[d] = sumdb;
        
    }
    
    }
    
    free(drv_loss_weights_transform);
    free(conv_input_transform);
    free(cur_layer_drv_loss_input_transform);
}

void backprop_conv_input(double *prev_layer_drv_loss_output, double *prev_layer_drv_loss_input, double *cur_layer_drv_loss_input, double *weights, double *prev_layer_out_drv, int batch_size, int pre_conv_channels, int pre_conv_height, int pre_conv_width, int post_conv_channels,  int post_conv_height, int post_conv_width, int filter_size, int stride, bool debug ) {
    
    // TODO get rid of this, use calc conv sizes too
   
    int padding_left, padding_top, padding_right, padding_bottom, out_h, out_w;
    
    calc_conv_sizes(post_conv_height, post_conv_width, filter_size, stride, true, padding_left, padding_right, padding_top, padding_bottom, out_h, out_w);
    int padded_post_conv_height = post_conv_height + padding_top + padding_bottom, padded_post_conv_width = post_conv_width + padding_left + padding_right;
    
    int bs =  batch_size;
    int ich = pre_conv_channels;
    int och = post_conv_channels;
    
    double *weights_transform = new double[pre_conv_channels * post_conv_channels * filter_size * filter_size];
    double *cur_layer_drv_loss_input_padded = new double[batch_size * post_conv_channels * padded_post_conv_height * padded_post_conv_width];
    
    #pragma acc data\
        pcopyout(prev_layer_drv_loss_output[:(batch_size*pre_conv_channels*pre_conv_height*pre_conv_width)], prev_layer_drv_loss_input[:(batch_size*pre_conv_channels*pre_conv_height*pre_conv_width)]) \
        pcopyin(cur_layer_drv_loss_input[:(batch_size*post_conv_channels*post_conv_width*post_conv_height)], weights[:(post_conv_channels*pre_conv_channels*filter_size*filter_size)], prev_layer_out_drv[:(batch_size*pre_conv_channels*pre_conv_width*pre_conv_height)]) \
        pcreate(weights_transform[:(post_conv_channels*pre_conv_channels*filter_size*filter_size)], cur_layer_drv_loss_input_padded[:(batch_size * post_conv_channels * padded_post_conv_height * padded_post_conv_width)])
    {
        
    // pad out delta
    pad2D(cur_layer_drv_loss_input, cur_layer_drv_loss_input_padded, batch_size, post_conv_channels, post_conv_height, post_conv_width, padding_left, padding_right, padding_top, padding_bottom);
    
    // flip weights x,y , transpose D-C    
    #pragma acc parallel loop collapse(4)
    for(int d = 0; d < och; d++) {
        for(int ch = 0; ch < ich; ch++) {
            for(int f1 = 0; f1 < filter_size; f1++) {
                for(int f2 = 0; f2 < filter_size; f2++) {
                    weights_transform[(ch*post_conv_channels + d)*filter_size * filter_size + (filter_size - 1 - f1)*filter_size + filter_size - 1 - f2] = weights[(d*pre_conv_channels + ch)*filter_size * filter_size + f1*filter_size + f2];
                }
            }
        }
    }
    
    tparallel_conv5(cur_layer_drv_loss_input_padded, weights_transform, prev_layer_drv_loss_output, batch_size, post_conv_channels, padded_post_conv_height, padded_post_conv_width, pre_conv_channels, pre_conv_height, pre_conv_width, filter_size, stride, false);
    
    tparallel_matrix_hadamard(prev_layer_drv_loss_output, prev_layer_out_drv, prev_layer_drv_loss_input, batch_size, pre_conv_channels*pre_conv_width*pre_conv_height);
    
    }
    
    free(weights_transform);
    free(cur_layer_drv_loss_input_padded);
}


void backprop_update(double *a, double *da, int asize, double learning_rate, int batch_size) {
    double mltp = (1.0f/(double)batch_size);
    #pragma acc parallel loop pcopyin(da[:asize]) pcopy(a[:asize])
    for(int i = 0; i < asize; i++) {
        a[i] -= mltp * learning_rate * da[i];
    }
}

void param2file_csv(double *param, string path, int param_typ, int num_param, int layrn ) {
    ofstream out_param;
    out_param.open(path, ios::out | ios::app);
    
    if( out_param.is_open() ) {
        //out_param << num_param << endl;
        out_param << layrn << ";" << param_typ << ";" << num_param << endl;
        
        for(int i = 0; i < num_param; i++) {
            out_param << param[i] << endl;
            //if(i< num_images-1) { out_labels << endl; }
        }
    }
    
    out_param.close();
}


/*
 * Name: acc_matrix_multiply
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the operation C = A * B, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 * }
 *
*/
 void acc_matrix_multiply(double  * A, double * B, double * C, int N, int K, int M) {
     
    
    #pragma acc data present(A[:(N*K)], B[0:K*M]) present(C[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            double csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            C[i*M + j] = csumd;
        }
    }
    
    }
}

/*
 * Name: acc_convolution
 * Description: Performs a convolution on the input 
 *
 * Parameters: {
 *      conv_input: BxCxNxM input matrix where
 *          B: batch size [B]
 *          C: input channels [in_channels]
 *          N: number of rows [in_height]
 *          M: number of columns [in_width]
 *      
 *      conv_filters: DxCxFxF convolution filters where
 *          D: number of output channels [depth]
 *          C: number of input channels [in_channels]
 *          F: filter size width/height [filter_size]
 *      
 *      conv_output: BxDxQxW output matrix where
 *          B: batch_size [B]
 *          D: output channels [depth]
 *          Q: number of rows [out_height]
 *          W: number of columns [out_width]
 *      
 *      stride: number of horizontal/vertical displacement during each convolution step
 * }
 * 
 * Example: 
 * TODO cross-correlation!
*/
void acc_convolution(double *conv_input, double *conv_filters, double *conv_output, int batch_size, int in_channels, int in_height, int in_width, int out_channels , int out_height, int out_width, int filter_size, int stride, bool debug) { 
    
    
    #pragma acc data present(conv_input[:(batch_size*in_width*in_height*in_channels)], conv_filters[:in_channels*filter_size*filter_size*out_channels]) present(conv_output[:(batch_size * out_width * out_height * out_channels)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int i = 0; i < batch_size; i++) {
        for(int d = 0; d < out_channels; d++) {
            for(int oh = 0; oh < out_height; oh++) {
                for(int ow = 0; ow < out_width; ow++) {
                    double bdhwsum = 0.0f;
                    
                    #pragma acc loop seq collapse(3) reduction(+:bdhwsum)
                    for(int ch = 0; ch < in_channels; ch++) {
//                         double csum = 0.0f;
                        for(int di = 0; di < filter_size; di++) {
                            for(int dj = 0; dj < filter_size; dj++) {
                                bdhwsum += conv_input[ (i*in_channels + ch)*in_height*in_width + (oh + di)*in_width + ow + dj ] * conv_filters[ (d*in_channels + ch)*filter_size*filter_size + di*filter_size + dj ];
                            }
                        }
//                         sum += csum;
                    }
                    
                    conv_output[(i*out_channels + d)*out_width*out_height + oh*out_width + ow] = bdhwsum;
                }
            }
        }
    }
    
    }
    
}




void acc_convolution3D(double *conv_input, double *conv_filters, double *conv_output, int batch_size, int in_channels, int in_height, int in_width, int out_channels , int out_height, int out_width, int filter_size, int stride, bool debug) { 
    
    
    #pragma acc data present(conv_input[:(batch_size*in_width*in_height*in_channels)], conv_filters[:in_channels*filter_size*filter_size*out_channels]) present(conv_output[:(batch_size * out_width * out_height * out_channels)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int i = 0; i < batch_size; i++) {
        for(int d = 0; d < out_channels; d++) {
            for(int oh = 0; oh < out_height; oh++) {
                for(int ow = 0; ow < out_width; ow++) {
                    double bdhwsum = 0.0f;
                    
                    #pragma acc loop seq collapse(3) reduction(+:bdhwsum)
                    for(int ch = 0; ch < in_channels; ch++) {
//                         double csum = 0.0f;
                        for(int di = 0; di < filter_size; di++) {
                            for(int dj = 0; dj < filter_size; dj++) {
                                bdhwsum += conv_input[ (i*in_channels + ch)*in_height*in_width + (oh + di)*in_width + ow + dj ] * conv_filters[ (d*in_channels + ch)*filter_size*filter_size + di*filter_size + dj ];
                            }
                        }
//                         sum += csum;
                    }
                    
                    conv_output[(i*out_channels + d)*out_width*out_height + oh*out_width + ow] = bdhwsum;
                }
            }
        }
    }
    
    }
    
}


/*
 * Name: acc_relu
 * Description: Performs a 'Linear rectified Unit' - ReLU function on the input and stores the result to the output
 *
 * Parameters: {
 *     relu_output: N x M size matrix containing the output of the ReLU operation
*      relu_output_drv: N x M size matrix containing the derivative of the output of the ReLU operation
 *     rl_input: N x M size matrix containing the input to the ReLU operation
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
void acc_relu(double *relu_output, double *relu_output_drv, double *rl_input, int batch_size, int num_outputs) {    
    #pragma acc data present(rl_input[:batch_size*num_outputs]) present(relu_output[:batch_size*num_outputs], relu_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = rl_input[i*num_outputs + j];
   
            if(val > 0) {
                relu_output[i*num_outputs + j] = val;
                relu_output_drv[i*num_outputs + j] = 1.0f;
            }
            else {
                relu_output[i*num_outputs + j] = 0.0f;
                relu_output_drv[i*num_outputs + j] = 0.0f;
            }
        }
    }
    }
}


/*
 * Name: acc_sigmoid
 * Description: Performs a Sigmoid - Logistic function on the input and stores the result to the output
 *
 * Parameters: {
 *     sigm_output: N x M size matrix containing the output of the sigmoid function
*      sigm_output_drv: N x M size matrix containing the derivative of the output of the sigmoid function
 *     sg_input: N x M size matrix containing the input to the sigmoid function
 * }
 * 
 * Example:
 *          |-4 5 2|               |0.0180 0.9933 0.8808|
 *      A = |0 1 -1|, Sigmoid(A) = |0.5000 0.7310 0.2689|
 *          |6 -8 3|               |0.9975 0.0003 0.9526|
*/
void acc_sigmoid(double *sigm_output, double *sigm_output_drv, double *sg_input, int batch_size, int num_outputs) {    
    #pragma acc data present(sg_input[:batch_size*num_outputs]) present(sigm_output[:batch_size*num_outputs], sigm_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = 1/(1 + exp(-1 * sg_input[i*num_outputs + j]));

            sigm_output[i*num_outputs + j] = val;
            sigm_output_drv[i*num_outputs + j] = val*(1-val);
        }
    }
    }
}

/*
 * Name: acc_softmax
 * Description: Performs a softmax function on the input and stores the result to the output
 *
 * Parameters: {
 *     softmax_output: N x M size matrix containing the output of the softmax operation
 *     softmax_input: N x M size matrix containing the input to the softmax operation
 *     softmax_output_sum: auxiliary placeholder containing the calculated sum of the softmax denominator
 * }
 *
*/
void acc_softmax(double *softmax_output, double *softmax_input, double *softmax_output_sum, int batch_size, int num_outputs) {
    #pragma acc data present(softmax_input[:batch_size*num_outputs]) present(softmax_output[:batch_size*num_outputs]) present(softmax_output_sum[:batch_size])
    {

      
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        softmax_output_sum[i] = 0.0f;
    }
    
     
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        double sumd = 0.0f;
        #pragma acc loop reduction(+:sumd)
        for(int j = 0; j < num_outputs; j++) {
            sumd += exp(softmax_input[i*num_outputs + j]);
//             #pragma acc atomic update
//             softmax_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
        }
        
        softmax_output_sum[i] = sumd;
    }
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_outputs; j++) {            
            #pragma acc atomic write
            softmax_output[i*num_outputs + j] = exp(softmax_input[i*num_outputs + j]) / softmax_output_sum[i];
        }
    }
    }
}

void acc_softmax2(double *softmax_output, double *softmax_input, int batch_size, int num_outputs) {
    
    double *softmax_output_sum;
    
    #pragma acc data present(softmax_input[:batch_size*num_outputs]) present(softmax_output[:batch_size*num_outputs]) create(softmax_output_sum[:batch_size])
    {

      
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        softmax_output_sum[i] = 0.0f;
    }
    
     
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        double sumd = 0.0f;
        #pragma acc loop reduction(+:sumd)
        for(int j = 0; j < num_outputs; j++) {
            sumd += exp(softmax_input[i*num_outputs + j]);
//             #pragma acc atomic update
//             softmax_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
        }
        
        softmax_output_sum[i] = sumd;
    }
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_outputs; j++) {            
            #pragma acc atomic write
            softmax_output[i*num_outputs + j] = exp(softmax_input[i*num_outputs + j]) / softmax_output_sum[i];
        }
    }
    }
}


/*
 * Name: acc_matrix_add_row
 * Description: Adds the 1xM vector B horizontally to N rows of matrix A
 * Example:  A         B
 *        -------   -------
 *        |1 2 3|             |2 4  3|
 *        |4 5 6| + |1 2 0| = |5 7  6|
 *        |7 8 9|             |8 10 9|
 * 
 * Parameters: {
 *      A: N x M size matrix containing the elements of the first operand, must be 1-dimensional
 *      B: 1 x M size vector containing the elements of the second operand, must be 1-dimensional
 *      N: number of rows of matrix A
 *      M: number of columns of matrix A, B
 * }
 *
*/
void acc_matrix_add_row(double *A, double *B, int N, int M) {
    #pragma acc data present(A[0:N*M]) present(B[:M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {            
            A[i*M + j] += B[j]; 
        }
    }
    }
}

void acc_matrix_add_depthwise(double *A, double *B, int N, int K, int D) {
    #pragma acc data present(A[0:N*K*D]) present(B[:D])
    {
    
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < K; k++) {
            for(int d = 0; d < D; d++) {
                A[i*K*D + k*D + d] += B[d];
            }
        }
    }
    }
}

/*
 * Name: acc_ad2D
 * Description: Zero-pads a 2-dimensional array by a specified amount of pixels in each direction 
 *
 * Parameters: {
 *      input: BxDxNxM input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padded_input: BxDxNxM padded input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padding_left: number of pixels to pad to the left
 *      padding_right: number of pixels to pad to the right
 *      padding_top: number of pixels to pad to the top
 *      padding_bottom: number of pixels to pad to the bottom
 *      
 *      N: number of rows of original input
 *      M: number of columns of original input
 *      padded_N: number of rows after padding
 *      padded_M: number of columns after padding
 * }
 * 
 * Example:
*/
void acc_pad2D(double *input, double *padded_input, int B, int C, int N, int M, int padding_left, int padding_right, int padding_top, int padding_bottom) {
    int padded_N = N + padding_top + padding_bottom;
    int padded_M = M + padding_left + padding_right;
    
    #pragma acc data present(input[:(B*C*M*N)]) present(padded_input[:(B*C*padded_N*padded_M)])
    {
        
    #pragma acc parallel loop collapse(4) 
    for(int b = 0; b < B; b++) {
        for(int c = 0; c < C; c++) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    padded_input[(b*C + c)*padded_N*padded_M + (i+padding_top)*padded_M + j + padding_left] = input[b*C*M*N + c*M*N + i*M + j];
                }
            }
        }
    }
    
    }
}

void acc_make_batch(double * inputs, double *batch, int batch_size, int input_size, int batch_start) {
    #pragma acc data present(inputs[(batch_start*input_size):(batch_size*input_size)]) present(batch[:(batch_size*input_size)])
    {
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < input_size; k++) {
            batch[i*input_size + k] = inputs[(i+batch_start)*input_size + k];
        }
    }
        
    }
}


template <class T>
struct Data4D {
    T *data;
    
    int n, rows, cols, depth;
    
    Data4D(int x, int y, int z, int w) : rows(x), cols(y), depth(z), n(w) {}
    Data4D(int x, int y, int z) : Data4D(x, y, z, 1) {}
    Data4D(int x, int y) : Data4D(x, y, 1) {}
    Data4D(int x) : Data4D(x, 1) {}
    Data4D() : Data4D(1) {}
    
    int getSize() {
        return rows*cols*depth*n;
    }
    
    void setShape(int a, int b, int c, int d) { 
        rows=a;
        cols=b;
        depth=c;
        n=d;
    }
    
    Data4D* setData(T *_data) {
        data=*_data;
        return this;
    }
    
    Data4D* alloc() { 
        data = new T[getSize()];
        return this;
    }

    Data4D* dealloc() {
        delete data[];
    }
    
    Data4D* accel(bool copyin) {
        if(copyin) {
            #pragma acc enter data copyin(data[:rows*cols*depth*n])
        }
        else {
            #pragma acc enter data create(data[:rows*cols*depth*n])
        }
        return this;
    }
    
    Data4D* deaccel(bool copyout) {
        if(copyout) {
            #pragma acc exit data copyout(data[:rows*cols*depth*n])
        }
        else {
            #pragma acc exit data delete(data[:rows*cols*depth*n])
        }
        return this;
    }
    
    Data4D* reset() {
        deaccel();
        dealloc();
        setShape(0,0,0,0);
        return this;
    }
    
    void print(string msg) {
        acc_update_self(data, getSize());
        
        cout << msg << endl;
        
        printf("Size: %d x %d x %d x %d = %d\n", n, depth, rows, cols, getSize());
        
        for(int nn = 0; nn < n; nn++) {
            printf("\n");
            for(int dd = 0; dd < depth; dd++) {
                print("----------\n");
                for(int nr = 0; nr < rows; nr++) {
                    for(int nc = 0; nc < cols; nc++) {
                        printf("%+4.5f|", io[nn* * depth * rows * cols + dd *rows*cols +  nr * cols + nc]);
                    }
                    printf("\n");
                }
                printf("---------");
            }
        }
    }
    
    
    
};

template <class T>
struct Data3D : Data4D<T> {
    Data3D(int x, int y, int z) : Data4D(x,y,z) {}
};


/////// NEW
/*
 * Name: acc_relu
 * Description: Performs a 'Linear rectified Unit' - ReLU function on the input and stores the result to the output
 *
 * Parameters: {
 *     relu_output: N x M size matrix containing the output of the ReLU operation
*      relu_output_drv: N x M size matrix containing the derivative of the output of the ReLU operation
 *     rl_input: N x M size matrix containing the input to the ReLU operation
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
template <class T>
void acc_relu4D(Data4D<T> *relu_output, Data4D<T> *relu_output_drv, Data4D<T> *rl_input) {
    int size = rl_input->getSize();
    #pragma acc data present(rl_input->data[:size], relu_output->data[:size], relu_output_drv->data[:size])
    {
    
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {
        T val = rl_input[j];

        if(val > 0) {
            relu_output[j] = val;
            relu_output_drv[j] = 1.0f;
        }
        else {
            relu_output[j] = 0.0f;
            relu_output_drv[j] = 0.0f;
        }
    }
    }
}


/*
 * Name: acc_sigmoid
 * Description: Performs a Sigmoid - Logistic function on the input and stores the result to the output
 *
 * Parameters: {
 *     sigm_output: N x M size matrix containing the output of the sigmoid function
*      sigm_output_drv: N x M size matrix containing the derivative of the output of the sigmoid function
 *     sg_input: N x M size matrix containing the input to the sigmoid function
 * }
 * 
 * Example:
 *          |-4 5 2|               |0.0180 0.9933 0.8808|
 *      A = |0 1 -1|, Sigmoid(A) = |0.5000 0.7310 0.2689|
 *          |6 -8 3|               |0.9975 0.0003 0.9526|
*/
template <class T>
void acc_sigmoid4D(Data4D<T> *sigm_output, Data4D<T> *sigm_output_drv, Data4D<T> *sg_input) {
    int size = sg_input->getSize();
    
    #pragma acc data present(sg_input->data[:size], sigm_output->data[:size], sigm_output_drv->data[:size])
    {
    
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {
        T val = 1/(1 + exp(-1 * sg_input[j]));

        sigm_output[j] = val;
        sigm_output_drv[j] = val*(1-val);
    }
    }
}

template <class T>
void acc_softmax4D(Data4D<T> *softmax_output, Data4D<T> *softmax_input) {
    int size = softmax_input->getSize();
    
    T softmax_output_sum = 0.0f;
    #pragma acc data present(softmax_input->data[:size], softmax_output->data[:batch_size*num_outputs]) create(softmax_output_sum)
    {
    
    #pragma acc parallel loop reduction(+:softmax_output_sum)
    for(int j = 0; j < size; j++) {
        softmax_output_sum += exp(softmax_input[j]);
//             #pragma acc atomic update
//             softmax_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
    }
        
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {            
        #pragma acc atomic write
        softmax_output[j] = exp(softmax_input[j]) / softmax_output_sum;
    }
    }
}


/*
 * Name: acc_matrix_add_row
 * Description: Adds the 1xM vector B horizontally to N rows of matrix A
 * Example:  A         B
 *        -------   -------
 *        |1 2 3|             |2 4  3|
 *        |4 5 6| + |1 2 0| = |5 7  6|
 *        |7 8 9|             |8 10 9|
 * 
 * Parameters: {
 *      A: N x M size matrix containing the elements of the first operand, must be 1-dimensional
 *      B: 1 x M size vector containing the elements of the second operand, must be 1-dimensional
 *      N: number of rows of matrix A
 *      M: number of columns of matrix A, B
 * }
 *
*/
template <class T>
void acc_matrix_add_row4D(Data4D<T> *A, Data4D<T> *B) {
    int N = A->rows, M = A->cols;
    #pragma acc data present(A->data[0:N*M], B->data[:M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {            
            A->data[i*M + j] += B->data[j]; 
        }
    }
    }
}

template <class T>
void acc_matrix_add_depthwise4D(Data4D<T> *A, Data4D<T> *B) {
    int N = A->rows, K = A->cols, D = A->depth;
    #pragma acc data present(A[0:N*K*D], B[:D])
    {
    
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < K; k++) {
            for(int d = 0; d < D; d++) {
                A[i*K*D + k*D + d] += B[d];
            }
        }
    }
    }
}

//TODO matrix multiply for
/*
 * Name: acc_matrix_multiply
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the operation C = A * B, must be 1-dimensional
 *      N: number of rows of matrix 'A'
 *      K: number of columns of matrix 'A' as well as number of rows of matrix 'B'
 *      M: number of columns of matrix 'B'
 * }
 *
*/
template <class T>
 void acc_matrix_multiply4D(Data4D<T>  * A, Data4D<T> * B, Data4D<T> * C, int N, int K, int M) {
    int N = A.rows, K = A.cols, M = B.cols;
    
    #pragma acc data present(A->data[:(N*K)], B->data[0:K*M], C->data[0:N*M])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            T csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += A->data[i*K + t]*B->data[t*M + j];
            }
            C->data[i*M + j] = csumd;
        }
    }
    
    }
}

template <class T>
void acc_convolution4D(Data4D<T> *conv_input, Data4D<T> *conv_filters, Data4D<T> *conv_output, int stride, bool debug) { 
    
    int in_cols = conv_input->cols;
    int in_rows = conv_input->rows;
    int in_channels = conv_input->depth;
    
    int filter_size = conv_filters->rows;
    int out_cols = conv_output->cols;
    int out_rows = conv_output->rows;
    int out_channels = conv_output->depth;
    
    T* in_data = conv_input->data, out_data = conv_output->data, filter_data = conv_filters->data;
    #pragma acc data present(in_data[:(in_cols*in_rows*in_channels)], filter_data[:in_channels*filter_size*filter_size*out_channels]) present(out_data[:(out_cols * out_rows * out_channels)])
    {
        
    #pragma acc parallel loop collapse(3)
    for(int d = 0; d < out_channels; d++) {
        for(int oh = 0; oh < out_rows; oh++) {
            for(int ow = 0; ow < out_cols; ow++) {
                T bdhwsum = 0.0f;
                
                #pragma acc loop seq collapse(3) reduction(+:bdhwsum)
                for(int ch = 0; ch < in_channels; ch++) {
//                         double csum = 0.0f;
                    for(int di = 0; di < filter_size; di++) {
                        for(int dj = 0; dj < filter_size; dj++) {
                            bdhwsum += in_data[ (i*in_channels + ch)*in_rows*in_cols + (oh + di)*in_cols + ow + dj ] * filter_data[ (d*in_channels + ch)*filter_size*filter_size + di*filter_size + dj ];
                        }
                    }
//                         sum += csum;
                }
                
                out_data[(i*out_channels + d)*out_cols*out_rows + oh*out_cols + ow] = bdhwsum;
            }
        }
    }
    
    }
    
}

//TODO activations as member functions?

/*
 * Name: acc_ad2D
 * Description: Zero-pads a 2-dimensional array by a specified amount of pixels in each direction 
 *
 * Parameters: {
 *      input: BxDxNxM input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padded_input: BxDxNxM padded input matrix where
 *          B: batch size [B]
 *          D: input channels [C]
 *          N: number of rows [N]
 *          M: number of columns [M]
 *      
 *      padding_left: number of pixels to pad to the left
 *      padding_right: number of pixels to pad to the right
 *      padding_top: number of pixels to pad to the top
 *      padding_bottom: number of pixels to pad to the bottom
 *      
 *      N: number of rows of original input
 *      M: number of columns of original input
 *      padded_N: number of rows after padding
 *      padded_M: number of columns after padding
 * }
 * 
 * Example:
*/
template <class T>
void acc_pad2D4D(Data4D<T> *input, Data4D<T> *padded_input, int padding_left, int padding_right, int padding_top, int padding_bottom) {
    int C = input->depth;
    int N = input->rows;
    int M = input->cols;
    
    int padded_N = N + padding_top + padding_bottom;
    int padded_M = M + padding_left + padding_right;
    
    #pragma acc data present(input[:(C*M*N)], padded_input[:(C*padded_N*padded_M)])
    {
        
    #pragma acc parallel loop collapse(3) 
    for(int c = 0; c < C; c++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++) {
                padded_input[c*padded_N*padded_M + (i+padding_top)*padded_M + j + padding_left] = input[c*M*N + i*M + j];
            }
        }
    }
    
    }
}

template <class T>
void acc_make_batch4D(Data4D<T> * inputs, Data4D<T> *batch, int batch_size, int input_size, int batch_start) {
    #pragma acc data present(inputs->data[(batch_start*input_size):(batch_size*input_size)]) present(batch->data[:(batch_size*input_size)])
    {
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < input_size; k++) {
            batch[i*input_size + k] = inputs[(i+batch_start)*input_size + k];
        }
    }
        
    }
}
struct size4D {
public:
    int width{0}, height{0}, depth{0}, hdepth{0};

public:
    size4D() {}
    size4D(int w, int h, int d, int hd): width(w), height(h), depth(d), hdepth(hd)  {}
    size4D(int w, int h, int d): size4D(w, h, d, 0) {}
    size4D(int w, int h) : size4D(w,h,0) {}
    size4D(int w): size4D(w, 0) {}
    
    virtual int product() {
        return this->width*this->height*this->depth*this->hdepth;
    }
    
    void setWidth(int w) {
        width = w;
    }
    
    void setHeight(int h) {
        height = h;
    }
    
    int getWidth() { return width; }
    int getHeight() { return height; }
};

struct size3D: public size4D {
public:
    size3D() : size4D() {}
    size3D(int w, int h, int d): size4D(w,h,d) {}
    
    int product() override {
        return width*height*depth;
    }
    
};

struct size2D: public size4D {
public:
    size2D() : size4D() {}
    size2D(int w, int h): size4D(w, h) {}
    
    int product() override {
        return width*height;
    }
};

struct size1D: public size4D{
public:
    size1D() : size4D() {}
    size1D(int w): size4D(w) {}
    
    int product() override {
        return width;
    }
};

struct ImgCl {
    int width, height, channels;
    
    
};

struct LayerNode {
    double value;
    
    
};

struct aImg {
    
};

template<class T> struct LayerData {
    T *data;
    
    int num_samples, num_nodes, node_width, node_height;
    
    LayerData() {}
    LayerData(int ns, int nn, int nw, int nh) : num_samples(ns), num_nodes(nn), node_width(nw), node_height(nh) {}
    
    void setData(int ns, int nn, int nw, int nh) {
        num_samples = ns;
        num_nodes = nn;
        node_width = nw;
        node_height = nh;
    }
    
    void alloc() {
        data = new T[num_samples * num_nodes * node_width * node_height];
    }
    
    void dealloc() {
        delete data[];
    }
    
    void reset() {
        num_samples = 0;
        num_nodes = 0;
        node_width = 0;
        node_height = 0;
        
        dealloc();
    }
    
    void flatten() {
        num_nodes = num_nodes * node_width * node_height;
        node_width = 1;
        node_height = 1;
    }
        
};


template<class T> struct __Data4D {
    T *data;
    
    int x,y,z,w;
    
    __Data4D(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
    
    void setData(int a, int b, int c, int d) {
        x=a;
        y=b;
        z=c;
        w=d;
    }
    
    void alloc() {
        data = new T[x*y*z*w];
    }
    
    void dealloc() {
        delete data[];
    }
    
    void reset() {
        x=0;
        y=0;
        z=0;
        h=0;
        dealloc();
    }
};



template <class T>
struct Weights4D {
    Data4D<T> *weights, *bias;
    
    Weights4D(int a, int b, int c, int d) {
        weights = Data4D(a,b,c,d);
        bias = Data4D(a);
    }
    
    void alloc() {
        weights::alloc();
        bias::alloc();
    }
    
    void accel() {
        weights->accel();
        bias->accel();
    }
};

struct LayerShape {
    int rows, cols, depth;
    
    LayerShape() {}
    LayerShape(int r, int c, int d): rows(r), cols(c), depth(d) {}
    
    int size() {
        return rows*cols*depth;
    }
};

template<class T>
struct Weights {
    T *data;
    int size;
    
    Weights(T *_data, int _size) : data(_data), size(_size) {}
};

//TODO overload Data4D operations? +-,
//TODO pooling layer, dropout layer, batch normalization layer, deconvolution layer
//TODO weight layers, non-weight layers OR weight_size = 0
//TODO require prev layer on constructor, set weight sizes
//TODO test batch now vs batch normal vs online (duration, accuracy)
// TODO Weights class including biases, dimensionality

class Network;
class Layer;

class Layer {
    
friend class Network;
friend class Layer;

protected:
    Layer *prev_l{nullptr}, *next_l{nullptr};
    
    LayerShape shape;
    Data4D<double> *input, *output, *output_drv;
    
    int batch_size{1}, num_nodes{0}, size{0};
    size2D node_size;
    string layerType{""}, layerOp{""};
    bool debug{false};
    
    Data4D<double> *weights, *biases;
    int weights_size{0}, biases_size{0};
    string activation_fn{""};
    bool debug_weights{false}, debug_biases{false}, debug_op_primary{false}, debug_op_secondary{false}, debug_output{false};    

public:
    
    Layer(int nn, string afn) {
        activation_fn = afn;
        num_nodes=nn;
    }
    
    Layer(int nn): Layer(nn, "") { }
    
    Layer() {}
    
    int const getNumNodes() { return num_nodes; }
       
    int const getNumWeights() { return weights_size; }
    
    int const getNumBiases() {  return biases_size; }
    
    int const get_size() { return size; }
    
    size2D const get_node_size() { return node_size; }
    
    Data4D<double>* const get_output() { return output; }
    
    Data4D<double>* getWeights() { return weights; }
    
    string const getActivationFn() { return activation_fn; }
    
    void setDebug() { debug=true; }
    
    void resetDebug() { debug=false; }

protected:
    virtual void print_param_info() = 0;

    virtual void runop_primary() {}
    virtual void runop_secondary() {}
    // pure virtual, different implementations/layer
    
//     virtual void backpropagation() = 0;
    
    virtual void init_params() = 0;

    // basic init weights, can override if lacking, TODO requires! weights size set before (in ctor?)
    virtual void init_weights() {
        weights->setData(makeweights(weights->getSize(), 0.1f))->accel(true);
        biases->setData(makebiases(biases->getSize(), 0.1f))->accel(true);
    }
    
    // basic init train data, can override if lacking
    virtual void init_train_data(int bs) {
        batch_size = bs;
        input = (new Data4D<double>(batch_size, shape.rows, shape.cols, shape.depth))->alloc()->accel(false);
        output = (new Data4D<double>(batch_size, shape.rows, shape.cols, shape.depth))->alloc()->accel(false);
        output_drv = (new Data4D<double>(batch_size, shape.rows, shape.cols, shape.depth))->alloc()->accel(false);
    }
    
    // basic activate, can override
    virtual void runfn() {
        if(activation_fn == "relu") { 
            acc_relu4D<double>(output, output_drv, input);
        }
        else if(activation_fn == "sigmoid") {
            acc_sigmoid4D<double>(output, output_drv, input);
        }
        else if(activation_fn == "") { 
            output = input;
        } //TODO output = copy(input)?
        else if(activation_fn == "softmax") {
           acc_softmax4D<double>(output, input);
        }
        else {
        }//TODO throw error
    }
    

    
    void setPrev(Layer *prev_layer)  {
        this->prev_l = prev_layer;
    }
    
    void setNext(Layer *next_layer) {
        this->next_l = next_layer;
    }
    
    void init(int bs) {
        init_params();
        init_weights();
        init_train_data(bs);
        
    }
    
    void forward() {
        
        if(debug) {
            printf("Forward | ");
            cout << "Op " << layerOp << endl;
        }
        
        if(debug) {
            weights->print("---------------------\n");            
        }
        
        runop_primary();
        
        if(debug && debug_op_primary) {
            input->print("---------------------\n");
        }
        
        if(debug && debug_biases) {
            biases->print("---------------------\n+Bias\n---------------------\n");
        }
        
        runop_secondary();
        
        //TODO check layername?, bools, weights print, op weights print, op bias print, op out print
        if(debug && debug_op_secondary) {
            input->print(---------------------\n);
        }
        
        
    }
    
    void print_weights() {
        weights->print("Weights");
    }
    
    void print_biases() {
        biases->print("Biases");
    }
    
    void activate() {
        runfn();
        
        if(debug) {
            printf("Activate | ");
            cout << "Fn " << activation_fn << endl;
            output->print("---------------------\n");
        }
    }
    
}
    
    
};


class Layer_FC: public Layer {
    
public:
    Layer_FC(int nn, string afn): Layer(nn, afn) {
        debug_weights=true;
        debug_biases=true;
        debug_op_primary=true;
        debug_op_secondary=true;
        layerType = "fc";
        layerOp = "matrixmul";
        shape = LayerShape(1,1,nn);
    }
    
protected:    
    void init_params() override {
        
        node_size.width = 1;
        node_size.height = 1;
        size = node_size.product() * num_nodes;
        
        weights_size = prev_l->shape.size() * shape.size();
        biases_size = shape.depth;
//         weights_size = prev_l->get_size() * size;
//         biases_size = size;
    }

    void runop_primary() override {
        acc_matrix_multiply(prev_l->get_output(), weights, input, batch_size, prev_l->get_size(), size);
    }
    
    void runop_secondary() override {
        acc_matrix_add_row(input, biases, batch_size, size);
    }
    
    void print_param_info() override {
        printf("FC | init -> nodes: %d fn: ", num_nodes);
        cout << activation_fn << endl;
        
        printf("FC | init calc -> | node_size: %d x %d = %d | size: %d | weights_size: %d | biases_size: %d\n", node_size.width, node_size.height, node_size.product(), size, weights_size, biases_size);
        
        printf("FC | op -> name: acc_matrix_multiply, params: [N: %d, K: %d -> M: %d ]\n", batch_size, prev_l->get_size(), size);
    }
    
    void print_weights() override {
        printf("Weights %d x %d, sample\n---------------------\n", prev_l->get_size(), size);

        
        acc_update_self(weights, weights_size);
        for(int i = 0; i < min(prev_l->get_size(), 50); i++) {
            for(int j = 0; j < min(size, 2); j++) {
                printf("|%+4.5f", weights[i*size + j]);
            }
            printf("|\n");
        }
    }
    
    
};

//TODO Flatten layer?

class ConvLayer: public Layer {

private:
    int stride, filter_size;
    size2D prev_node_size;

public:
    ConvLayer(int num_nodes, string activation_fn, int filter_size, int stride): Layer(num_nodes, activation_fn), filter_size(filter_size), stride(stride) {
        debug_weights=true;
        debug_biases=true;
        debug_op_primary=true;
        debug_op_secondary=true;
        layerType = "conv";
        layerOp = "conv";
    }

protected:
    void init_params() override {
        int out_height, out_width, padding = 0;
        calc_conv_sizes(prev_l->get_node_size().height, prev_l->get_node_size().width, filter_size, stride, false, padding, padding, padding, padding, out_height, out_width);
        
        node_size.width = out_width;
        node_size.height = out_height;
        size = node_size.product()*num_nodes;
        activation_fn = activation_fn;
        
        weights_size = num_nodes * prev_l->getNumNodes() * filter_size * filter_size;
        biases_size = num_nodes;
    }

    void runop_primary() override {
        acc_convolution(prev_l->get_output(), weights, input, batch_size, prev_l->getNumNodes(), prev_l->get_node_size().height, prev_l->get_node_size().width, num_nodes, node_size.height, node_size.width, filter_size, stride, false);
    }
    
    void runop_secondary() override {
        acc_matrix_add_depthwise(input, biases, batch_size, node_size.product(), num_nodes);
    }
    
    void print_weights() override {
        printf("Weights %d(nodes) x %d(prev_nodes) x [%d x %d]\n---------------------\n", num_nodes, prev_l->getNumNodes(), filter_size, filter_size);

        acc_update_self(weights, weights_size);
        for(int i = 0; i < min(num_nodes, 3); i++) {
            for(int j = 0; j < prev_l->getNumNodes(); j++) {
                for(int k = 0; k< filter_size; k++) {\
                    for(int g = 0; g< filter_size; g++) {
                        printf("|%+4.5f", weights[i*prev_l->getNumNodes() * filter_size * filter_size + j * filter_size * filter_size + k*filter_size + g]);
                    }
                    printf("|\n");
                }
            }
        }
    }
    
    void print_param_info() override {
        printf("Conv | init -> nodes: %d fn: ", num_nodes);
        cout << activation_fn << " ";
        printf("filter_size: %d | stride: %d\n", filter_size, stride);
        
        printf("Conv | calc param info | node_size: %d x %d = %d, size: %d, weights_size: %d, biases_size: %d\n",node_size.width, node_size.height, node_size.product(), size, weights_size, biases_size);

        printf("Conv | op -> name: acc_convolution, params: [batch_size: %d, in_channels: %d, in_height: %d, in_width: %d, out_channels: %d, out_height: %d, out_width: %d, filter_size: %d, stride: %d]\n", batch_size, prev_l->getNumNodes(), prev_l->get_node_size().height, prev_l->get_node_size().width, num_nodes, node_size.height, node_size.width, filter_size, stride);

    }
    
};

class InputLayer: public Layer {
    friend class Network;
    
private:
    double *img_data;
    bool loaded;
    int total_images, step;
    int img_width, img_height;
    
public:
    
    InputLayer(int img_width, int img_height, int num_channels) : Layer(num_channels), img_width(img_width), img_height(img_height), loaded(false), step(0) {
        debug_weights=false;
        debug_biases=false;
        debug_op_primary=true;
        debug_op_secondary=false;
        layerType = "input";
        layerOp = "make_batch";
    }
    
    InputLayer(size2D img_size, int num_channels) : InputLayer(img_size.width, img_size.height, num_channels) { }
    

protected:
    
    void load_img_data(double *imdata, int num_images) {
        img_data = imdata;
        total_images = num_images;
        loaded = true;
    }
    
    void init_params() override {
        node_size.width = img_width;
        node_size.height= img_height;
        cout << "Input node size product: " << node_size.product() << endl;
        size = node_size.product() * num_nodes;
        
        weights_size = 0;
        biases_size = 0;
    }
        
    void init_train_data(int bs) override {
        batch_size = bs;
        
        if(!loaded) {
            throw(std::invalid_argument("Input layer not loaded"));
        }
        input = new double[batch_size * size];
        output = input;
        
        step = 0;
        
        #pragma acc enter data create(input[:batch_size*size])
    }
    
    void load_gpu_imdata() {
        #pragma acc enter data copyin(img_data[:total_images*size])
    }
    
    void runop_primary() override {
        //pcopyin img_data        
        acc_make_batch(img_data, input, batch_size, size, (step*batch_size)%(total_images));
        step++;
    }
    
    void print_param_info() override {
        printf("Input | init -> nodes: %d, img_width: %d, img_height: %d\n", num_nodes, img_width, img_height);
        printf("Input | calc init -> node_size: %d x %d = %d, size: %d\n", node_size.height, node_size.width, node_size.product(), size);
        printf("Input | op -> name: acc_make_batch, params: [batch_size: %d, input_size: %d]\n", batch_size, size);
    }
        
    
};

//TODO enforce out/softmax check on ctor
class OutputLayer: public Layer_FC {
    
private:
    double *output_sum;
    
    
protected:
    
    void runfn() override {

        else {
            //TODO throw error
        }
    }

public:
    OutputLayer(int nn, string afn) : Layer_FC(nn, afn) {}
    
};

class PadLayer: public Layer {

private:
    int padding[4], in_width_padded, in_height_padded;

public:
    PadLayer(int padding_left, int padding_right, int padding_top, int padding_bottom): Layer(), in_width_padded(0), in_height_padded(0) {
        this->padding[0] = padding_top;
        this->padding[1] = padding_bottom;
        this->padding[2] = padding_left;
        this->padding[3] = padding_right;
        debug_weights=false;
        debug_biases=false;
        debug_op_primary=true;
        debug_op_secondary=false;
        layerType = "pad";
        layerOp = "pad2D";
        
    }
    
    PadLayer(int padding[4]) : PadLayer(padding[0], padding[1], padding[2], padding[3]) {}
    
protected:
        
    void init_params() override {
        in_width_padded = prev_l->get_node_size().width + padding[0] + padding[1];
        in_height_padded = prev_l->get_node_size().height + padding[2] + padding[3];
        
        node_size.height = in_height_padded;
        node_size.width = in_width_padded;
        num_nodes = prev_l->getNumNodes();
        size = num_nodes * node_size.product();
        
        weights_size = 0;
        biases_size = 0;
    }

    void init_train_data(int bs) override {
        batch_size = bs;
        input = new double[batch_size * size];
        
        #pragma acc enter data create(input[:batch_size*size])
    }
    
    void runop_primary() override {
        acc_pad2D(prev_l->get_output(), input, batch_size, num_nodes, prev_l->get_node_size().height, prev_l->get_node_size().width, padding[0], padding[1], padding[2], padding[3]);
    }
    
    void print_param_info() override {
        printf("Pad | init -> padding:[%d, %d, %d, %d]\n", padding[0], padding[1], padding[2], padding[3]);
        
        printf("Pad | calc init -> num_nodes: %d, node_size: %d x %d = %d , size : %d\n", num_nodes, node_size.height, node_size.width, node_size.product(), size);
        
        printf("Pad | op -> name: acc_pad2D, params: [batch_size: %d, nodes: %d, start_height: %d, start_width: %d, padding_left: %d, padding_right: %d, padding_top: %d, padding_bottom: %d]\n", batch_size, num_nodes, prev_l->get_node_size().height, prev_l->get_node_size().width, padding[0], padding[1], padding[2], padding[3]);
    }
};

//TODO template layers double,float?
class NetworkException {
};

class Network {
    vector<Layer *> layers;
    InputLayer *input_layer{nullptr};
    int batch_size;
    bool hasInputLayer;

public:
    Network(): hasInputLayer(false) {}
    Network(int bs) : batch_size(bs), hasInputLayer(false) {}
    
    void add_input(int num_channels, int img_width, int img_height) {
        
        if(!hasInputLayer) {
            input_layer = new InputLayer(img_width, img_height, num_channels);
            layers.insert(layers.begin(), input_layer);
            hasInputLayer = true;
        }
        else {
            throw(std::invalid_argument("Network already has input layer."));
        }
    }
    
    Layer *get_output_layer() {
        return layers[layers.size()-1];
    }
    
    void load_input(double *inputs, int num_inputs) {
        input_layer->load_img_data(inputs, num_inputs);
    }
    
    void add_layer(Layer *l) {
        if(layers.size() >0 && hasInputLayer) {
            layers.back()->setNext(l);
            l->setPrev(layers.back());
            layers.push_back(l);
        }
        else { 
            /* TODO throw error */
            throw(std::invalid_argument("Network has no input layer."));
        }
        
    }
    
    void init(int bs) {
        int l = 0;
        
        printf("\nInit\n\n");
        
        for(auto it = begin(layers); it!=end(layers); ++it, l++) {
            printf("Layer %d\n", l);
            (*it)->init(bs);
            (*it)->print_param_info();
            (*it)->setDebug();
            cout << endl;
        }
    }
    
    void train(int iters, int bs, bool accel_inp) {
        init(bs);
        
        if(accel_inp) {
            input_layer->load_gpu_imdata();
        }
        
        clock_t start = clock();
        double duration;

        for(int i = 0; i < iters; i++) {
            printf("\nStep %d\n\n",i);
            
            int l = 0;
            
            for(auto it = begin(layers); it!=end(layers); ++it, l++) {
                printf("\n************* Layer %d *************\n\n", l);

                (*it)->forward();
                
                (*it)->activate();
                
                
                cout<<endl;
                printf("****************************************\n");

            }
            
        }
        
        duration = (clock() - start)/CLOCKS_PER_SEC;
        cout << "Train duration: " << duration << endl;
                  
    }
};  

class Base {
public:
    int a, b;
    
    Base() {
        printf("Base default\n");
    }
    
    Base(int x): Base(x,1) {
        a = x;
        printf("Base int\n");
    }
    
    Base(int x, int y) {
        a = x;
        b = y;
        printf("Base int int\n");
    
    }
    
    void printAB() {
        printf("Base a: %d, b: %d\n", a, b);
    }
    
    void product() {
        printf("A*b -> %d x %d = %d\n", a,b, a*b);
    }
    
};

class Derived: public Base {
public:
    Derived() {
        printf("Derived default\n");
    }
    
    Derived(int x):Base(x) {
        printf("Derived int\n");
    }
};


int main(int argc, char *argv[]) {
    printf("Hello World Classes training\n");
    

    int num_images = 0, num_labels = 0, img_size = 0, num_rows = 0, num_cols = 0; 
    
    // Load the data
    uchar **train_img_data = read_mnist_images("data/train-images-idx3-ubyte", num_images, img_size, num_rows, num_cols);
    uchar *train_labels_data = read_mnist_labels("data/train-labels-idx1-ubyte", num_labels);

    double dml = 1;
    // TODO backprop divide by batch size, emltp? original
    int num_samples = num_images, num_inputs = img_size, num_channels = 1, num_outputs = 10;
    int batch_size = 32, num_hidden_nodes3 = 256;
    
    double *train_images = data2mono(train_img_data, num_images, img_size, dml);
    double *train_labels = labels1hot(train_labels_data, num_labels, num_outputs);
    
    double *train_images_orig = data2mono_orig(train_img_data, num_images, img_size);

    
   /* 
    Network net;

    net.add_input(num_channels, num_cols, num_rows);
    net.add_layer(new PadLayer(new int[4]{2, 2, 2, 2}));
    net.add_layer(new ConvLayer(64, "relu", 5,1));
    net.add_layer(new PadLayer(new int[4]{2, 2, 2, 2}));
    net.add_layer(new ConvLayer(64, "relu", 5,1));
    net.add_layer(new Layer_FC(256, "relu"));
    net.add_layer(new OutputLayer(10,"softmax"));
    net.load_input(train_images, num_images);*/
//     
// //     net.add_layer(&fc1);
// //     net.add_layer(fc2);
//     
    int sst = atoi(argv[1]);
    //TODO layer data allocation control
    //TODO factories?
    //TODO helper structs? e.g size2D, imgData3D?
    
    int num_r = 3, num_c = 3;
    double *test_input = new double[num_r*num_c];
    for(int i = 0; i < num_r; i++) {
        for(int j = 0; j < num_c; j++) {
            test_input[i*num_c + j] = num_c*i + j + 1;            
        }
    }

    cv::Mat im1(num_r, num_c, CV_64F, test_input);
    cout << im1 << endl;
   
    
    //TODO row major all functions
    //TODO good prints, operations, inputs outputs weights, parameters, ability to poll each stage
    //TODO accel input as switch? ON/OFF
    
    Network testnet;
    testnet.add_input(1, num_c, num_r);
    testnet.load_input(test_input, 1);
    testnet.add_layer(new PadLayer(new int[4]{2, 2, 2, 2}));
    testnet.add_layer(new ConvLayer(1, "relu", 2, 1));
    testnet.train(1,1, true);

    
    return 0;
    std::cout << "Hello World convolutions" << std::endl;    
/*    
    void (*testf_pointer)();
    testf_pointer = &testf;
    cout << testf << " | " << &testf << " | " << testf_pointer << endl;
    testf_pointer();*/
//     int num_images = 0, num_labels = 0, img_size = 0, num_rows = 0, num_cols = 0; 
//     
//     // Load the data
//     uchar **train_img_data = read_mnist_images("data/train-images-idx3-ubyte", num_images, img_size, num_rows, num_cols);
//     uchar *train_labels_data = read_mnist_labels("data/train-labels-idx1-ubyte", num_labels);
//     
//     // Transform input to one-dimensional array, labels to 1-hot encoding
//     double dml = 1;
//     // TODO backprop divide by batch size, emltp? original
//     int num_samples = num_images, num_inputs = img_size, num_channels = 1, num_outputs = 10;
//     int batch_size = 32, num_hidden_nodes3 = 256;
//     
//     double *train_images = data2mono(train_img_data, num_images, img_size, dml);
//     double *train_labels = labels1hot(train_labels_data, num_labels, num_outputs);


//     float **c_train_images = convert_train_dataset_f(train_img_data, num_images, 28, 28);
    
    double *train_batch = new double[batch_size * num_channels * num_rows * num_cols];
    double *train_labels_batch = new double[batch_size * num_outputs];
    
    ///*
    // Initialize weights, biases etc.
    
    // layer 1 - convolution
    ////////////////////////////////////////////////////////////////////
    int in_width_conv1 = num_cols, in_height_conv1 = num_rows, num_channels_conv1 = 1, filter_size_conv1 = 5, depth_conv1 = 64, stride_conv1 = 1;
    int padding_top_conv1, padding_bottom_conv1, padding_left_conv1, padding_right_conv1;
    int padding_height_conv1, padding_width_conv1,  padded_width_input_conv1, padded_height_input_conv1, out_width_conv1, out_height_conv1, out_size_conv1, out_vol_size_conv1;
    int wflat_conv1 = filter_size_conv1 * filter_size_conv1 * num_channels_conv1;
    int filter_flat_size_conv1 = wflat_conv1 * depth_conv1;
    
    calc_conv_padding(in_height_conv1, in_width_conv1, filter_size_conv1, stride_conv1, out_height_conv1, out_width_conv1, padding_left_conv1, padding_right_conv1, padding_top_conv1, padding_bottom_conv1);
    
    
    calc_conv_sizes(in_height_conv1, in_width_conv1, filter_size_conv1, stride_conv1, true, padding_left_conv1, padding_right_conv1, padding_top_conv1, padding_bottom_conv1, out_height_conv1, out_width_conv1);
    
    padding_width_conv1 = padding_left_conv1 + padding_right_conv1;
    padding_height_conv1 = padding_top_conv1 + padding_bottom_conv1;
    padded_width_input_conv1 = in_width_conv1 + padding_width_conv1;
    padded_height_input_conv1 = in_height_conv1 + padding_height_conv1;
    out_size_conv1 = out_width_conv1 * out_height_conv1;
    out_vol_size_conv1 = out_size_conv1 * depth_conv1;

    double *train_weights_conv1 = makeweights(filter_flat_size_conv1, 0.1f);
    double *train_biases_conv1 = makebiases(depth_conv1, 0.0f);
    
    double *train_x_in_conv1 = new double[batch_size * padded_width_input_conv1 * padded_height_input_conv1 * num_channels_conv1];
    zeros<double>(train_x_in_conv1, batch_size * padded_width_input_conv1 * padded_height_input_conv1 * num_channels_conv1);
    
    double *train_in_conv1 = new double[batch_size * out_vol_size_conv1];
    double *train_out_conv1 = new double[batch_size * out_vol_size_conv1];
    double *train_out_conv1_drv = new double[batch_size * out_vol_size_conv1];

    ///////////////////////////////////////////////////////////////////////////////
    ///*
    // layer 2 - convolution
    ///////////////////////////////////////////////////////////////////////////
    int in_width_conv2 = out_width_conv1, in_height_conv2 = out_height_conv1, num_channels_conv2 = depth_conv1, filter_size_conv2 = 5, depth_conv2 = 64, stride_conv2 = 1;
    int padding_height_conv2, padding_width_conv2, padding_top_conv2, padding_bottom_conv2, padding_left_conv2, padding_right_conv2, padded_width_input_conv2, padded_height_input_conv2;
    int out_width_conv2, out_height_conv2, out_size_conv2, out_vol_size_conv2;
    int wflat_conv2 = filter_size_conv2 * filter_size_conv2 * num_channels_conv2;
    int filter_flat_size_conv2 = wflat_conv2 * depth_conv2;

    calc_conv_sizes(in_height_conv2, in_width_conv2, filter_size_conv2, stride_conv2, true, padding_left_conv2, padding_right_conv2, padding_top_conv2, padding_bottom_conv2, out_height_conv2, out_width_conv2);
    
    padding_height_conv2 = padding_top_conv2 + padding_bottom_conv2;
    padding_width_conv2 = padding_left_conv2 + padding_right_conv2;
    padded_width_input_conv2 = in_width_conv2 + padding_width_conv2;
    padded_height_input_conv2 = in_height_conv2 + padding_height_conv2;
    out_size_conv2 = out_width_conv2 * out_height_conv2;
    out_vol_size_conv2 = out_size_conv2 * depth_conv2;
    
    double *train_weights_conv2 = makeweights(filter_flat_size_conv2, 0.1f);
    double *train_biases_conv2 = makebiases(depth_conv2, 0.0f);
    
    double *train_x_in_conv2 = new double[batch_size * num_channels_conv2 * padded_width_input_conv2 * padded_height_input_conv2];
    zeros<double>(train_x_in_conv2, batch_size * num_channels_conv2 * padded_width_input_conv2 * padded_height_input_conv2);

    double *train_in_conv2 = new double[batch_size * out_vol_size_conv2];
    double *train_out_conv2 = new double[batch_size * out_vol_size_conv2];
    double *train_out_conv2_drv = new double[batch_size * out_vol_size_conv2];
    
    ///////////////////////////////////////////////////////////////////////////////////
    ///*
    // layer 3 - relu
    //////////////////////////////////////////////////////////////////////////////////
    int in_size_hidden3 = out_vol_size_conv2, out_size_hidden3 = num_hidden_nodes3;
    double *train_weights_hidden3 = makeweights(in_size_hidden3 * out_size_hidden3, 0.1f);
    double *train_biases_hidden3 = makebiases(out_size_hidden3, 0.0f);
    double *train_wx_hidden3 = new double[batch_size * out_size_hidden3];
    double *train_out_hidden3 = new double[batch_size * out_size_hidden3];
    double *train_out_hidden3_drv = new double[batch_size * out_size_hidden3];

    ////////////////////////////////////////////////////////////////////////////////////
    ///*
    // layer 4 - softmax
    ////////////////////////////////////////////////////////////////////////////////
    int in_size_output4 = out_size_hidden3, out_size_output4 = num_outputs;
    double *train_weights_output4 = makeweights(in_size_output4 * out_size_output4, 0.1f);
    double *train_biases_output4 = makebiases(out_size_output4, 0.0f);
    double *train_wx_output4 = new double[batch_size * out_size_output4];
    double *train_out_output4 = new double[batch_size * out_size_output4];
    double *train_out_output4_sum = new double[batch_size];
    //////////////////////////////////////////////////////////////////////////////////
    
    ///*
    // backpropagation results initialization
    //////////////////////////////////////////////////////////////////////////////////
    double *drv_loss_input_output4 = new double[batch_size * out_size_output4];
    double *drv_loss_weights_output4 = new double[in_size_output4 * out_size_output4];
    double *drv_loss_biases_output4 = new double[out_size_output4];

    double *drv_loss_output_hidden3 = new double[batch_size* out_size_hidden3];
    double *drv_loss_input_hidden3= new double[batch_size* out_size_hidden3];
    double *drv_loss_weights_hidden3 = new double[in_size_hidden3 * out_size_hidden3];
    double *drv_loss_biases_hidden3 = new double[out_size_hidden3];

    double *drv_loss_output_conv2 = new double[batch_size * out_vol_size_conv2];
    double *drv_loss_input_conv2 = new double[batch_size * out_vol_size_conv2];
    double *drv_loss_weights_conv2 = new double[filter_flat_size_conv2];
    double *drv_loss_biases_conv2 = new double[depth_conv2];

    double *drv_loss_output_conv1 = new double[batch_size * out_vol_size_conv1];
    double *drv_loss_input_conv1 = new double[batch_size * out_vol_size_conv1];
    double *drv_loss_weights_conv1 = new double[filter_flat_size_conv1];
    double *drv_loss_biases_conv1 = new double[depth_conv1];
    ///////////////////////////////////////////////////////////////////////////////////////
    
    //////////////
    cout << "Size of argv: " << argc << endl;
    
    if(argc<=1) {
        cout << "Error: Steps argument not provided" << endl;
        return 0;
    }
    
    int steps = atoi(argv[1]);
    cout  << "Steps: " << steps << endl;
    double learning_rate = 0.05;
    int batch_start = 0;
    
    printf("num images: %d, num_labels: %d, num_outputs: %d, num_hidden_nodes: %d\n", num_images, num_labels, num_outputs, num_hidden_nodes3);
    printf("Batch size: %d | Img size: %d | num_rows: %d | num_cols: %d \n",batch_size ,img_size, num_rows, num_cols);
    
    printf("Layer 1 ||| Input size: %dx%d | Padded input size: %d x %d  | Num channels: %d  | Depth: %d | filter size:  %dx%d  | stride: %d  | output size: %dx%d\n", in_width_conv1, in_height_conv1, padded_width_input_conv1, padded_height_input_conv1, num_channels_conv1, depth_conv1, filter_size_conv1, stride_conv1, out_width_conv1, out_height_conv1);
  
    printf("Layer 2 ||| Input size: %dx%d | Padded input size: %d x %d  | Num channels: %d  | Depth: %d | filter size:  %dx%d  | stride: %d  | output size: %dx%d\n", in_width_conv2, in_width_conv2, padded_width_input_conv2, padded_height_input_conv2, num_channels_conv2, depth_conv2, filter_size_conv2, stride_conv2, out_width_conv2, out_height_conv2);
    
    printf("Layer 3 ||| Num outputs: %d\n", out_size_hidden3);
    printf("Layer 4 ||| Num outputs: %d\n", out_size_output4);

    cout << "Steps: " << steps << " | learning rate: " << learning_rate << endl;
    clock_t start;
    double duration;
    
    cout << endl;
    
    cout << "begin" << endl;
    
    // Allocate data in GPU and copy back when done
    #pragma acc data \
    copyin( train_images[:(num_images*img_size*num_channels)], \
            train_labels[:num_images*num_outputs], \
            train_batch[:(batch_size*num_inputs)], \
            train_labels_batch[:batch_size*num_outputs]) \
    \
    \
    copy(   train_weights_conv1[:filter_flat_size_conv1], \
            train_biases_conv1[:depth_conv1] ) \
    \
    copyin( train_x_in_conv1[:(batch_size*padded_width_input_conv1*padded_height_input_conv1*num_channels_conv1)] ) \
    \
    create( train_in_conv1[:(batch_size* out_vol_size_conv1)], \
            train_out_conv1[:(batch_size*out_vol_size_conv1)], \
            train_out_conv1_drv[:(batch_size*out_vol_size_conv1)], \
            drv_loss_weights_conv1[:filter_flat_size_conv1], \
            drv_loss_biases_conv1[:depth_conv1], \
            drv_loss_output_conv1[:batch_size*out_vol_size_conv1], \
            drv_loss_input_conv1[:batch_size*out_vol_size_conv1]) \
    \
    \
    copy(   train_weights_conv2[:filter_flat_size_conv2], \
            train_biases_conv2[:depth_conv2] ) \
    \
    copyin( train_x_in_conv2[:(batch_size*padded_width_input_conv2*padded_height_input_conv2*num_channels_conv2)] ) \
    \
    create( train_in_conv2[:(batch_size* out_vol_size_conv2)], \
            train_out_conv2[:(batch_size*out_vol_size_conv2)], \
            train_out_conv2_drv[:(batch_size*out_vol_size_conv2)], \
            drv_loss_weights_conv2[:filter_flat_size_conv2], \
            drv_loss_biases_conv2[:depth_conv2], \
            drv_loss_output_conv2[:batch_size*out_vol_size_conv2], \
            drv_loss_input_conv2[:batch_size*out_vol_size_conv2]) \
    \
    \
    copy(   train_weights_hidden3[:(out_vol_size_conv2*out_size_hidden3)], \
            train_biases_hidden3[:out_size_hidden3]) \
    \
    create( train_wx_hidden3[:batch_size*out_size_hidden3], \
            train_out_hidden3[:batch_size*out_size_hidden3],\
            train_out_hidden3_drv[:batch_size*out_size_hidden3], \
            drv_loss_biases_hidden3[:out_size_hidden3], \
            drv_loss_weights_hidden3[:out_vol_size_conv2*out_size_hidden3], \
            drv_loss_output_hidden3[:batch_size*out_size_hidden3], \
            drv_loss_input_hidden3[:batch_size*out_size_hidden3]) \
    \
    \
    copy(   train_weights_output4[:in_size_output4*out_size_output4], \
            train_biases_output4[:out_size_output4]) \
    \
    create( train_wx_output4[:batch_size*out_size_output4], \
            train_out_output4[:batch_size*out_size_output4], \
            train_out_output4_sum[:batch_size], \
            drv_loss_biases_output4[:out_size_output4], \
            drv_loss_weights_output4[:in_size_output4*out_size_output4], \
            drv_loss_input_output4[:batch_size*out_size_output4])
    {
    start = clock();
    
    for(int s = 0; s < steps; s++) {
        batch_start = ( s * batch_size ) % (60000 - batch_size);
//         batch_start = 0;
        bool debug = false, debug_b = false, debug_bw = false;
        
        if(s%5 == 0) {
            cout << "step " << s << endl;
        }
        
        make_batch(train_images, train_batch, batch_size, num_inputs, batch_start);
        make_batch(train_labels, train_labels_batch, batch_size, out_size_output4, batch_start);
        
        ///////////// Forward pass ///////////////
        // TODO fix argument order, outputs->inputs, aux vars
        
        ///////////// 1st layer /////////////
        pad2D(train_batch, train_x_in_conv1, batch_size, num_channels_conv1, in_height_conv1, in_width_conv1, padding_left_conv1, padding_right_conv1, padding_top_conv1, padding_bottom_conv1);
        
        tparallel_conv5(train_x_in_conv1, train_weights_conv1, train_in_conv1, batch_size, num_channels_conv1, padded_height_input_conv1 , padded_width_input_conv1, depth_conv1, out_height_conv1, out_width_conv1, filter_size_conv1, stride_conv1, false);
        
        tparallel_matrix_add_depthwise(train_in_conv1, train_biases_conv1, batch_size, out_size_conv1, depth_conv1);
        
        tparallel_relu(train_out_conv1, train_out_conv1_drv, train_in_conv1, batch_size, out_vol_size_conv1);
        ////////////////////////////////////        
        
        ///////////// 2nd layer ////////////
        pad2D(train_out_conv1, train_x_in_conv2, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, padding_left_conv2, padding_right_conv2, padding_top_conv2, padding_bottom_conv2);
        
        tparallel_conv5(train_x_in_conv2, train_weights_conv2, train_in_conv2, batch_size, num_channels_conv2, padded_height_input_conv2, padded_width_input_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);        
        
        tparallel_matrix_add_depthwise(train_in_conv2, train_biases_conv2, batch_size, out_size_conv2, depth_conv2);
        
        tparallel_relu(train_out_conv2, train_out_conv2_drv, train_in_conv2, batch_size, out_vol_size_conv2);
        ////////////////////////////////////        

        ///////////// 3rd layer ////////////
        tparallel_matrix_multiply(train_out_conv2, train_weights_hidden3, train_wx_hidden3, batch_size, in_size_hidden3, num_hidden_nodes3);
        tparallel_matrix_add_row(train_wx_hidden3, train_biases_hidden3, batch_size, num_hidden_nodes3);
        tparallel_relu(train_out_hidden3, train_out_hidden3_drv, train_wx_hidden3, batch_size, num_hidden_nodes3);
        ////////////////////////////////////        

        ///////////// 4th layer ////////////
        tparallel_matrix_multiply(train_out_hidden3, train_weights_output4, train_wx_output4, batch_size, in_size_output4, out_size_output4);
        tparallel_matrix_add_row(train_wx_output4, train_biases_output4, batch_size, out_size_output4);
        tparallel_softmax(train_out_output4, train_wx_output4, train_out_output4_sum, batch_size, out_size_output4);  
        ////////////////////////////////////        

        
        ///////////// Backpropagation ////////////
        
        // First calculation of loss function derivative relative to input of output nodes. Will be used in subsequent backward passes
        
        #pragma acc parallel loop collapse(2)
        for(int i = 0; i < batch_size; i++) {
            for(int m = 0; m < out_size_output4; m++ ) {
                drv_loss_input_output4[i*out_size_output4 + m] = train_out_output4[i*out_size_output4 + m] - train_labels_batch[i*out_size_output4 + m];
            }
        }
        
        //// Calculate error derivatives on weights, biases
        // backpropagation layer 4
        backprop_fc(drv_loss_weights_output4, drv_loss_biases_output4, drv_loss_output_hidden3, drv_loss_input_hidden3, drv_loss_input_output4, train_out_hidden3, train_out_hidden3_drv, train_weights_output4, in_size_output4, out_size_output4, batch_size, false);
        
        // backpropagation layer 3
        backprop_fc(drv_loss_weights_hidden3, drv_loss_biases_hidden3, drv_loss_output_conv2, drv_loss_input_conv2, drv_loss_input_hidden3, train_out_conv2, train_out_conv2_drv, train_weights_hidden3, in_size_hidden3, out_size_hidden3, batch_size, false);
        
        // TODO: Merge backprops for conv, seperate header, library for math calculations, reuse on eval.cpp
        // backpropagation layer 2
        backprop_conv_weights(drv_loss_weights_conv2, drv_loss_biases_conv2, train_x_in_conv2, drv_loss_input_conv2, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, depth_conv2, out_width_conv2, out_height_conv2, filter_size_conv2, padded_height_input_conv2, padded_width_input_conv2, stride_conv2, false );

        backprop_conv_input(drv_loss_output_conv1, drv_loss_input_conv1, drv_loss_input_conv2, train_weights_conv2, train_out_conv1_drv, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);
        
        // backpropagation layer 1
        backprop_conv_weights(drv_loss_weights_conv1, drv_loss_biases_conv1, train_x_in_conv1, drv_loss_input_conv1, batch_size, num_channels_conv1, in_height_conv1, in_width_conv1, depth_conv1, out_width_conv1, out_height_conv1, filter_size_conv1, padded_height_input_conv1, padded_width_input_conv1, stride_conv1,  false );
        
        // Update weights, biases with calculated deltas
        backprop_update(train_weights_output4, drv_loss_weights_output4, in_size_output4*out_size_output4, learning_rate, batch_size);
        backprop_update(train_biases_output4, drv_loss_biases_output4, out_size_output4, learning_rate, batch_size);
        
        backprop_update(train_weights_hidden3, drv_loss_weights_hidden3, in_size_hidden3*out_size_hidden3, learning_rate, batch_size);
        backprop_update(train_biases_hidden3, drv_loss_biases_hidden3, out_size_hidden3, learning_rate, batch_size);

        backprop_update(train_weights_conv2, drv_loss_weights_conv2, filter_flat_size_conv2, learning_rate, batch_size);
        backprop_update(train_biases_conv2, drv_loss_biases_conv2, depth_conv2, learning_rate, batch_size);

        backprop_update(train_weights_conv1, drv_loss_weights_conv1, filter_flat_size_conv1, learning_rate, batch_size);
        backprop_update(train_biases_conv1, drv_loss_biases_conv1, depth_conv1, learning_rate, batch_size);
        ///////////////////////////////////////
        
    }
    
    duration = clock() - start;
    
    double exectime = duration/CLOCKS_PER_SEC;
    double steptime = exectime/steps;
    cout << "Duration: " << std::setprecision(15) << std::fixed << exectime << endl;
    cout << "Step duration: " << std::setprecision(15) << std::fixed << steptime << endl;
    }
    
    cout << endl;
    

    return 0;
}
