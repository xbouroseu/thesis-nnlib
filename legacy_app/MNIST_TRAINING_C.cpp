#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <string>
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

void rngnormalf(int n, float *nr) {
    //float *a;
    int i, istat;
    curandGenerator_t g;

    //a = (float *) malloc(n*4);
    #pragma acc parallel loop
    for (i = 0; i < n; i++)
        nr[i] = 0.0f;
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);

    /* Now Normal */
    printf("Should be normal around 0.0\n");
    istat = curandGenerateNormal(g, nr, n, 0.0f, 1.0f);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    float mx = 0, mn = 0;
    
    //#pragma acc parallel loop reduction(max:mx) reduction(min:mn)
    for(int i = 0; i < n; i++) {
        //cout << "i: "<< a[i] << endl;
        if(nr[i] - mx > 0) mx = nr[i];
        if(nr[i] - mn < 0) mn = nr[i];
    }
    
    float range = mx - mn;
    float ml = 0.2f / range;
    
    cout << "Mu; " << ml << endl;
    cout << "Max: " << mx << ", Min: " << mn << endl;

    //#pragma acc parallel loop
    for (i = 0; i < n; i++)
        nr[i] = nr[i] * ml;
    
    float mx2 = 0, mn2 = 0;
    
    //#pragma acc parallel loop reduction(max:mx2) reduction(min:mn2)
    for(int i = 0; i < n; i++) {
        //cout << "i: "<< a[i] << endl;
        if(nr[i] - mx2 > 0) mx2 = nr[i];
        if(nr[i] - mn2 < 0) mn2 = nr[i];
    }
    cout << "New Max: " << mx2 << ", New Min: " << mn2 << endl;
    
    
    istat = curandDestroyGenerator(g);

}

void rngnormal(double *inp, int n, double multiplier) {
    int i, istat;
    curandGenerator_t g;
    int pn = n;
    
    if(pn%2 == 1) pn = pn+1;
    
    float *nr = new float[pn];
    double *nrd = new double[pn];
    
    #pragma acc parallel loop
    for (i = 0; i < pn; i++)
        nr[i] = 0.0f;
    
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    cout << "N: " << n << " pn: " << pn << " m: " << multiplier << " | " << inp << endl;
    /* Now Normal */
    istat = curandGenerateNormal(g, nr, pn, 0.0f, 1.0f);
    
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
    for(int i = 0; i <pn; i++) {
        nrd[i] = nr[i];
    }
    
    double mx = 0, mn = nrd[0];
    
    //#pragma acc parallel loop reduction(max:mx) reduction(min:mn)
    for(int i = 0; i < pn; i++) {
        if(nrd[i] > mx) mx = nrd[i];
        if(nrd[i] < mn) mn = nrd[i];
    }
    
    double range = mx - mn;
    double ml = 2.0f * multiplier / range;
    /*
    cout << "Mu; " << ml << endl;
    cout << "Max: " << mx << ", Min: " << mn << endl;
*/
    //#pragma acc parallel loop
    for (i = 0; i < pn; i++)
        nrd[i] = nrd[i] * ml;
    
    double mx2 = 0, mn2 = 0;
    
    //#pragma acc parallel loop reduction(max:mx2) reduction(min:mn2)
    for(int i = 0; i < pn; i++) {
        //cout << "i: "<< a[i] << endl;
        if(nrd[i] > mx2) mx2 = nrd[i];
        if(nrd[i] < mn2) mn2 = nrd[i];
    }
//     cout << "New Max: " << mx2 << ", New Min: " << mn2 << endl;
    
    for(int i = 0; i < n; i++) {
        inp[i] = nrd[i];
    }
    istat = curandDestroyGenerator(g);

}
void macc_update_self(double *a, int asize) {
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

void emltp(double *arr, double mltp, int asize) {
    #pragma acc parallel loop pcopy(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] *= mltp;
    }
}

double *makeweights(int N, double multp) {
    double *retw = new double[N];
    
    rngnormal(retw, N, multp);
    
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
                                bdhwsum += conv_input[ (i*in_channels + ch)*in_height*in_width + (oh*stride + di)*in_width + ow*stride + dj ] * conv_filters[ (d*in_channels + ch)*filter_size*filter_size + di*filter_size + dj ];
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
    
    // delta weights transform C D F1 F2 - > delta weights D C F1 F2
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
                    weights_transform[(ch*och + d)*filter_size * filter_size + (filter_size - 1 - f1)*filter_size + filter_size - 1 - f2] = weights[(d*ich + ch)*filter_size * filter_size + f1*filter_size + f2];
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

double dur(double start) {
    return (clock() - start)/CLOCKS_PER_SEC;
}

int main(int argc, char *argv[]) {
    std::cout << "Hello World convolutions" << std::endl;    
    
    int num_images = 0, num_labels = 0, img_size = 0, num_rows = 0, num_cols = 0;
    
    // Load the data
    uchar **train_img_data = read_mnist_images("data/train-images-idx3-ubyte", num_images, img_size, num_rows, num_cols);
    uchar *train_labels_data = read_mnist_labels("data/train-labels-idx1-ubyte", num_labels);

//     printf("Num images: %d, img_size: %d, num_rows:%d, num_cols: %sd, num_labels: %d, dml: %f\n", num_images, img_size, num_rows, num_cols, num_labels, 3);

    // Transform input to one-dimensional array, labels to 1-hot encoding
    double dml = 1;
    int num_samples = num_images, num_inputs = img_size, num_channels = 1, num_outputs = 10;
    
    double *train_images = data2mono(train_img_data, num_images, img_size, dml);
    double *train_labels = labels1hot(train_labels_data, num_labels, num_outputs);
    
    // TODO backprop divide by batch size, emltp? original
    int batch_size = 32, num_hidden_nodes3 = 256;
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
    
//     calc_conv_padding(in_height_conv1, in_width_conv1, filter_size_conv1, stride_conv1, out_height_conv1, out_width_conv1, padding_left_conv1, padding_right_conv1, padding_top_conv1, padding_bottom_conv1);
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
    zeros(train_x_in_conv1, batch_size * padded_width_input_conv1 * padded_height_input_conv1 * num_channels_conv1);
    
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
    zeros(train_x_in_conv2, batch_size * num_channels_conv2 * padded_width_input_conv2 * padded_height_input_conv2);

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
    
    printf("Batch size: %d | Img size: %d | num_rows: %d | num_cols: %d \n",batch_size ,img_size, num_rows, num_cols);
    
    cout << "Layer 1 ||| " << "Input size: 28 x 28" << " | Padded input size: " << padded_width_input_conv1 << " x " << padded_height_input_conv1 << " | Num channels: " << num_channels_conv1 << " | Depth: " << depth_conv1 << " | filter size: " << filter_size_conv1 << " x " << filter_size_conv1 << " | stride: " << stride_conv1 << " | output size: " << out_width_conv1 << " x " << out_height_conv1 << endl;
    
    cout << "Layer 2 ||| " << "Input size: " << out_width_conv1 << " x " << out_height_conv1 << " | Padded input size: " << padded_width_input_conv2 << " x " << padded_height_input_conv2 << " | Num channels: " << num_channels_conv2 << " | Depth: " << depth_conv2 << " | filter size: " << filter_size_conv2 << " x " << filter_size_conv2 << " | stride: " << stride_conv2 << " | output size: " << out_width_conv2 << " x " << out_height_conv2 << endl;
    
    cout << "Layer 3 ||| " << "Num hidden nodes: " << num_hidden_nodes3 << endl;
    
    cout << "Layer 4 ||| " << "Num outputs: " << num_outputs << endl << endl;

    cout << "Steps: " << steps << " | learning rate: " << learning_rate << endl;
    clock_t start;
    double duration;
    
    cout << endl;
    
    cout << "begin" << endl;
    
    // Allocate data in GPU and copy back when done
    #pragma acc data \
    copyin( train_images[:(num_samples*num_inputs)], \
            train_labels[:num_samples*num_outputs], \
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
        
        // clock_t inst = clock();
        ///////////// 1st layer /////////////
        pad2D(train_batch, train_x_in_conv1, batch_size, num_channels_conv1, in_height_conv1, in_width_conv1, padding_left_conv1, padding_right_conv1, padding_top_conv1, padding_bottom_conv1);
        
        tparallel_conv5(train_x_in_conv1, train_weights_conv1, train_in_conv1, batch_size, num_channels_conv1, padded_height_input_conv1 , padded_width_input_conv1, depth_conv1, out_height_conv1, out_width_conv1, filter_size_conv1, stride_conv1, false);
        
        tparallel_matrix_add_depthwise(train_in_conv1, train_biases_conv1, batch_size, out_size_conv1, depth_conv1);
        
        tparallel_relu(train_out_conv1, train_out_conv1_drv, train_in_conv1, batch_size, out_vol_size_conv1);
        ////////////////////////////////////        
        // cout << "Forward 0 duration: " << dur(inst) << endl; inst=clock();
        ///////////// 2nd layer ////////////
        pad2D(train_out_conv1, train_x_in_conv2, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, padding_left_conv2, padding_right_conv2, padding_top_conv2, padding_bottom_conv2);
        
        tparallel_conv5(train_x_in_conv2, train_weights_conv2, train_in_conv2, batch_size, num_channels_conv2, padded_height_input_conv2, padded_width_input_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);        
        
        tparallel_matrix_add_depthwise(train_in_conv2, train_biases_conv2, batch_size, out_size_conv2, depth_conv2);
        
        tparallel_relu(train_out_conv2, train_out_conv2_drv, train_in_conv2, batch_size, out_vol_size_conv2);
        ////////////////////////////////////        
        // cout << "Forward 1 duration: " << dur(inst) << endl; inst=clock();
        ///////////// 3rd layer ////////////
        tparallel_matrix_multiply(train_out_conv2, train_weights_hidden3, train_wx_hidden3, batch_size, in_size_hidden3, num_hidden_nodes3);
        tparallel_matrix_add_row(train_wx_hidden3, train_biases_hidden3, batch_size, num_hidden_nodes3);
        tparallel_relu(train_out_hidden3, train_out_hidden3_drv, train_wx_hidden3, batch_size, num_hidden_nodes3);
        ////////////////////////////////////        
        // cout << "Forward 2 duration: " << dur(inst) << endl; inst=clock();
        ///////////// 4th layer ////////////
        tparallel_matrix_multiply(train_out_hidden3, train_weights_output4, train_wx_output4, batch_size, in_size_output4, out_size_output4);
        tparallel_matrix_add_row(train_wx_output4, train_biases_output4, batch_size, out_size_output4);
        tparallel_softmax(train_out_output4, train_wx_output4, train_out_output4_sum, batch_size, out_size_output4);  
        ////////////////////////////////////        
        // cout << "Forward 3 duration: " << dur(inst) << endl; inst=clock();
        
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
        // cout << "Backward 3 duration: " << dur(inst) << endl; inst=clock();
        // backpropagation layer 3
        backprop_fc(drv_loss_weights_hidden3, drv_loss_biases_hidden3, drv_loss_output_conv2, drv_loss_input_conv2, drv_loss_input_hidden3, train_out_conv2, train_out_conv2_drv, train_weights_hidden3, in_size_hidden3, out_size_hidden3, batch_size, false);
        // cout << "Backward 2 duration: " << dur(inst) << endl; inst=clock();
        // TODO: Merge backprops for conv, seperate header, library for math calculations, reuse on eval.cpp
        // backpropagation layer 2
        backprop_conv_weights(drv_loss_weights_conv2, drv_loss_biases_conv2, train_x_in_conv2, drv_loss_input_conv2, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, depth_conv2, out_width_conv2, out_height_conv2, filter_size_conv2, padded_height_input_conv2, padded_width_input_conv2, stride_conv2, false );

        backprop_conv_input(drv_loss_output_conv1, drv_loss_input_conv1, drv_loss_input_conv2, train_weights_conv2, train_out_conv1_drv, batch_size, num_channels_conv2, in_height_conv2, in_width_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);
        // cout << "Backward 1 duration: " << dur(inst) << endl; inst=clock();
        // backpropagation layer 1
        backprop_conv_weights(drv_loss_weights_conv1, drv_loss_biases_conv1, train_x_in_conv1, drv_loss_input_conv1, batch_size, num_channels_conv1, in_height_conv1, in_width_conv1, depth_conv1, out_width_conv1, out_height_conv1, filter_size_conv1, padded_height_input_conv1, padded_width_input_conv1, stride_conv1,  false );
        // cout << "Backward 0 duration: " << dur(inst) << endl; inst=clock();
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
