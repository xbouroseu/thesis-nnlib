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
    //float *a;
    int i, istat;
    curandGenerator_t g;
    int pn = n;
    if(pn%2 == 1) pn = pn+1;
    float *nr = new float[pn];
    double *nrd = new double[pn];
    //a = (float *) malloc(n*4);
    #pragma acc parallel loop
    for (i = 0; i < pn; i++)
        nr[i] = 0.0f;
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    cout << "N: " << n << " pn: " << pn << " m: " << multiplier << " | " << inp << endl;
    /* Now Normal */
    //printf("Should be normal around 0.0\n");
    istat = curandGenerateNormal(g, nr, pn, 0.0f, 1.0f);
    
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
    for(int i = 0; i <pn; i++) {
        nrd[i] = nr[i];
    }
    
    double mx = 0, mn = nrd[0];
    
    //#pragma acc parallel loop reduction(max:mx) reduction(min:mn)
    for(int i = 0; i < pn; i++) {
        //cout << "i: "<< a[i] << endl;
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

void zerosDD(double *arr, int asize) {
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

double *makeweights(int A, int B, double multp) {
    double *retw = new double[A*B];
    
    rngnormal(retw, A*B, multp);
    
    return retw;
}

double *makebiases(int A, double initl) {
    double *retb = new double[A];
    
    #pragma acc parallel loop pcopyout(retb[:A])
    for(int i = 0; i < A; i++) {
        retb[i] = initl;
    }
    
    return retb;
}

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
            #pragma acc loop seq
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
 * Description: Performs the matrix multiplication 'A * B' and stores the result into matrix 'C' multiplied by a given scalar multiplier
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
            #pragma acc loop seq
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            C[i*M + j] = csumd * mlt;
        }
    }
    
    }
}

/*
 * Name: tparallel_matrix_multiply
 * Description: Performs the matrix multiplication 'A * B' and afterwards a hadamard product '(A * B) * H' and stores the result into matrix 'C'
 * Parameters: {
 *      A: N x K size matrix containing the elements of the first matrix operand, must be 1-dimensional
 *      B: K x M size matrix containing the elements of the second matrix operand, must be 1-dimensional
 *      C: N x M size matrix containing the elements of the  matrix multiplication operation C = A * B, must be 1-dimensional
 *      H: N x M size matrix containing the elements of the second operand in the hadamard product
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
            #pragma acc loop seq
            for(int t = 0; t < K; t++) {
                csumd += A[i*K + t]*B[t*M + j];
            }
            C[i*M + j] = csumd * H[i*M + j];
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

void tparallel_matrix_add_depth(double *A, double *B, int N, int K, int D) {
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
 *     train_output: N x M size matrix containing the output of the softmax operation
 *     softmax_input: N x M size matrix containing the input to the softmax operation
 *     train_output_sum: auxiliary placeholder containing the calculated sum of the softmax denominator
 * }
 *
*/
void tparallel_softmax(double *train_output, double *softmax_input, double *train_output_sum, int batch_size, int num_outputs) {
    #pragma acc data pcopyin(softmax_input[:batch_size*num_outputs]) pcopyout(train_output[:batch_size*num_outputs]) pcopyout(train_output_sum[:batch_size])
    {

      
    #pragma acc parallel loop
    for(int i = 0; i < batch_size; i++) {
        train_output_sum[i] = 0.0f;
    }
    
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
//         double sumd = 0.0f;
//         #pragma acc loop reduction(+:sumd)
        for(int j = 0; j < num_outputs; j++) {
            //sumd += exp(softmax_input[i*num_outputs + j]);
            #pragma acc atomic update
            train_output_sum[i] += exp(softmax_input[i*num_outputs + j]);
        }
        
        //train_output_sum[i] = sumd;
    }
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_outputs; j++) {            
            #pragma acc atomic write
            train_output[i*num_outputs + j] = exp(softmax_input[i*num_outputs + j]) / train_output_sum[i];
        }
    }
    }
}

/*
 * Name: tparallel_relu
 * Description: Performs a 'Linear rectified Unit' - ReLU function on the input and stores the result to the output
 *
 * Parameters: {
 *     train_output: N x M size matrix containing the output of the ReLU operation
*      train_output_drv: N x M size matrix containing the derivative of the output of the ReLU operation
 *     rl_input: N x M size matrix containing the input to the ReLU operation
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
void tparallel_relu(double *train_output, double *train_output_drv, double *rl_input, int batch_size, int num_outputs) {    
    #pragma acc data pcopyin(rl_input[:batch_size*num_outputs]) pcopyout(train_output[:batch_size*num_outputs], train_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = rl_input[i*num_outputs + j];
   
            if(val > 0) {
                train_output[i*num_outputs + j] = val;
                train_output_drv[i*num_outputs + j] = 1.0f;
            }
            else {
                train_output[i*num_outputs + j] = 0.0f;
                train_output_drv[i*num_outputs + j] = 0.0f;
            }
        }
    }
    }
}

void tparallel_relu_dummy(double *train_output, double *rl_input, int batch_size, int num_outputs) {    
    #pragma acc data pcopyin(rl_input[:batch_size*num_outputs]) pcopyout(train_output[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = rl_input[i*num_outputs + j];
   
            if(val > 0) {
                train_output[i*num_outputs + j] = val;
            }
            else {
                train_output[i*num_outputs + j] = 0.0f;
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
 *     train_output: N x M size matrix containing the output of the sigmoid function
*      train_output_drv: N x M size matrix containing the derivative of the output of the sigmoid function
 *     sg_input: N x M size matrix containing the input to the sigmoid function
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
void tparallel_sigmoid(double *train_output, double *train_output_drv, double *sg_input, int batch_size, int num_outputs) {    
    #pragma acc data pcopyin(sg_input[:batch_size*num_outputs]) pcopyout(train_output[:batch_size*num_outputs], train_output_drv[:batch_size*num_outputs])
    {
    
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        //#pragma acc loop
        for(int j = 0; j < num_outputs; j++) {
            double val = 1/(1 + exp(-1 * sg_input[i*num_outputs + j]));

            train_output[i*num_outputs + j] = val;
            train_output_drv[i*num_outputs + j] = val*(1-val);
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
void tparallel_conv5(double *conv_input, double *conv_filters, double *conv_output, int B, int in_channels, int in_height, int in_width, int depth, int out_height, int out_width, int filter_size, int stride, bool debug) { 
    
    
    #pragma acc data pcopyin(conv_input[:(B*in_width*in_height*in_channels)], conv_filters[:in_channels*filter_size*filter_size*depth]) pcopyout(conv_output[:(B * out_width * out_height * depth)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int i = 0; i < B; i++) {
        for(int d = 0; d < depth; d++) {
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
                    
                    conv_output[(i*depth + d)*out_width*out_height + oh*out_width + ow] = bdhwsum;
                }
            }
        }
    }
    
    }
    
}

void prepare_conv(int batch_size, int in_rows, int in_cols, int num_channels, int filter_size, int stride, int padding, int &pad_height, int &pad_width, int &pad_left, int &pad_right, int &pad_top, int &pad_bottom, int &out_height, int &out_width) {

    if(padding == 1) {
        if(in_rows%stride == 0) {
            pad_height = max(filter_size - stride, 0);
        }
        else {
            pad_height = max(filter_size - (in_rows%stride), 0);
        }

        if(in_cols%stride == 0) {
            pad_width = max(filter_size - stride, 0);
        }
        else {
            pad_width = max(filter_size - (in_cols%stride), 0);
        }
        
        pad_top = pad_height/2;
        pad_bottom = pad_height - pad_top;
        pad_left = pad_width/2;
        pad_right = pad_width - pad_left;
        
        out_width = ceil((1.0f * in_cols)/stride);
        out_height = ceil((1.0f * in_rows)/stride);
    }
    else {
        pad_width = 0;
        pad_height = 0;
        pad_top = 0;
        pad_bottom = 0;
        pad_left = 0;
        pad_right = 0;
        
        out_width = ceil( (1.0f * (in_cols - filter_size + 1))/stride );
        out_height = ceil( (1.0f * (in_rows - filter_size + 1))/stride );
    }
}

void make_batch(double * input, double *batch, int batch_size, int num_inputs, int batch_start) {
    #pragma acc data pcopyin(input[(batch_start*num_inputs):(batch_size*num_inputs)]) pcopy(batch[:(batch_size*num_inputs)])
    {
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < num_inputs; k++) {
            batch[i*num_inputs + k] = input[(i+batch_start)*num_inputs + k];
        }
    }
        
    }
}

void copypad(double *input, double *pad_input, int N, int M, int D, int B, int pad_left, int pad_right, int pad_top, int pad_bottom, int padded_N, int padded_M) {
    #pragma acc data pcopyin(input[:(B*D*M*N)]) pcopy(pad_input[:(B*D*padded_N*padded_M)])
    {
        
    #pragma acc parallel loop collapse(4) 
    for(int b = 0; b < B; b++) {
        for(int d = 0; d < D; d++) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    pad_input[(b*D + d)*padded_N*padded_M + (i+pad_top)*padded_M + j + pad_left] = input[b*D*M*N + d*M*N + i*M + j];
                }
            }
        }
    }
    
    }
}

void backprop_fc(double *delta_weights, double *delta_biases, double *layer0_delta_out, double *layer0_delta_in, double *layer1_delta_in, double *layer0_out, double *layer0_out_drv, double *weights, int layer0_size, int layer1_size, int batch_size, bool debug) {
        
    double mltp = (1.0f/(double)batch_size);
    double *layer0_out_transpose = new double[batch_size*layer0_size];
    double *weights_transpose = new double[layer0_size*layer1_size];
    
    #pragma acc data pcopyout(delta_weights[:layer0_size*layer1_size], delta_biases[:layer1_size], layer0_delta_in[:batch_size*layer0_size], layer0_delta_out[:batch_size*layer0_size])\
    pcopyin(layer1_delta_in[:batch_size*layer1_size], layer0_out[:batch_size*layer0_size], layer0_out_drv[:batch_size*layer0_size], weights[:layer0_size*layer1_size]) \
    pcreate(layer0_out_transpose[:batch_size*layer0_size], weights_transpose[:layer0_size*layer1_size])
    {
    //delta biases 
    #pragma acc parallel loop
    for(int m = 0; m < layer1_size; m++) {
        double sumdb = 0.0f;
        #pragma acc loop reduction(+:sumdb)
        for(int i = 0; i < batch_size; i++) {
            sumdb += layer1_delta_in[i*layer1_size + m];
        }
        delta_biases[m] = sumdb * mltp;
    }
    
 

    // delta weights
    
    transpose(layer0_out, layer0_out_transpose, batch_size, layer0_size);
    
    tparallel_matrix_multiply_mltp(layer0_out_transpose, layer1_delta_in, delta_weights, layer0_size, batch_size, layer1_size,  mltp );

    // delta out layer 0
    
    transpose(weights, weights_transpose, layer0_size, layer1_size);
    
    tparallel_matrix_multiply(layer1_delta_in, weights_transpose, layer0_delta_out, batch_size, layer1_size, layer0_size);    
    
    tparallel_matrix_hadamard(layer0_delta_out, layer0_out_drv, layer0_delta_in, batch_size, layer0_size);

    }
    
    free(weights_transpose);
    free(layer0_out_transpose);
    
}

void backprop_conv_weights(double *delta_weights, double *delta_biases, double *conv_in, double *layer1_delta_in, int batch_size, int in_channels, int in_height, int in_width, int depth, int out_width, int out_height, int filter_size, int padded_height, int padded_width, int stride, bool debug ) {
    
    double mltp = 1/((double)batch_size);
    int padded_size = padded_width * padded_height;
    int out_size = out_height * out_width;
    int bs =  batch_size;
    int ich = in_channels;
    int ps = padded_width * padded_height;
    int dd = depth;
    
    double *delta_weights_transform = new double[(depth*in_channels*filter_size*filter_size)];
    double *conv_in_transform = new double[(batch_size * in_channels * padded_size)];
    double *layer1_delta_in_transform = new double[(batch_size * depth * out_size)];
    
    #pragma acc data \
    pcopyout(delta_weights[:(depth*in_channels*filter_size*filter_size)], delta_biases[:depth]) pcopyin(conv_in[:(batch_size * in_channels * padded_size)]) pcopyin(layer1_delta_in[:(batch_size * depth * out_size)]) \
    pcopyin(delta_weights_transform[:(depth*in_channels*filter_size*filter_size)], conv_in_transform[:(batch_size * in_channels * padded_size)], layer1_delta_in_transform[:(batch_size * depth * out_size)])
    {

    // conv in B C S -> C B S1
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < bs; i++) {
        for(int ch = 0; ch < ich; ch++) {
            for(int s = 0; s < ps; s++) {
                conv_in_transform[(ch*bs+ i)*ps + s] = conv_in[(i*ich + ch)*ps + s];
            }
        }
    }
    
    // delta in B D S2 -> D B S2
    #pragma acc parallel loop collapse(3)
    for(int i = 0; i < bs; i++) {
        for(int d = 0; d < dd; d++) {
            for(int s = 0; s < out_size; s++) {
                layer1_delta_in_transform[(d*bs + i)*out_size + s] = layer1_delta_in[(i*dd + d)*out_size + s];
            }
        }
    }
    

    tparallel_conv5(conv_in_transform, layer1_delta_in_transform, delta_weights_transform, in_channels, batch_size, padded_height, padded_width, depth, filter_size, filter_size, out_width, stride, debug);
    emltp(delta_weights_transform, mltp, in_channels*depth*filter_size*filter_size);
    
    // delta weights transform D C F1 F2 - > C D F1 F2
    #pragma acc parallel loop collapse(4)
    for(int ch = 0; ch < ich; ch++) {
        for(int d = 0; d < dd; d++) {
            for(int f1 = 0; f1 < filter_size; f1++) {
                for(int f2 = 0; f2 < filter_size; f2++) {
                    delta_weights[(d*ich + ch)*filter_size*filter_size + f1*filter_size + f2] = delta_weights_transform[ (ch*dd + d)*filter_size*filter_size + f1*filter_size + f2];
                }
            }
        }
    }
    
    #pragma acc parallel loop
    for(int d = 0; d < depth; d++) {
        double sumdb = 0.0f;
        
        #pragma acc loop seq reduction(+:sumdb)
        for(int i = 0; i < batch_size; i++) {
            double sumdbi = 0.0f;
            
            #pragma acc loop reduction(+:sumdbi)
            for(int c = 0; c < out_size; c++) {
                sumdbi += layer1_delta_in[ (i*depth + d)*out_size + c];
            }
            
            sumdb += sumdbi;
        }
        
        delta_biases[d] = sumdb * mltp;
        
    }
    
    }
    
    free(delta_weights_transform);
    free(conv_in_transform);
    free(layer1_delta_in_transform);
}

void backprop_conv_input(double *layer0_delta_out, double *layer0_delta_in, double *layer1_delta_in, double *weights, double *layer0_out_drv, int batch_size, int in_channels, int in_height, int in_width, int depth, int out_width, int out_height, int filter_size, int padded_height, int padded_width, int stride, bool debug ) {
    
    int pad_left = 2, pad_top = 2, pad_right = 2, pad_bottom = 2;
    int padded_out_height = out_height + pad_top + pad_bottom, padded_out_width = out_width + pad_left + pad_right;
    
    int bs =  batch_size;
    int ich = in_channels;
    int ps = padded_width * padded_height;
    int dd = depth;
    
    double *weights_transform = new double[in_channels * depth * filter_size * filter_size];
    double *layer1_delta_in_padded = new double[batch_size * depth * padded_out_height * padded_out_width];
    
    #pragma acc data\
    pcopyout(layer0_delta_out[:(batch_size*in_channels*in_height*in_width)], layer0_delta_in[:(batch_size*in_channels*in_height*in_width)]) \
    pcopyin(layer1_delta_in[:(batch_size*depth*out_width*out_height)], weights[:(depth*in_channels*filter_size*filter_size)], layer0_out_drv[:(batch_size*in_channels*in_width*in_height)]) \
    pcreate(weights_transform[:(depth*in_channels*filter_size*filter_size)], layer1_delta_in_padded[:(batch_size * depth * padded_out_height * padded_out_width)])
    {
        
    // pad out delta
    copypad(layer1_delta_in, layer1_delta_in_padded, out_height, out_width, depth, batch_size, pad_left, pad_right, pad_top, pad_bottom, padded_out_height, padded_out_width);
    
    // flip weights x,y , transpose D-C    
    #pragma acc parallel loop collapse(4)
    for(int d = 0; d < dd; d++) {
        for(int ch = 0; ch < ich; ch++) {
            for(int f1 = 0; f1 < filter_size; f1++) {
                for(int f2 = 0; f2 < filter_size; f2++) {
                    weights_transform[(ch*depth + d)*filter_size * filter_size + (filter_size - 1 - f1)*filter_size + filter_size - 1 - f2] = weights[(d*in_channels + ch)*filter_size * filter_size + f1*filter_size + f2];
                }
            }
        }
    }
    
    tparallel_conv5(layer1_delta_in_padded, weights_transform, layer0_delta_out, batch_size, depth, padded_out_height, padded_out_width, in_channels, in_height, in_width, filter_size, stride, false);
    
    tparallel_matrix_hadamard(layer0_delta_out, layer0_out_drv, layer0_delta_in, batch_size, in_channels*in_width*in_height);
    
    }
    
    free(weights_transform);
    free(layer1_delta_in_padded);
}


void backprop_update(double *a, double *da, int asize, double learning_rate) {
    #pragma acc parallel loop pcopyin(da[:asize]) pcopy(a[:asize])
    for(int i = 0; i < asize; i++) {
        a[i] -= learning_rate * da[i];
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

int main(int argc, char *argv[]) {
    std::cout << "Hello World convolutions" << std::endl;    
    
    int num_images = 0, num_labels = 0, img_size = 0, num_rows = 0, num_cols = 0; 
    uchar **train_img_data = read_mnist_images("data/train-images-idx3-ubyte", num_images, img_size, num_rows, num_cols);
    uchar *train_labels_data = read_mnist_labels("data/train-labels-idx1-ubyte", num_labels);
    
    int test_num_images = 0, test_num_labels = 0, test_img_size = 0, test_num_rows = 0, test_num_cols = 0;
    
    uchar **test_img_data = read_mnist_images("data/t10k-images-idx3-ubyte", test_num_images, test_img_size, test_num_rows, test_num_cols);
    uchar *test_labels_data = read_mnist_labels("data/t10k-labels-idx1-ubyte", test_num_labels);
    
    float **c_test_images = convert_train_dataset_f(test_img_data, test_num_images, 28, 28);
    float **c_train_images = convert_train_dataset_f(train_img_data, num_images, 28, 28);
    
    int num_samples = num_images, num_inputs = img_size, num_outputs = 10;
    
    double dml = 1;
    
    double *train_images = data2mono(train_img_data, num_images, img_size, dml);
    double *train_labels = labels1hot(train_labels_data, num_images, num_outputs);
    double *test_images = data2mono(test_img_data, test_num_images, test_img_size, dml);
    double *test_labels = labels1hot(test_labels_data, test_num_images, num_outputs);
   
    int batch_size = 32;
    int num_hidden_nodes3 = 256;
    int num_channels = 1;
    
    double *train_batch = new double[batch_size * num_channels * num_rows * num_cols];
    double *train_labels_batch = new double[batch_size * num_outputs];

    // layer 1 - convolutions
    ////////////////////////////////////////////////////////////////////
    int num_channels_conv1 = 1, filter_size_conv1 = 5, depth_conv1 = 64, stride_conv1 = 1;
    int pad_height_conv1, pad_width_conv1, pad_top_conv1, pad_bottom_conv1, pad_left_conv1, pad_right_conv1;
    int out_width_conv1, out_height_conv1, out_size_conv1, out_vol_size_conv1;
    int wflat_conv1 = filter_size_conv1 * filter_size_conv1 * num_channels_conv1;
    
    prepare_conv(batch_size, 28, 28, num_channels_conv1, filter_size_conv1, stride_conv1, 1, pad_height_conv1, pad_width_conv1, pad_top_conv1, pad_bottom_conv1, pad_left_conv1, pad_right_conv1, out_height_conv1, out_width_conv1);
    
    out_size_conv1 = out_width_conv1 * out_height_conv1;
    out_vol_size_conv1 = out_size_conv1 * depth_conv1;
    
    int p_width_conv1, p_height_conv1;
    p_width_conv1 = num_cols + pad_width_conv1;
    p_height_conv1 = num_rows + pad_height_conv1;
    
    double *train_x_in_conv1 = new double[batch_size * p_width_conv1 * p_height_conv1 * num_channels_conv1];
    zerosDD(train_x_in_conv1, batch_size * p_width_conv1 * p_height_conv1 * num_channels_conv1);
    
    double *train_weights_conv1 = makeweights(wflat_conv1, depth_conv1, 0.1f);
    double *train_biases_conv1 = makebiases(depth_conv1, 0.0f);
    
    double *train_in_conv1 = new double[batch_size * out_vol_size_conv1];
    double *train_out_conv1 = new double[batch_size * out_vol_size_conv1];
    double *train_out_conv1_drv = new double[batch_size * out_vol_size_conv1];

    ///////////////////////////////////////////////////////////////////////////////
    
    // layer 2 - convolutions
    ///////////////////////////////////////////////////////////////////////////
    int num_channels_conv2 = depth_conv1, filter_size_conv2 = 5, depth_conv2 = 64, stride_conv2 = 1;
    int pad_height_conv2, pad_width_conv2, pad_top_conv2, pad_bottom_conv2, pad_left_conv2, pad_right_conv2;
    int out_width_conv2, out_height_conv2, out_size_conv2, out_vol_size_conv2;
    int wflat_conv2 = filter_size_conv2 * filter_size_conv2 * num_channels_conv2;
    
    prepare_conv(batch_size, out_height_conv1, out_width_conv1, num_channels_conv2, filter_size_conv2, stride_conv2, 1, pad_height_conv2, pad_width_conv2, pad_top_conv2, pad_bottom_conv2, pad_left_conv2, pad_right_conv2, out_height_conv2, out_width_conv2);
    
    out_size_conv2 = out_width_conv2 * out_height_conv2;
    out_vol_size_conv2 = out_size_conv2 * depth_conv2;
    
    int p_width_conv2, p_height_conv2;
    p_width_conv2 = out_width_conv1 + pad_width_conv2;
    p_height_conv2 = out_height_conv2 + pad_height_conv2;
        
    double *train_weights_conv2 = makeweights(wflat_conv2, depth_conv2, 0.1f);

    double *train_biases_conv2 = makebiases(depth_conv2, 0.0f);
    
    double *train_x_in_conv2 = new double[batch_size * num_channels_conv2 * p_width_conv2 * p_height_conv2];
    zerosDD(train_x_in_conv2, batch_size * num_channels_conv2 * p_width_conv2 * p_height_conv2);
    
    double *train_in_conv2 = new double[batch_size * out_vol_size_conv2];
    double *train_out_conv2 = new double[batch_size * out_vol_size_conv2];
    double *train_out_conv2_drv = new double[batch_size * out_vol_size_conv2];
    
    ///////////////////////////////////////////////////////////////////////////////////
    
    // layer 3 - relu
    //////////////////////////////////////////////////////////////////////////////////
    double *train_weights_hidden3 = makeweights(out_vol_size_conv2, num_hidden_nodes3, 0.1f);
    double *train_biases_hidden3 = makebiases(num_hidden_nodes3, 0.0f);
    double *train_wx_hidden3 = new double[batch_size * num_hidden_nodes3];
    double *train_out_hidden3 = new double[batch_size * num_hidden_nodes3];
    double *train_out_hidden3_drv = new double[batch_size * num_hidden_nodes3];

    ////////////////////////////////////////////////////////////////////////////////////
    
    // layer 4 - softmax
    ////////////////////////////////////////////////////////////////////////////////
    double *train_weights_output4 = makeweights(num_hidden_nodes3, num_outputs, 0.1f);
    double *train_biases_output4 = makebiases(num_outputs, 0.0f);
    double *train_wx_output4 = new double[batch_size * num_outputs];
    double *train_out_output4 = new double[batch_size * num_outputs];
    double *train_out_output4_sum = new double[batch_size];
    //////////////////////////////////////////////////////////////////////////////////
    
    // backpropagation
    //////////////////////////////////////////////////////////////////////////////////
    double *delta_out_output4 = new double[batch_size * num_outputs];
    double *delta_biases_output4 = new double[num_outputs];
    double *delta_weights_output4 = new double[num_hidden_nodes3 * num_outputs];
    
    double *delta_out_hidden3 = new double[batch_size* num_hidden_nodes3];
    double *delta_in_hidden3 = new double[batch_size* num_hidden_nodes3];
    double *delta_biases_hidden3 = new double[num_hidden_nodes3];
    double *delta_weights_hidden3 = new double[out_vol_size_conv1 * num_hidden_nodes3];

    double *delta_out_conv2 = new double[batch_size * out_vol_size_conv2];
    double *delta_in_conv2 = new double[batch_size * out_vol_size_conv2];
    double *delta_weights_conv2 = new double[wflat_conv2*depth_conv2];

    double *delta_biases_conv2 = new double[depth_conv2];

    double *delta_out_conv1 = new double[batch_size * out_vol_size_conv1];
    double *delta_in_conv1 = new double[batch_size * out_vol_size_conv1];
    double *delta_weights_conv1 = new double[wflat_conv1*depth_conv1];
    double *delta_biases_conv1 = new double[depth_conv1];
    ///////////////////////////////////////////////////////////////////////////////////////
    
    //////////////
    int arg1 = atoi(argv[1]);
    cout  << "Steps: " << arg1 << endl;
//     string step_str = argv[1];
    int steps = arg1;
    double learning_rate = 0.05;
    int batch_start = 0;
    
    cout << endl << "Batch size: " << batch_size << " | Input size: " << endl;
    
    cout << "Layer 1 ||| " << "Input size: 28 x 28" << " | Padded input size: " << p_width_conv1 << " x " << p_height_conv1 << " | Num channels: " << num_channels_conv1 << " | Depth: " << depth_conv1 << " | filter size: " << filter_size_conv1 << " x " << filter_size_conv1 << " | stride: " << stride_conv1 << " | output size: " << out_width_conv1 << " x " << out_height_conv1 << endl;
    
    cout << "Layer 2 ||| " << "Input size: " << out_width_conv1 << " x " << out_height_conv1 << " | Padded input size: " << p_width_conv2 << " x " << p_height_conv2 << " | Num channels: " << num_channels_conv2 << " | Depth: " << depth_conv2 << " | filter size: " << filter_size_conv2 << " x " << filter_size_conv2 << " | stride: " << stride_conv2 << " | output size: " << out_width_conv2 << " x " << out_height_conv2 << endl;
    
    cout << "Layer 3 ||| " << "Num hidden nodes: " << num_hidden_nodes3 << endl;
    
    cout << "Layer 4 ||| " << "Num outputs: " << num_outputs << endl << endl;

    cout << "Steps: " << steps << " | learning rate: " << learning_rate << endl;
    clock_t start;
    double duration;
    
    cout << endl;
    
    cout << "begin" << endl;
    #pragma acc data copyin(train_images[:(num_samples*num_inputs)], train_labels[:num_samples*num_outputs]) copyin(train_batch[:(batch_size*num_inputs)], train_labels_batch[:batch_size*num_outputs]) \
    \
    copyin(train_x_in_conv1[:(batch_size*p_width_conv1*p_height_conv1*num_channels_conv1)]) \
    copy(train_weights_conv1[:wflat_conv1*depth_conv1], train_biases_conv1[:depth_conv1]) \
    create(train_in_conv1[:(batch_size* out_vol_size_conv1)], train_out_conv1[:(batch_size*out_vol_size_conv1)], train_out_conv1_drv[:(batch_size*out_vol_size_conv1)]) \
    \
    copyin(train_x_in_conv2[:(batch_size*p_width_conv2*p_height_conv2*num_channels_conv2)]) \
    copy(train_weights_conv2[:wflat_conv2*depth_conv2], train_biases_conv2[:depth_conv2]) \
    create(train_in_conv2[:(batch_size* out_vol_size_conv2)], train_out_conv2[:(batch_size*out_vol_size_conv2)], train_out_conv2_drv[:(batch_size*out_vol_size_conv2)]) \
    \
    copy(train_weights_hidden3[:(out_vol_size_conv2*num_hidden_nodes3)], train_biases_hidden3[:num_hidden_nodes3]) \
    create(train_wx_hidden3[:batch_size*num_hidden_nodes3], train_out_hidden3[:batch_size*num_hidden_nodes3], train_out_hidden3_drv[:batch_size*num_hidden_nodes3]) \
    \
    copy(train_weights_output4[:num_hidden_nodes3*num_outputs], train_biases_output4[:num_outputs]) \
    create(train_wx_output4[:batch_size*num_outputs], train_out_output4[:batch_size*num_outputs], train_out_output4_sum[:batch_size]) \
    \
    create(delta_biases_output4[:num_outputs], delta_weights_output4[:num_hidden_nodes3*num_outputs], \
    delta_biases_hidden3[:num_hidden_nodes3], delta_weights_hidden3[:out_vol_size_conv2*num_hidden_nodes3], \
    delta_weights_conv2[:wflat_conv2*depth_conv2], delta_biases_conv2[:depth_conv2], \
    delta_weights_conv1[:wflat_conv1*depth_conv1], delta_biases_conv1[:depth_conv1]) \
    \
    create(delta_out_output4[:batch_size*num_outputs], \
    delta_out_hidden3[:batch_size*num_hidden_nodes3], delta_in_hidden3[:batch_size*num_hidden_nodes3], \
    delta_out_conv2[:batch_size*out_vol_size_conv2], delta_in_conv2[:batch_size*out_vol_size_conv2], \
    delta_out_conv1[:batch_size*out_vol_size_conv1], delta_in_conv1[:batch_size*out_vol_size_conv1])
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
        make_batch(train_labels, train_labels_batch, batch_size, num_outputs, batch_start);
        
        // 1st layer
        copypad(train_batch, train_x_in_conv1, num_rows, num_cols, num_channels_conv1, batch_size, pad_left_conv1, pad_right_conv1, pad_top_conv1, pad_bottom_conv1, p_height_conv1, p_width_conv1);
        tparallel_conv5(train_x_in_conv1, train_weights_conv1, train_in_conv1, batch_size, num_channels_conv1, p_height_conv1 , p_width_conv1, depth_conv1, out_height_conv1, out_width_conv1, filter_size_conv1, stride_conv1, false);
        tparallel_matrix_add_depth(train_in_conv1, train_biases_conv1, batch_size, out_size_conv1, depth_conv1);
        tparallel_relu(train_out_conv1, train_out_conv1_drv, train_in_conv1, batch_size, out_vol_size_conv1);
        
        // 2nd layer
        copypad(train_out_conv1, train_x_in_conv2, out_height_conv1, out_width_conv1, num_channels_conv2, batch_size, pad_left_conv2, pad_right_conv2, pad_top_conv2, pad_bottom_conv2, p_height_conv2, p_width_conv2);
        tparallel_conv5(train_x_in_conv2, train_weights_conv2, train_in_conv2, batch_size, num_channels_conv2, p_height_conv2, p_width_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);        
        tparallel_matrix_add_depth(train_in_conv2, train_biases_conv2, batch_size, out_size_conv2, depth_conv2);
        tparallel_relu(train_out_conv2, train_out_conv2_drv, train_in_conv2, batch_size, out_vol_size_conv2);
        
        // 3rd layer
        tparallel_matrix_multiply(train_out_conv2, train_weights_hidden3, train_wx_hidden3, batch_size, out_vol_size_conv2, num_hidden_nodes3);
        tparallel_matrix_add_row(train_wx_hidden3, train_biases_hidden3, batch_size, num_hidden_nodes3);
        tparallel_relu(train_out_hidden3, train_out_hidden3_drv, train_wx_hidden3, batch_size, num_hidden_nodes3);
        
        // 4th layer
        tparallel_matrix_multiply(train_out_hidden3, train_weights_output4, train_wx_output4, batch_size, num_hidden_nodes3, num_outputs);
        tparallel_matrix_add_row(train_wx_output4, train_biases_output4, batch_size, num_outputs);
        tparallel_softmax(train_out_output4, train_wx_output4, train_out_output4_sum, batch_size, num_outputs);  
        
        #pragma acc parallel loop collapse(2)
        for(int i = 0; i < batch_size; i++) {
            for(int m = 0; m < num_outputs; m++ ) {
                delta_out_output4[i*num_outputs + m] = train_out_output4[i*num_outputs + m] - train_labels_batch[i*num_outputs + m];
            }
        }
        
        // backpropagation layer 4
        backprop_fc(delta_weights_output4, delta_biases_output4, delta_out_hidden3, delta_in_hidden3, delta_out_output4, train_out_hidden3, train_out_hidden3_drv, train_weights_output4, num_hidden_nodes3, num_outputs, batch_size, false);
        
        // backpropagation layer 3
        backprop_fc(delta_weights_hidden3, delta_biases_hidden3, delta_out_conv2, delta_in_conv2, delta_in_hidden3, train_out_conv2, train_out_conv2_drv, train_weights_hidden3, out_vol_size_conv2, num_hidden_nodes3, batch_size, false);
        
        // backpropagation layer 2
        backprop_conv_weights(delta_weights_conv2, delta_biases_conv2, train_x_in_conv2, delta_in_conv2, batch_size, num_channels_conv2, out_height_conv1, out_width_conv1, depth_conv2, out_width_conv2, out_height_conv2, filter_size_conv2, p_height_conv2, p_width_conv2, stride_conv2, false );
        
        // backpropagation layer 1
        backprop_conv_input(delta_out_conv1, delta_in_conv1, delta_in_conv2, train_weights_conv2, train_out_conv1_drv, batch_size, num_channels_conv2, out_height_conv1, out_width_conv1, depth_conv2, out_width_conv2, out_height_conv2, filter_size_conv2, p_height_conv2, p_width_conv2, stride_conv2, false);
        
        backprop_conv_weights(delta_weights_conv1, delta_biases_conv1, train_x_in_conv1, delta_in_conv1, batch_size, num_channels_conv1, num_rows, num_cols, depth_conv1, out_width_conv1, out_height_conv1, filter_size_conv1, p_height_conv1, p_width_conv1, stride_conv1,  false );
        
        backprop_update(train_weights_output4, delta_weights_output4, num_hidden_nodes3*num_outputs, learning_rate);
        backprop_update(train_biases_output4, delta_biases_output4, num_outputs, learning_rate);
        
        backprop_update(train_weights_hidden3, delta_weights_hidden3, out_vol_size_conv2*num_hidden_nodes3, learning_rate);
        backprop_update(train_biases_hidden3, delta_biases_hidden3, num_hidden_nodes3, learning_rate);

        backprop_update(train_weights_conv2, delta_weights_conv2, num_channels_conv2*depth_conv2*filter_size_conv2*filter_size_conv2, learning_rate);
        backprop_update(train_biases_conv2, delta_biases_conv2, depth_conv2, learning_rate);

        backprop_update(train_weights_conv1, delta_weights_conv1, num_channels_conv1*depth_conv1*filter_size_conv1*filter_size_conv1, learning_rate);
        backprop_update(train_biases_conv1, delta_biases_conv1, depth_conv1, learning_rate);

    }
    
    duration = clock() - start;
    
    cout << "Duration: " << std::setprecision(15) << std::fixed << duration/CLOCKS_PER_SEC << endl;
    }
    
    cout << endl;
    
    bool save_network = true;
    // save network configuration in file
    if(save_network) {
        clock_t tmstmp = clock();
        time_t rawtime;
        struct tm *timeinfo;
        time(&rawtime);
        timeinfo = localtime (&rawtime);
        
        string tmss = ctime(&rawtime);
        
        char time_string[80];
        strftime(time_string, 80, "%F_%T", timeinfo);
        
        cout << "Saving network: " << tmstmp << endl;
//         cout << rawtime << endl << tmss << endl << time_string << endl;
        string iters = to_string(steps);
        string time_str(time_string);
        string netname = "NEURAL_NETWORK_TRAINED_" + iters + "_" + time_str + ".csv";
        
        ofstream out_param;
        out_param.open(netname, ios::out | ios::app);
        
        out_param << 4 << endl;
        
        // TODO: ADD to save network training configuration, steps etc.
        
        // input
        out_param << num_channels_conv1 << ";28;28" << endl;
        
        out_param << "1;3;2;" << depth_conv1 << ";" << num_channels_conv1 << ";" << filter_size_conv1 << ";" << filter_size_conv1 << ";" << stride_conv1 << ";0" << endl;
        out_param << "2;3;2;" << depth_conv2 << ";" << num_channels_conv2 << ";" << filter_size_conv2 << ";" << filter_size_conv2 << ";" << stride_conv2 << ";0" << endl;
        out_param << "3;2;2;" << out_vol_size_conv2 << ";" << num_hidden_nodes3 << endl;
        out_param << "4;1;1;" << num_hidden_nodes3 << ";" << num_outputs << endl;
        
        // conv 1
        param2file_csv(train_weights_conv1, netname, 1, wflat_conv1*depth_conv1, 1);
        param2file_csv(train_biases_conv1, netname, 2, depth_conv1, 1);
        
        // conv 2
        param2file_csv(train_weights_conv2, netname, 1, wflat_conv2*depth_conv2, 2);
        param2file_csv(train_biases_conv2, netname, 2, depth_conv2, 2);
        
        // hidden 3
        param2file_csv(train_weights_hidden3, netname, 1, out_vol_size_conv2 * num_hidden_nodes3, 3);
        param2file_csv(train_biases_hidden3, netname, 2, num_hidden_nodes3, 3);
        
        // output 4
        param2file_csv(train_weights_output4, netname, 1, num_hidden_nodes3*num_outputs, 4);
        param2file_csv(train_biases_output4, netname, 2, num_outputs, 4);
        
        out_param.close();
    }
    
    /*
    int a = 0;
    
    #pragma acc parallel loop
    for(int i = 0; i < 1000; i++) {
        a = a + 2;
    }*/
    return 0;
}
