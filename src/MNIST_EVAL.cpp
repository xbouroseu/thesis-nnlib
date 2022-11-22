#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <sstream>
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

unsigned int** convert_train_dataset(uchar **dataset, int num_images, int num_rows, int num_cols) {
    unsigned int** _cdata = new unsigned int*[num_images];
    
    for(int k = 0; k < num_images; k++) {
        _cdata[k] = new unsigned int[num_rows * num_cols];
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                _cdata[k][28*i + j] = (unsigned int)(dataset[k][28*i+j]);
            }
        }
    }
    
    return _cdata;
}

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

void zeros(double *arr, int asize) {
    #pragma acc parallel loop pcopyout(arr[:asize])
    for(int i = 0; i < asize; i++) {
        arr[i] = 0.0f;
    }
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

float cross_entropy_total(double *y, double *ty, int num_samples, int num_outputs) {
    float error = 0.0f;
    
    for(int i = 0; i < num_samples; i++) {
        float D = 0.0f;
        for(int m = 0; m < num_outputs; m++) {
            D += -ty[i*num_outputs + m]*log(y[i*num_outputs + m]);
        }
        error+=D;
    }
    
    return error;
}


double accuracy(double *predictions, double *labels, int batch_size, int num_outputs) {
    double prc = 0.0f;
    
    for(int i = 0; i < batch_size; i++) {
        double mx = 0.0f;
        int imx = 0;
        double lmx = 0.0f;
        int limx = 0;
        
        for(int m = 0; m < num_outputs; m++) {
            if(predictions[i*num_outputs + m] > mx) {
                mx = predictions[i*num_outputs + m];
                imx = m;
            }
            if(labels[i*num_outputs + m] > lmx) {
                lmx = labels[i*num_outputs + m];
                limx = m;
            }
            
        }
        
        if(imx == limx) {
            prc+=1.0f;
        }
    }
    
    return prc/batch_size;
}


void eval_model(double *input, double *labels, double *train_weights_conv1, double *train_biases_conv1, double *train_weights_conv2, double *train_biases_conv2, double *train_weights_hidden3, double *train_biases_hidden3, double *train_weights_output4, double *train_biases_output4, int num_samples, int num_rows, int num_cols, int num_channels_conv1, int depth_conv1, int filter_size_conv1, int stride_conv1, int num_channels_conv2, int depth_conv2, int filter_size_conv2, int stride_conv2, int num_hidden_nodes3, int num_outputs , string ndata, int start) {
    int num_inputs = num_rows * num_cols;
    int batch_size = 500, batch_start = start, eval_batch_start = start;
    int num_channels = num_channels_conv1;

    double *eval_batch = new double[batch_size * num_channels * num_rows * num_cols];
    double *eval_labels_batch = new double[batch_size * num_outputs];

    ////////////////////////////////////////////////////////////////////
    int pad_height_conv1, pad_width_conv1, pad_top_conv1, pad_bottom_conv1, pad_left_conv1, pad_right_conv1;
    int out_width_conv1, out_height_conv1, out_size_conv1, out_vol_size_conv1;
    int wflat_conv1 = filter_size_conv1 * filter_size_conv1 * num_channels_conv1;
    
    prepare_conv(batch_size, 28, 28, num_channels_conv1, filter_size_conv1, stride_conv1, 1, pad_height_conv1, pad_width_conv1, pad_top_conv1, pad_bottom_conv1, pad_left_conv1, pad_right_conv1, out_height_conv1, out_width_conv1);
    
    out_size_conv1 = out_width_conv1 * out_height_conv1;
    out_vol_size_conv1 = out_size_conv1 * depth_conv1;
    
    int p_width_conv1, p_height_conv1;
    p_width_conv1 = num_cols + pad_width_conv1;
    p_height_conv1 = num_rows + pad_height_conv1;
    
    double *eval_x_in_conv1 = new double[batch_size * p_width_conv1 * p_height_conv1 * num_channels_conv1];
    zeros(eval_x_in_conv1, batch_size * p_width_conv1 * p_height_conv1 * num_channels_conv1);
    
    double *eval_in_conv1 = new double[batch_size * out_vol_size_conv1];
    double *eval_out_conv1 = new double[batch_size * out_vol_size_conv1];

    ///////////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////
    int pad_height_conv2, pad_width_conv2, pad_top_conv2, pad_bottom_conv2, pad_left_conv2, pad_right_conv2;
    int out_width_conv2, out_height_conv2, out_size_conv2, out_vol_size_conv2;
    int wflat_conv2 = filter_size_conv2 * filter_size_conv2 * num_channels_conv2;
    
    prepare_conv(batch_size, out_height_conv1, out_width_conv1, num_channels_conv2, filter_size_conv2, stride_conv2, 1, pad_height_conv2, pad_width_conv2, pad_top_conv2, pad_bottom_conv2, pad_left_conv2, pad_right_conv2, out_height_conv2, out_width_conv2);
    
    out_size_conv2 = out_width_conv2 * out_height_conv2;
    out_vol_size_conv2 = out_size_conv2 * depth_conv2;
    
    int p_width_conv2, p_height_conv2;
    p_width_conv2 = out_width_conv1 + pad_width_conv2;
    p_height_conv2 = out_height_conv2 + pad_height_conv2;
        
    double *eval_x_in_conv2 = new double[batch_size * num_channels_conv2 * p_width_conv2 * p_height_conv2];
    zeros(eval_x_in_conv2, batch_size * num_channels_conv2 * p_width_conv2 * p_height_conv2);
    
    double *eval_in_conv2 = new double[batch_size * out_vol_size_conv2];
    double *eval_out_conv2 = new double[batch_size * out_vol_size_conv2];
    
    ///////////////////////////////////////////////////////////////////////////////////
    
    //////////////////////////////////////////////////////////////////////////////////
    double *eval_wx_hidden3 = new double[batch_size * num_hidden_nodes3];
    double *eval_out_hidden3 = new double[batch_size * num_hidden_nodes3];

    ////////////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////////

    double *eval_wx_output4 = new double[batch_size * num_outputs];
    double *eval_out_output4 = new double[batch_size * num_outputs];
    double *eval_out_output4_sum = new double[batch_size];
    double *total_out_output4 = new double[num_samples * num_outputs];
    //////////////////////////////////////////////////////////////////////////////////
    
    double error_total = 0.0f;
    double error = 0.0f;
    
    int iterations = num_samples/batch_size;
    cout << ndata << endl;
    cout << "iterations " << iterations << endl;
    int total_acc = 0;
    
    #pragma acc data copy(eval_batch[:batch_size*num_inputs], eval_labels_batch[:batch_size*num_outputs]) copyin(input[:num_samples*num_inputs], labels[:num_samples*num_outputs]) \
    copyin(eval_x_in_conv1[:(batch_size*p_width_conv1*p_height_conv1*num_channels_conv1)]) \
    copyin(train_weights_conv1[:wflat_conv1*depth_conv1], train_biases_conv1[:depth_conv1]) \
    create(eval_in_conv1[:(batch_size* out_vol_size_conv1)], eval_out_conv1[:(batch_size*out_vol_size_conv1)]) \
    \
    copyin(eval_x_in_conv2[:(batch_size*p_width_conv2*p_height_conv2*num_channels_conv2)]) \
    copyin(train_weights_conv2[:wflat_conv2*depth_conv2], train_biases_conv2[:depth_conv2]) \
    create(eval_in_conv2[:(batch_size* out_vol_size_conv2)], eval_out_conv2[:(batch_size*out_vol_size_conv2)]) \
    \
    copyin(train_weights_hidden3[:(out_vol_size_conv2*num_hidden_nodes3)], train_biases_hidden3[:num_hidden_nodes3]) \
    create(eval_wx_hidden3[:batch_size*num_hidden_nodes3], eval_out_hidden3[:batch_size*num_hidden_nodes3]) \
    \
    copyin(train_weights_output4[:num_hidden_nodes3*num_outputs], train_biases_output4[:num_outputs]) \
    copyout(eval_wx_output4[:batch_size*num_outputs], eval_out_output4[:batch_size*num_outputs], eval_out_output4_sum[:batch_size])
    {
    for(int b = 0; b < iterations; b++) {
        batch_start = b*batch_size;
        make_batch(input, eval_batch, batch_size, num_inputs, batch_start);
        make_batch(labels, eval_labels_batch, batch_size, num_outputs, batch_start);
        
        copypad(eval_batch, eval_x_in_conv1, num_rows, num_cols, num_channels_conv1, batch_size, pad_left_conv1, pad_right_conv1, pad_top_conv1, pad_bottom_conv1, p_height_conv1, p_width_conv1);
        
        tparallel_conv5(eval_x_in_conv1, train_weights_conv1, eval_in_conv1, batch_size, num_channels_conv1, p_height_conv1 , p_width_conv1, depth_conv1, out_height_conv1, out_width_conv1, filter_size_conv1, stride_conv1, false);
        tparallel_matrix_add_depth(eval_in_conv1, train_biases_conv1, batch_size, out_size_conv1, depth_conv1);
        tparallel_relu_dummy(eval_out_conv1, eval_in_conv1, batch_size, out_vol_size_conv1);

        copypad(eval_out_conv1, eval_x_in_conv2, out_height_conv1, out_width_conv1, num_channels_conv2, batch_size, pad_left_conv2, pad_right_conv2, pad_top_conv2, pad_bottom_conv2, p_height_conv2, p_width_conv2);

        tparallel_conv5(eval_x_in_conv2, train_weights_conv2, eval_in_conv2, batch_size, num_channels_conv2, p_height_conv2, p_width_conv2, depth_conv2, out_height_conv2, out_width_conv2, filter_size_conv2, stride_conv2, false);
        tparallel_matrix_add_depth(eval_in_conv2, train_biases_conv2, batch_size, out_size_conv2, depth_conv2);
        tparallel_relu_dummy(eval_out_conv2, eval_in_conv2, batch_size, out_vol_size_conv2);
        
        tparallel_matrix_multiply(eval_out_conv2, train_weights_hidden3, eval_wx_hidden3, batch_size, out_vol_size_conv2, num_hidden_nodes3);
        tparallel_matrix_add_row(eval_wx_hidden3, train_biases_hidden3, batch_size, num_hidden_nodes3);
        tparallel_relu_dummy(eval_out_hidden3, eval_wx_hidden3, batch_size, num_hidden_nodes3);
        
        tparallel_matrix_multiply(eval_out_hidden3, train_weights_output4, eval_wx_output4, batch_size, num_hidden_nodes3, num_outputs);
        tparallel_matrix_add_row(eval_wx_output4, train_biases_output4, batch_size, num_outputs);
        tparallel_softmax(eval_out_output4, eval_wx_output4, eval_out_output4_sum, batch_size, num_outputs);
        
        macc_update_self(eval_out_output4, batch_size*num_outputs);
        for(int k = 0; k < batch_size; k++) {
            for(int m = 0; m < num_outputs; m++) {
                total_out_output4[(k + batch_start)*num_outputs + m] = eval_out_output4[k*num_outputs + m];
            }
        }
        
        macc_update_self(eval_labels_batch, batch_size*num_outputs);
        error = cross_entropy_total(eval_out_output4, eval_labels_batch, batch_size, num_outputs);
        error_total += error;
        double accr = accuracy(eval_out_output4, eval_labels_batch, batch_size, num_outputs);
        int accrn = accr*batch_size;
        total_acc += accrn;
        cout << "Iteration: " << b << " | Accurary: " << accrn << "/" << batch_size << " = " << accr*100 << "%" << " | Error: " << b << " -> " << error << " /// " << error/batch_size << endl;
    }
    }
    
    cout << "Total error for " << ndata << " dataset: " << error_total << " -> " << error_total/num_samples << endl;
    cout << "Total accuracy: " << total_acc << " / " << iterations*batch_size << " = " << (total_acc)/(iterations*batch_size) * 100 << "%" << endl;
    for(int i = 0; i < min(10, batch_size); i++) {
        cout << "Eval conv model for " << ndata << " data[" << (i + eval_batch_start) << "]" << endl;
        cout << "Expected:  [ ";
        for(int j = 0; j < num_outputs; j++) {
            printf("%.3f ", labels[(i + eval_batch_start)*num_outputs + j]);
        }
        cout << "]" << endl;
        cout << "Predicted: [ ";
        for(int j = 0; j < num_outputs; j++) {
            printf("%.3f ", total_out_output4[(i + eval_batch_start)*num_outputs + j]);
        }
        cout << "]" << endl << endl;
    }
    
    free(eval_batch);
    free(eval_labels_batch);
    free(eval_x_in_conv1);
    free(eval_in_conv1);
    free(eval_out_conv1);
    free(eval_x_in_conv2); 
    free(eval_in_conv2);
    free(eval_out_conv2);
    free(eval_wx_hidden3);
    free(eval_out_hidden3);
    free(eval_wx_output4);
    free(eval_out_output4);
    free(eval_out_output4_sum);
    
}
/*
int **readtrainfile(string path, double ***&data) {
    ifstream csvfile(path);
    string line;
    
    int layers;

    getline(csvfile, line);
    layers = stoi(line);
    
    cout << "Number of layers " << layers << endl;
    int **archit;
    archit = new int*[layers+1];
    
    
    cout << "printing lines from " << path << endl;
    int i = 0;
    
    while(i <= layers && getline(csvfile, line)) {
//         cout << line << endl;
        archit[i] = new int[9];
        istringstream s(line);
        string field;
        int j = 0;
        while(getline(s, field, ';')) {
//             cout << field << endl;
            archit[i][j] = stoi(field);
            j++;
        }
        i++;
//         cout << line << endl;
    }
    
    
    data = new double**[layers];
    for(int l = 0; l < layers; l++) {
        data[l] = new double*[2];
    }
    
    while(getline(csvfile, line)) {
        cout << line << endl;
        istringstream s(line);
        string field;
        int j = 0;
        int layerparam[3];
        
        while(getline(s, field, ';')) {
            cout << field << endl;
            layerparam[j] = stoi(field);
            j++;
        }
        
        int layern = layerparam[0], paramtype = layerparam[1], numparams = layerparam[2];
        cout << layern << " | " << paramtype << " | " << numparams << endl;
        data[layern - 1][paramtype - 1] = new double[numparams];
        
        for(int p = 0; p < numparams; p++) {
            getline(csvfile, line);
            double param = atof(line.c_str());
            data[layern - 1][paramtype - 1][p] = param;
        }
    }
    
    return archit;
}*/

int **readtrainfile(string path, double ***&data) {
    ifstream csvfile(path);
    std::string line;
    
    int layers;

    getline(csvfile, line);
    layers = stoi(line);
    
    int **archit;
    archit = new int*[layers+1];
    
    
    int i = 0;
    
    while(i <= layers && getline(csvfile, line)) {
        archit[i] = new int[9];
        std::istringstream s (line);
        string field;
        int j = 0;
        while(getline(s, field, ';')) {
            archit[i][j] = stoi(field);
            j++;
        }
        i++;
    }
    
    
    data = new double**[layers];
    for(int l = 0; l < layers; l++) {
        data[l] = new double*[2];
    }
    
    while(getline(csvfile, line)) {
        istringstream s(line);
        string field;
        int j = 0;
        int layerparam[3];
        
        while(getline(s, field, ';')) {
            layerparam[j] = stoi(field);
            j++;
        }
        
        int layern = layerparam[0], paramtype = layerparam[1], numparams = layerparam[2];
        data[layern - 1][paramtype - 1] = new double[numparams];
        
        for(int p = 0; p < numparams; p++) {
            getline(csvfile, line);
            double param = atof(line.c_str());
            data[layern - 1][paramtype - 1][p] = param;
        }
    }
    
    return archit;
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
    int test_num_samples = test_num_images;
    double dml = 1;
    
    double *train_images = data2mono(train_img_data, num_images, img_size, dml);
    double *train_labels = labels1hot(train_labels_data, num_images, num_outputs);
    double *test_images = data2mono(test_img_data, test_num_images, test_img_size, dml);
    double *test_labels = labels1hot(test_labels_data, test_num_images, num_outputs);
   
    
    int batch_size = 32;
    int num_channels = 1;
    
    double *train_batch = new double[batch_size * num_channels * num_rows * num_cols];
    double *train_labels_batch = new double[batch_size * num_outputs];
    
    string arg1(argv[1]);
    
    cout << arg1 << endl;
    
    double ***data;
    int **archit = readtrainfile(arg1,data);
    
    /*
    cout << "Data" << endl;
    cout << data[0][0][0] << endl;
    cout << "AAAA " << archit[1][1] << endl;*/
    
//     eval_conv_model2_n(train_images, train_labels, train_weights_conv1, train_biases_conv1, train_weights_conv2, train_biases_conv2, train_weights_hidden3, train_biases_hidden3, train_weights_output4, train_biases_output4, num_samples, num_rows, num_cols, num_channels_conv1, depth_conv1, num_channels_conv2, depth_conv2, num_hidden_nodes3, num_outputs , "train", 0);
    int num_channels_conv1 = archit[1][4], filter_size_conv1 = archit[1][5], depth_conv1 = archit[1][3], stride_conv1 = archit[1][7];
//     double *eval_weights_czzonv1 = new double[depth_conv1 * num_channels_conv1 * filter_size_conv1 * stride_conv1];
//     double *eval_biases_conv1 = new double[depth_conv1];
    
    int num_channels_conv2 = archit[2][4], filter_size_conv2 = archit[2][5], depth_conv2 = archit[2][3], stride_conv2 = archit[2][7];
//     double *eval_weights_conv2 = new double[depth_conv2 * num_channels_conv2 * filter_size_conv2 * stride_conv2];
//     double *eval_biases_conv2 = new double[depth_conv2];
    int out_vol_size_conv2 = archit[3][3];
    
    int num_hidden_nodes3 = archit[3][4];
//     double *eval_weights_hidden3 = new double[out_vol_size_conv2 * num_hidden_nodes3];
//     double *eval_biases_hidden3 = new double[num_hidden_nodes3];
//     
//     double *eval_weights_output4 = new double[num_hidden_nodes3 * num_outputs];
//     double *eval_biases_output4 = new double[num_outputs];
    
    eval_model(train_images, train_labels, data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1], data[3][0], data[3][1], num_samples, num_rows, num_cols, num_channels_conv1, depth_conv1, filter_size_conv1, stride_conv1, num_channels_conv2, depth_conv2, filter_size_conv2, stride_conv2, num_hidden_nodes3, num_outputs , "train", 0);
    
    eval_model(test_images, test_labels, data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1], data[3][0], data[3][1], test_num_samples, num_rows, num_cols, num_channels_conv1, depth_conv1, filter_size_conv1, stride_conv1, num_channels_conv2, depth_conv2, filter_size_conv2, stride_conv2, num_hidden_nodes3, num_outputs , "test", 0);
    return 0;
}
