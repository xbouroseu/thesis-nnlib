#include <cstdio>
#include <cmath>
#include <curand.h>
#include <iostream>
#include <vector>
#include "ops.h"
#include "tensor.h"

using namespace std;

/////// NEW
/*
 * Name: acc_relu
 * Description: Performs a 'Linear rectified Unit' - ReLU function on the input and stores the result to the output
 *
 * Parameters: {
 *     output: N x M size matrix containing the output of the ReLU operation
*      output_drv: N x M size matrix containing the derivative of the output of the ReLU operation
 *     rl_input: N x M size matrix containing the input to the ReLU operation
 * }
 * 
 * Example:
 *          |-4 5 2|            |0 5 2|
 *      A = |0 1 -1|, ReLU(A) = |0 1 0|
 *          |6 -8 3|            |6 0 3|
*/
template <class T>
void acc_relu(const Tensor4D<T> &input, Tensor4D<T> &output, Tensor4D<T> &output_drv) {
    int size = input.get_shape().size();
    const T *in_data = input.get_data(), *out_data = output.get_data(), *out_drv_data = output_drv.get_data();
    #pragma acc data present(in_data[:size]) present(out_data[:size]) present(out_drv_data[:size])
    {
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {
        T val = in_data[j];

        if(val > 0) {
            output.set(j, val);
            output_drv.set(j, (T)1.0f);
        }
        else {
            output.set(j, (T)0.0f);
            output_drv.set(j, (T)0.0f);
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
void acc_sigmoid(const Tensor4D<T> &input, Tensor4D<T> &output, Tensor4D<T> &output_drv) {
    int size = input.get_shape().size();
    const T *in_data = input.get_data(), *out_data = output.get_data(), *out_drv_data = output_drv.get_data();
    #pragma acc data present(in_data[:size], out_data[:size], out_drv_data[:size])
    {
    
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {
        T val = 1/(1 + exp(-1 * in_data[j]));

        output.set(j, val);
        output_drv.set_inline(j, val*(1-val));
    }
    }
}

template <class T>
void acc_softmax(const Tensor4D<T> &input, Tensor4D<T> &output) {
    int size = input.get_shape().size();
    const T *in_data = input.get_data(), *out_data = output.get_data();
    T output_sum = 0.0f;
    
    #pragma acc data present(in_data[:size], out_data[:size]) create(output_sum)
    {
    
    #pragma acc parallel loop reduction(+:output_sum)
    for(int j = 0; j < size; j++) {
        output_sum += exp(in_data[j]);
//             #pragma acc atomic update
//             output_sum[i] += exp(input[i*num_outputs + j]);
    }
        
    #pragma acc parallel loop
    for(int j = 0; j < size; j++) {            
        output.set_atomic(j, exp(in_data[j]) / output_sum);
    }
    }
}



//TODO find where N,M,K are in Tensor4D, readjust Tensor4D? or shape2D?, shape3D?
/*
 * Name: acc_matrix_multiply4D
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
void acc_matrix_multiply(const Tensor4D<T> &A, const Tensor4D<T> &B, Tensor4D<T> &C) {
    Shape4D a_shape = A.get_shape(), b_shape = B.get_shape(), c_shape = C.get_shape();
    Shape4D a_shape_flat = a_shape.flat(1);
    
    if(a_shape_flat[1] != b_shape[0]) {
        throw(std::invalid_argument("Error: cols(A)!=rows(B). "));
    }
    
    if(a_shape_flat[0] != c_shape[0]) {
        throw(std::invalid_argument("Error: rows(A)!=rows(C)"));
    }
    
    if(b_shape[1] != c_shape[1]) {
        throw(std::invalid_argument("Error: cols(B) != cols(C)"));
    }
    
    int N = a_shape[0], K = a_shape[1], M = b_shape[1];
    const T *a_data = A.get_data(), *b_data = B.get_data(), *c_data = C.get_data();
    
    #pragma acc data present(a_data[:(N*K)], b_data[0:K*M], c_data[0:N*M])
    {

    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            T csumd = 0.0f;
            #pragma acc loop seq reduction(+:csumd)
            for(int t = 0; t < K; t++) {
                csumd += a_data[i*K + t]*b_data[t*M + j];
            }
            C.set(i*M + j, csumd);
        }
    }

    }
}

//TODO stride 2D?
template <class T>
void acc_convolution2D(const Tensor4D<T> &input, const Tensor4D<T> &filters, Tensor4D<T> &output, vector<int> stride) { 
    
    int batch = input.get_shape()[0];
    int in_channels = input.get_shape()[1];
    int in_rows = input.get_shape()[2];
    int in_cols = input.get_shape()[3];
    
    int filter_height = filters.get_shape()[2], filter_width = filters.get_shape()[3];
    int stride_r = stride[0], stride_c = stride[1];
    int out_channels = filters.get_shape()[1];
    
    int out_cols = output.get_shape()[3];
    int out_rows = output.get_shape()[2];
    
    if(in_channels != filters.get_shape()[0]) {
        throw(std::invalid_argument("Error: input channels != filter.get_shape()[0]"));
    }
    
    if(out_channels != output.get_shape()[1]) {
        throw(std::invalid_argument("Error: output channels != output.get_shape()[1]"));
    }
    
    if(batch != output.get_shape()[0]) {
        throw(std::invalid_argument("Error: batch != output.get_shape()[0]"));
    }
    
    const T *in_data = input.get_data(), *out_data = output.get_data(), *filter_data = filters.get_data();
    #pragma acc data present(in_data[:(batch*in_cols*in_rows*in_channels)]) \
    present(filter_data[:in_channels*out_channels*filter_height*filter_width]) \
    present(out_data[:(batch* out_channels * out_cols * out_rows)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int i = 0 ; i < batch; i++) {
        for(int och = 0; och < out_channels; och++) {
            for(int oh = 0; oh < out_rows; oh++) {
                for(int ow = 0; ow < out_cols; ow++) {
                    T bdhwsum = 0.0f;
                    
                    #pragma acc loop seq collapse(3) reduction(+:bdhwsum)
                    for(int ich = 0; ich < in_channels; ich++) {
    //                         double csum = 0.0f;
                        for(int fi = 0; fi < filter_height; fi++) {
                            for(int fj = 0; fj < filter_width; fj++) {
                                bdhwsum += input[ (i*in_channels + ich)*in_rows*in_cols + (oh*stride_r + fi)*in_cols + ow*stride_c + fj ] * filters[ (och*in_channels + ich)*filter_height*filter_width + fi*filter_width + fj ];
                            }
                        }
    //                         sum += csum;
                    }
                    
                    output.set((i*out_channels + och)*out_cols*out_rows + oh*out_cols + ow, bdhwsum);
                }
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
void acc_pad2D(const Tensor4D<T> &input, Tensor4D<T> &output, int padding_left, int padding_right, int padding_top, int padding_bottom) {
    int B = input.get_shape()[0], C = input.get_shape()[1], N = input.get_shape()[2], M = input.get_shape()[3];
    int padded_N = N + padding_top + padding_bottom;
    int padded_M = M + padding_left + padding_right;
    
    const T *in_data = input.get_data(), *out_data = output.get_data();
    #pragma acc data present(in_data[:(B*C*M*N)], out_data[:(B*C*padded_N*padded_M)])
    {
        
    #pragma acc parallel loop collapse(4)
    for(int b = 0; b < B; b++) {
        for(int c = 0; c < C; c++) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    output.set(b*padded_N*padded_M * C + c*padded_N*padded_M + (i+padding_top)*padded_M + j + padding_left, input[b*C*M*N + c*M*N + i*M + j]);
                }
            }
        }
    }
    
    }
}

template<class T>
void acc_make_batch(Tensor4D<T> &inputs, Tensor4D<T> &batch, int batch_start) {
    cout << "acc_make_batch"<<endl;
    int batch_size = batch.get_shape()[0], num_inputs = inputs.get_shape()[0], input_size = inputs.get_shape()[1]*inputs.get_shape()[2]*inputs.get_shape()[3], batch_input_size = batch.get_shape()[1]*batch.get_shape()[2]*batch.get_shape()[3];
    
    if((batch_size > num_inputs) || input_size!=batch_input_size) {
        throw(std::invalid_argument("Error batch,inputs not compatible"));
    }
    
    T *batch_data = batch.get_data(), *inputs_data = inputs.get_data();
    
    cout << "Entering data" << endl;
    #pragma acc data copyin(inputs_data[(batch_start*input_size):(batch_size*input_size)]) present(batch_data[:(batch_size*input_size)])
    {
    cout << "Entering loop" << endl;
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < input_size; k++) {
            batch_data[i*input_size + k] = inputs.get((i+batch_start)*input_size + k);
        }
    }
        
    }
    cout << "/acc_make_batch"<<endl;

}

void make_batch(double *inputs, double *batch, int batch_size, int input_size, int batch_start) {
    #pragma acc data copyin(inputs[(batch_start*input_size):(batch_size*input_size)]) copy(batch[:(batch_size*input_size)])
    {
     
    #pragma acc parallel loop collapse(2)
    for(int i = 0; i < batch_size; i++) {
        for(int k = 0; k < input_size; k++) {
            batch[i*input_size + k] = inputs[(i+batch_start)*input_size + k];
        }
    }
        
    }
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


/*

//TODO acc?
template<class T>
void rngnormal(T *inp, int n, T multiplier) {
    int i, istat;
    curandGenerator_t g;
    int pn = n;
    
    if(pn%2 == 1) pn = pn+1;
    
    float *nr = new float[pn];
    T *nrT = new T[pn];
        
    for (i = 0; i < pn; i++) {
        nr[i] = 0.0f;
    }
    
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
//     cout << "N: " << n << " pn: " << pn << " m: " << multiplier << " | " << inp << endl;
    
    // Now Normal
    istat = curandGenerateNormal(g, nr, pn, 0.0f, 1.0f);
    
    if (istat != CURAND_STATUS_SUCCESS) printf("Error %d\n",istat);
    
    for(int i = 0; i <pn; i++) {
        nrT[i] = nr[i];
    }
    
    T mx = 0, mn = nrT[0];
    
    for(int i = 0; i < pn; i++) {
        if(nrT[i] > mx) mx = nrT[i];
        if(nrT[i] < mn) mn = nrT[i];
    }
    
    T range = mx - mn;
    T ml = 2.0f * multiplier / range;
    
//     cout << "Max: " << mx << ", Min: " << mn << ", Range: " << range << ", Ml: " << ml << endl;

    for (i = 0; i < pn; i++) {
        nrT[i] = nrT[i] * ml;
    }
    
    T mx2 = 0, mn2 = 0;
    
    for(int i = 0; i < pn; i++) {
        //cout << "i: "<< a[i] << endl;
        if(nrT[i] > mx2) mx2 = nrT[i];
        if(nrT[i] < mn2) mn2 = nrT[i];
    }
//     cout << "New Max: " << mx2 << ", New Min: " << mn2 << endl;
    
    for(int i = 0; i < n; i++) {
        inp[i] = nrT[i];
    }
    istat = curandDestroyGenerator(g);

}
*/
void transpose(double *A, double *A_tr, int NN, int MM) {
    #pragma acc data copyin(A[:NN*MM]) copyout(A_tr[:NN*MM])
    {
    #pragma acc parallel loop collapse(2)
    for(int n = 0; n < NN; n++) {
        for(int m = 0; m < MM; m++) {
            A_tr[m*NN + n] = A[n*MM + m];
        }
    }
    }
}

template<class T>
void acc_rng(Tensor4D<T> &A, T mtlp) {
    T* a_data = A.get_data();
    int i, istat, n = A.get_shape().size(), pn = n;
    float stddev = 1.0f, mean = 0.0f;
    curandGenerator_t g;
    
    istat = curandCreateGeneratorHost(&g, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS) {
        printf("Error creating generator host %d\n",istat);
    }
    
    if(pn%2 == 1) {
        pn = pn+1;
    }
    
    float *array_rng_f = new float[pn];
    
    #pragma acc data create(array_rng_f[:pn]), copyout(a_data[:n])
    {
    #pragma acc parallel loop
    for (i = 0; i < pn; i++) {
        array_rng_f[i] = 0.0f;
    }
    
    /* Create Normal distribution */
    istat = curandGenerateNormal(g, array_rng_f, pn, mean, stddev);
    if (istat != CURAND_STATUS_SUCCESS) {
        printf("Error creating normal distribution %d\n", istat);
    }
    
    //Copy float to _data
    #pragma acc parallel loop
    for(int i = 0; i < n; i++) {
        A.set(i, (T)array_rng_f[i]);
    }
    
    //find max, min
    T mx = 0, mn = a_data[0];
    
    #pragma acc parallel loop reduction(max:mx) reduction(min:mn)
    for(int i = 0; i < n; i++) {
        if(A[i] > mx) {
            mx = A[i];
        }
        
        if(a_data[i] < mn) {
            mn = A[i];
        }
    }
    
    //normalize
    T range = mx - mn;
    T ml = 2.0f * mtlp / range;
    
    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        A.mltp(i, ml);
    }
    
    }
    
    istat = curandDestroyGenerator(g);

    delete[] array_rng_f;
}

template<class T>
void acc_val(Tensor4D<T> &A, T val) {
    int asize = A.get_shape().size();
    
    const T* a_data = A.get_data();
    #pragma acc parallel loop copyout(a_data[:asize])
    for(int i = 0; i < asize; i++) {
        A.set(i, val);
    }
}

template<class T>
void acc_zeros(Tensor4D<T> &A) {
    acc_val(A, (T)0.0f);
}

template<class T>
void acc_mltp(Tensor4D<T> &A, T mltp) {
    int asize = A.get_shape().size();
    
    const T *a_data = A.get_data();
    #pragma acc parallel loop copy(a_data[:asize])
    for(int i = 0; i < asize; i++) {
        A.mltp(i, mltp);
    }
}



// TODO: Not parallelizable
/*
template<class T>
Tensor4D<T> AddVecBroadCast(const Tensor4D<T> &a, const Tensor4D<T> &b) {
    int sizea = a.size(), sizeb = b.size();
    
    int mxa, mxb, mxc, mxd, ai, aj, ak, al, bi, bj, bk, bl;
    
    if( !(a.a==b.a || ( (a.a==1) ^ (b.a==1) ) ) ) {
        throw(std::invalid_argument(("Error: Tensors dim(a) not addition compatible.");
    }
    
    if( !(a.b==b.b || ( (a.b==1) ^ (b.b==1) ) ) ) {
        throw(std::invalid_argument(("Error: Tensors dim(b) not addition compatible.");
    }

    if( !(a.c==b.c || ( (a.c==1) ^ (b.c==1) ) ) ) {
        throw(std::invalid_argument(("Error: Tensors dim(c) not addition compatible.");
    }

    if( !(a.d==b.d || ( (a.d==1) ^ (b.d==1) ) ) ) {
        throw(std::invalid_argument(("Error: Tensors dim(d) not addition compatible.");
    }
    
    mxa = max(a.a, b.a);
    mxb = max(a.b, b.b);
    mxc = max(a.c, b.c);
    mxd = max(a.d, b.d);
    
    if(a.a == 1) {
        ai = 0;
    }
    
    if(a.b == 1) {
        aj = 0;
    }
    
    if(a.c == 1) {
        ak = 0;
    }
    
    if(a.d == 1) {
        al = 0;
    }
    
    if(b.a == 1) {
        bi = 0;
    }
    
    if(b.b == 1) {
        bj = 0;
    }
    
    if(b.c == 1) {
        bk = 0;
    }
    
    if(b.d == 1) {
        bl = 0;
    }    
    
    for(int i = 0, ; i < mxa; i++) {
        if(a.a != 1) {
            ai = i;
        }
        
        if(b.a != 1) {
            bi = i;
        }
        
        for(int j = 0; j<mxb; j++) {
            if(a.b != 1) {
                aj = j;
            }
            
            if(b.b != 1) {
                bj = j;
            }
            
            for(int k=0; k<mxc; k++) {
                if(a.c != 1) {
                    ak = k;
                }
                
                if(b.c != 1) {
                    bk = k;
                }
            
                for(int l=0; l<mxd; l++) {
                    if(a.d != 1) {
                        al = l;
                    }
                    
                    if(b.d != 1) {
                        bl = l;
                    }
                    
                    a_data[ai*mxb*mxc*mxd + aj*mxc*mxd + ak*mxd + al] += b_data[bi*mxb*mxc*mxd + bj*mxc*mxd + bk*mxd + bl];
                }
            }
        }
    }
}
*/

template<class T>
void AddVecDim0(Tensor4D<T> &A, const Tensor4D<T> &B) {
    if(!(B.get_shape()[1]==1 && B.get_shape()[2]==1 && B.get_shape()[3]==1)) {
        throw(std::invalid_argument("Error: Tensor B is not Vector(1xbx1x1)."));
    }

    if(B.get_shape()[0] != A.get_shape()[0]) {
        throw(std::invalid_argument("Error: Tensor A.get_shape()[0] != B.get_shape()[0]."));
    }
    
    int sizeA = A.get_shape().size(), sizeB = B.get_shape().size(), a=A.get_shape()[0], b=A.get_shape()[1], c=A.get_shape()[2], d=A.get_shape()[3];
    int bcd = b*c*d, cd = c*d;
    const T *a_data = A.get_data(), *b_data = B.get_data();
    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop collapse(4)
    for(int i=0; i<a; i++) {
        for(int j=0; j<b; j++) {
            for(int k=0; k<c; k++) {
                for(int l=0; l<d ; l++) {
                    A.add(i*bcd + j*cd + k*d + l,  B[i]);
                }
            }
        }
    }
    }
}

template<class T>
void AddVecDim1(Tensor4D<T> &A, const Tensor4D<T> &B) {
    if(!(B.get_shape()[0]==1 && B.get_shape()[2]==1 && B.get_shape()[3]==1)) {
        throw(std::invalid_argument("Error: Tensor B is not Vector(1xbx1x1)."));
    }

    if(B.get_shape()[1] != A.get_shape()[1]) {
        throw(std::invalid_argument("Error: Tensor A.get_shape()[1] != B.get_shape()[1]."));
    }
    
    int sizeA = A.get_shape().size(), sizeB = B.get_shape().size(), a=A.get_shape()[0], b=A.get_shape()[1], c=A.get_shape()[2], d=A.get_shape()[3];
    int bcd = b*c*d, cd = c*d;
    const T *a_data = A.get_data(), *b_data = B.get_data();

    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop collapse(4)
    for(int i=0; i<a; i++) {
        for(int j=0; j<b; j++) {
            for(int k=0; k<c; k++) {
                for(int l=0; l<d ; l++) {
                    A.add(i*bcd + j*cd + k*d + l,  B[j]);
                }
            }
        }
    }
    }
}


template<class T>
void AddVecDim2(Tensor4D<T> &A, const Tensor4D<T> &B) {
    if(!(B.get_shape()[0]==1 && B.get_shape()[1]==1 && B.get_shape()[3]==1)) {
        throw(std::invalid_argument("Error: Tensor B is not Vector(1x1xcx1)."));
    }

    if(B.get_shape()[2] != A.get_shape()[2]) {
        throw(std::invalid_argument("Error: Tensor A.get_shape()[2] != B.get_shape()[2]."));
    }
    
    int sizeA = A.get_shape().size(), sizeB = B.get_shape().size(), a=A.get_shape()[0], b=A.get_shape()[1], c=A.get_shape()[2], d=A.get_shape()[3];
    int bcd = b*c*d, cd = c*d;
    const T *a_data = A.get_data(), *b_data = B.get_data();

    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop collapse(4)    
    for(int i=0; i<a; i++) {
        for(int j=0; j<b; j++) {
            for(int k=0; k<c; k++) {
                for(int l=0; l<d ; l++) {
                    A.add(i*bcd + j*cd + k*d + l,  B[k]);
                }
            }
        }
    }
    }
}

template<class T>
void AddVecDim3(Tensor4D<T> &A, const Tensor4D<T> &B) {
    if(!(B.get_shape()[0]==1 && B.get_shape()[1]==1 && B.get_shape()[2]==1)) {
        throw(std::invalid_argument("Error: Tensor B is not Vector(1x1x1xd)."));
    }
    
    if(B.get_shape()[3] != A.get_shape()[3]) {
        throw(std::invalid_argument("Error: Tensor A.get_shape()[3] != B.get_shape()[3]."));
    }
    
    int sizeA = A.get_shape().size(), sizeB = B.get_shape().size(), a=A.get_shape()[0], b=A.get_shape()[1], c=A.get_shape()[2], d=A.get_shape()[3];
    int bcd = b*c*d, cd = c*d;
    const T *a_data = A.get_data(), *b_data = B.get_data();

    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop    
    for(int i=0; i<a; i++) {
        int ibcd = i*bcd;
        #pragma acc loop collapse(3)
        for(int j=0; j<b; j++) {
            for(int k=0; k<c; k++) {
                for(int l=0; l<d ; l++) {
                    A.add(ibcd + j*cd + k*d + l,  B[l]);
                }
            }
        }
    }
    }
}

template<class T>
void AddVecDim(Tensor4D<T> &A, const Tensor4D<T> &B, int dim) {
    vector<int> dim_indices = {0, 0, 0, 0};
    Shape4D a_shape = A.get_shape(), b_shape = B.get_shape();
    
    if(!(a_shape[0]==1 && b_shape[1]==1 && b_shape[2]==1)) {
        throw(std::invalid_argument("Error: Tensor B is not Vector(1x1x1xd)."));
    }
    
    if(b_shape[3] != a_shape[3]) {
        throw(std::invalid_argument("Error: Tensor A.get_shape()[3] != B.get_shape()[3]."));
    }
    
    int sizeA = a_shape.size(), sizeB = b_shape.size(), a=a_shape[0], b=a_shape[1], c=a_shape[2], d=a_shape[3];
    int bcd = b*c*d, cd = c*d;
    const T *a_data = A.get_data(), *b_data = B.get_data();

    #pragma acc data present(a_data[:sizeA], b_data[:sizeB])
    {
    #pragma acc parallel loop    
    for(dim_indices[0]=0; dim_indices[0]<a; dim_indices[0]++) {
        int ibcd = dim_indices[0]*bcd;
        #pragma acc loop collapse(3)
        for(int dim_indices[1]=0; dim_indices[1]<b; dim_indices[1]++) {
            for(int dim_indices[2]=0; dim_indices[2]<c; dim_indices[3]++) {
                for(int dim_indices[3]=0; dim_indices[3]<d ; dim_indices[3]++) {
                    A.add(ibcd + j*cd + k*d + l,  B[dim_indices[dim]]);
                }
            }
        }
    }
    }
}

template<class T>
void AddVecBroadCast(Tensor4D<T> &A, const Tensor4D<T> &B, int dimension) {
    if(dimension == 0) {
        AddVecDim0(A, B);
    }
    else if(dimension == 1) {
        AddVecDim1(A, B);
    }
    else if(dimension == 2) {
        AddVecDim2(A, B);
    }
    else if(dimension == 3) {
        AddVecDim3(A, B);
    }
    else {
        throw(std::invalid_argument("Error: dimension to broadcast not valid."));
    }
}

template void acc_rng(Tensor4D<double> &, double);
template void acc_val(Tensor4D<double> &, double);
template void acc_zeros(Tensor4D<double> &);
template void acc_mltp(Tensor4D<double> &, double);

template void acc_relu(const Tensor4D<double> &, Tensor4D<double> &, Tensor4D<double> &);
template void acc_sigmoid(const Tensor4D<double> &, Tensor4D<double> &, Tensor4D<double> &);
template void acc_softmax(const Tensor4D<double> &, Tensor4D<double> &);

template void acc_matrix_multiply(const Tensor4D<double> &, const Tensor4D<double> &, Tensor4D<double> &);
template void acc_convolution2D(const Tensor4D<double> &, const Tensor4D<double> &, Tensor4D<double> &, std::vector<int>);
template void acc_pad2D(const Tensor4D<double> &, Tensor4D<double> &, int , int , int , int );
template void acc_make_batch(Tensor4D<double> &, Tensor4D<double> &, int );

template void AddVecDim0(Tensor4D<double> &, const Tensor4D<double> &);
template void AddVecDim1(Tensor4D<double> &, const Tensor4D<double> &);
template void AddVecDim2(Tensor4D<double> &, const Tensor4D<double> &);
template void AddVecDim3(Tensor4D<double> &, const Tensor4D<double> &);
template void AddVecBroadCast(Tensor4D<double> &, const Tensor4D<double> &, int);

/////
