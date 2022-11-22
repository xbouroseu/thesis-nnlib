#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#define E 10E-15
using namespace std;

template<class T>
bool is_eq(T a, T b) {
    return abs(a-b)<E;
}

void calc_conv( int in_h, int in_w, vector<int> filter_size, vector<int> stride, std::string padding_type) {
    float out_h{0}, out_w{0}, padding_height{0}, padding_width{0}, padding_top{0}, padding_bottom{0}, padding_left{0}, padding_right{0};
    float out_h_2{0}, out_w_2{0};
/*
    int out_h = floor((prev_shape[2] - filter_size[0] + padding[0] + padding[1])/stride[0]) + 1;
    int out_w = floor((prev_shape[3] - filter_size[1] + padding[2] + padding[3])/stride[1]) + 1;*/
    
    if(padding_type=="same") {
        if(in_h%stride[0] == 0) {
            padding_height = max(filter_size[0] - stride[0], 0);
        }
        else {
            padding_height = max(filter_size[0] - (in_h%stride[0]), 0);
        }

        if(in_w%stride[1] == 0) {
            padding_width = max(filter_size[1] - stride[1], 0);
        }
        else {
            padding_width = max(filter_size[1] - (in_w%stride[1]), 0);
        }
        
        padding_top = padding_height/2;
        padding_bottom = padding_height - padding_top;
        padding_left = padding_width/2;
        padding_right = padding_width - padding_left;
        
        out_h = ceil((1.0f * in_h)/stride[0]);
        out_w = ceil((1.0f * in_w)/stride[1]);
    }
    else if(padding_type=="valid") {
        out_h = ceil( (1.0f * (in_h - filter_size[0] + 1))/stride[0] );
        out_w = ceil( (1.0f * (in_w - filter_size[1] + 1))/stride[1] );
    }
    else {
    }
    
    out_h_2 = (in_h + padding_height - filter_size[0])/stride[0] + 1;
    out_w_2 = (in_w + padding_width - filter_size[1])/stride[1] + 1;
    
    float rev_stride0 = floor(1.0f*(in_h + padding_height - out_h)/(1.0f * (filter_size[0]-1)));
    float rev_stride1 = floor(1.0f*(in_w + padding_width - out_w)/(1.0f * (filter_size[1]-1)));
  
    cout << "Calc conv | padding_type: " << padding_type << " ";
    printf("| In: [%d x %d], filter: [%d x %d], stride: [%d x %d], rev_stride[%f x %f], padding: [%f , %f], Out: [%f x %f], Out2: [%f x %f] \n", in_h, in_w, filter_size[0], filter_size[1], stride[0], stride[1], rev_stride0, rev_stride1, padding_height, padding_width, out_h, out_w, out_h_2, out_w_2);
    
    if(!is_eq(rev_stride0, 1.0f*stride[0])) {
        cout << "Problem"  << endl;
    }
    else {
        cout << rev_stride0 << "==" << stride[0] << endl;
        cout << is_eq(rev_stride0, 1.0f*stride[0]) << endl;
    }
    
    
}

void calc_conv2( vector<int> input_size, vector<int> filter_size, vector<int> stride, std::string padding_type) {
    int in_h = input_size[0], in_w = input_size[1];
    float out_h{0}, out_w{0}, padding_height{0}, padding_width{0}, padding_top{0}, padding_bottom{0}, padding_left{0}, padding_right{0};
/*
    int out_h = floor((prev_shape[2] - filter_size[0] + padding[0] + padding[1])/stride[0]) + 1;
    int out_w = floor((prev_shape[3] - filter_size[1] + padding[2] + padding[3])/stride[1]) + 1;*/
    
    if(padding_type=="same") {
        if(stride[0] != 1 && stride[1] != 1) {
            throw(std::invalid_argument("SAME padding cannot have stride != 1"));
        }
        
        padding_height = filter_size[0] - 1;
        padding_width = filter_size[1] - 1;
        
        padding_top = padding_height/2;
        padding_bottom = padding_height - padding_top;
        padding_left = padding_width/2;
        padding_right = padding_width - padding_left;
        
        out_h = in_h;
        out_w = in_w;
    }
    else if(padding_type=="valid") {
        int nom_h = in_h - filter_size[0];
        int nom_w = in_w - filter_size[1];
        
        if( (nom_h % stride[0])!=0 ) {
            throw(std::invalid_argument("VALID padding: (input_height - filter_height) not divisible by stride."));
        }
        
        if( (nom_w % stride[1])!=0 ) {
            throw(std::invalid_argument("VALID padding: (input_width - filter_width) not divisible by stride."));
        }
        out_h = nom_h/stride[0] + 1;
        out_w = nom_w/stride[1] + 1;
    }
    else {
        throw(std::invalid_argument("Padding type not compatible."));
    }
    
    float rev_stride0 = (in_h + padding_height - out_h)/(filter_size[0]-1);
    float rev_stride1 = (in_w + padding_width - out_w)/(filter_size[1]-1);
  
    cout << "Conv | padding_type: " << padding_type << endl;
    printf("Calc In: [%d x %d], filter: [%d x %d], stride: [%d x %d], padding: [%4.2f , %4.2f], Out: [%4.2f x %4.2f] \n", in_h, in_w, filter_size[0], filter_size[1], stride[0], stride[1], padding_height, padding_width, out_h, out_w);
    printf("Rev: In: [%d x %d], filter: [%4.2f x %4.2f], stride[%4.2f x %4.2f], padding: [%4.2f , %4.2f], Out: [%d, %d] \n", in_h, in_w, out_h, out_w, rev_stride0, rev_stride1, padding_height, padding_width, filter_size[0], filter_size[1]);
}

int main(int argc, char *argv[]) {
//     int is = atoi(argv[1]);
//     int fs = atoi(argv[2]);
//     int strd = atoi(argv[3]);
//     int pi = atoi(argv[4]);
//     
//     vector<string> padding_types = {"valid", "same"};
//     vector<int> input_size{is, is}, filter_size{fs,fs}, stride{strd,strd};
//     calc_conv2(input_size, filter_size, stride, padding_types[pi]);
    

    return 0;
}
