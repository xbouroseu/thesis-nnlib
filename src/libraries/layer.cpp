#include "layer.hpp"
#include "neural.hpp"
#include "ops.hpp"
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cassert>
#include <iomanip>

typedef Tensor4D<double> t4d;

using namespace std;

void helper_CalcConvPaddingOutSize(string padding_type, int in_h, int in_w, vector<int> &filter_size, vector<int> &stride, int &out_h, int &out_w, vector<int> &padding)
{
    if (padding_type == "same")
    {
        if (stride[0] != 1 && stride[1] != 1)
        {
            throw(std::invalid_argument("SAME padding cannot have stride != 1"));
        }

        int padding_height = filter_size[0] - 1;
        int padding_width = filter_size[1] - 1;

        padding[0] = padding_height - padding_height / 2;
        padding[1] = padding_height / 2;
        padding[2] = padding_width - padding_width / 22;
        padding[3] = padding_width / 2;

        out_h = in_h;
        out_w = in_w;
    }
    else if (padding_type == "valid")
    {
        int nom_h = in_h - filter_size[0];
        int nom_w = in_w - filter_size[1];

        if ((nom_h % stride[0]) != 0)
        {
            throw(std::invalid_argument("VALID padding: (input_height - filter_height) not divisible by stride."));
        }

        if ((nom_w % stride[1]) != 0)
        {
            throw(std::invalid_argument("VALID padding: (input_width - filter_width) not divisible by stride."));
        }
        out_h = nom_h / stride[0] + 1;
        out_w = nom_w / stride[1] + 1;
    }
    else
    {
        throw(std::invalid_argument("Padding type not compatible."));
    }
}

// void helper_InnerActivate(const t4d &input, t4d* output, string activation_fn) {
//     if(activation_fn == "relu") {
//         acc_relu(input, output);
//     }
//     else if(activation_fn == "sigmoid") {
//         acc_sigmoid(input, output);
//     }
//     else if(activation_fn == "identity") {
//         // output = input;
//         acc_copy(input, output);
//     }
//     else if(activation_fn == "") {
//     }//TODO output = copy(input)?
//     else if(activation_fn == "softmax") {
//         acc_softmax(input, output);
//     }
//     else {
//         throw(std::invalid_argument("Forward: activation fn not valid"));
//     }//TODO throw error
// }
//
// void helper_InnerCalcErrorOutputPreAct(const t4d &drv_error_output, const t4d &output, t4d *drv_error_output_preact, string activation_fn) {
//     if(activation_fn == "softmax") {
//         acc_softmax_backward(drv_error_output, output,  drv_error_output_preact);
//     }
//     else if (activation_fn ==  "sigmoid") {
//         acc_sigmoid_backward(drv_error_output, output,  drv_error_output_preact);
//
//     }
//     else if (activation_fn ==  "relu") {
//         acc_relu_backward(drv_error_output, output,  drv_error_output_preact);
//     }
//     else if (activation_fn == "identity") {
//         // TODO something here acc_copy?
//         acc_copy(drv_error_output,  drv_error_output_preact);
//     }
//     else {
//         throw(std::invalid_argument("Backprop: activation fn not valid"));
//     }
//
// }

// TODO write correctly the helper forward
template <class... Args>
void helper_forward(void (*fcnptr)(), Args... args)
{

    (*fcnptr)(args);
}

///////////////////////////Layer//////////////////////////////////////
/*
 *
 *
 *
 *
 *
 */

using Neural::Layers::Conv;
using Neural::Layers::Fc;
using Neural::Layers::Layer;
using Neural::Layers::Weighted;

////////// <Layer> ////////////
int Layer::nl = 0;

Layer::Layer(Shape4D &prev_shape_proto, int features, string afn) : prev_shape_proto(prev_shape_proto), features(features) {
    LOG("Layer::Layer");

    cout << "Generating layer number from: " << nl;
    id = ++nl;

    cout << " -> " << id << endl;

    if (afn == "relu") {
        activation_fn = Neural::Activations::Relu;
    }
    else if (afn == "softmax") {
        activation_fn = Neural::Activations::Softmax;
    }
    else if (afn == "sigmoid") {
        activation_fn = Neural::Activations::Sigmoid;
    }
}

string Layer::gph() {
    string ret("[Layer " + to_string(id) + "] [" + layerType + "]");
    return ret;
}

t4d * Layer::activate(t4d &output_preact) {
    LOG("Activation: " + activation_fn.name());

    LOG("output = make_unique<t4d>(" + output_preact.shape().to_string() + ", 1, " + to_string(_acc) + ")");
    t4d * output = new t4d(output_preact.shape());
    
    if (_acc) {
        output->create_acc();
    }

    // helper_InnerActivate(*output_preact, output, activation_fn);
    activation_fn.apply(output_preact, output);

    LOG("[Output]");
    output->print();

    return output;
}

t4d * Layer::backprop_calc_loss(string loss_fn, double &loss_value, t4d & output, Tensor4D<int> &labels_batch) {
    LOG("Layer::loss");
    
    LOG("activation_type: " << this->get_activation_name());
    string activation_type = this->get_activation_name();

    Shape4D output_shape = output.shape();
    assert(labels_batch.shape() == output_shape);

    LOG("unique_ptr<t4d> drv_error_output_preact = make_unique<t4d>(" + output_shape.to_string() + ", 1, " + to_string(_acc) + ")");
    t4d *drv_error_output_preact = new t4d(output_shape);
    
    if (_acc) {
        drv_error_output_preact->create_acc();
    }

    LOG("Layer::loss getting data pointers");
    double *output_data = output.data(), *drv_error_output_preact_data = drv_error_output_preact->data();
    int *labels_data = labels_batch.data();

    int B = output_shape[0], M = output_shape[1];

    LOG("[Output]");
    output.print();

    LOG("[Labels]");
    labels_batch.print();

    int lsize = output_shape.size();

    loss_value = 0.0f;

    if ((loss_fn == "CrossEntropy") && (activation_type == "softmax")) {
        #pragma acc parallel loop reduction(+:loss_value) collapse(2) present(labels_data[:lsize], output_data[:lsize])
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < M; j++) {
                loss_value += (labels_data[i * M + j]) * log(output_data[i * M + j]);
            }
        }

        loss_value *= -1;

        #pragma acc parallel loop collapse(2) present(labels_data[:lsize], output_data[:lsize], drv_error_output_preact_data[:lsize])
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < M; j++) {
                double d_lbl = (double)labels_data[i * M + j];
                drv_error_output_preact_data[i * M + j] = output_data[i * M + j] - d_lbl;
            }
        }
    }

    return drv_error_output_preact;
}

t4d * Layer::backprop_delta_output(t4d &drv_error_output, t4d &output) {
    LOG(gph() + "backward: learning_rate = " + to_string(learning_rate));

    Shape4D output_shape = output.shape();
    // If not set from somewhere else calc here drv_error_output_preact
    LOG("[drv_error_output]");
    drv_error_output.print();

    LOG("[output]");
    output.print();

    LOG("t4d * drv_error_output_preact = new t4d(" + output_shape.to_string() + ", 1, " + to_string(_acc) + ")");
    t4d * drv_error_output_preact = new t4d(output_shape);
    if (_acc) {
        drv_error_output_preact->create_acc();
    }

    activation_fn.backward(drv_error_output, output, drv_error_output_preact);

    LOG("[drv_error_output_preact]");
    drv_error_output_preact->print();

    return drv_error_output_preact;
}

/// @brief or calculate prev drv error output
t4d * Layer::backprop(double learning_rate, t4d &drv_error_output_preact, t4d &input) {

    this->_backprop(learning_rate, drv_error_output_preact, input);

    t4d *prev_drv_error_output = this->backprop_delta_prev_output(drv_error_output_preact, input);

    return prev_drv_error_output;
}


Weighted::~Weighted() {
    cout << gph() + "destructor" << endl;
}


void Weighted::alloc() {
    cout << gph() + "alloc" << endl;

    cout << "weights = make_unique<t4d>(" << weights_shape.to_string() << ", 1, << " + _acc << "), size: " << weights_shape.size() << endl;
    weights = make_unique<t4d>(weights_shape);
    if (_acc) {
        weights->create_acc();
    }

    LOG("weights rng");
    acc_rng(weights.get(), (double)0.1f);

    LOG("weights init print");
    weights->print();

    cout << "biases = make_unique<t4d>(" << biases_shape.to_string() << ", 1, << " + _acc << "), size: " << biases_shape.size() << endl;
    biases = make_unique<t4d>(biases_shape);
    if (_acc) {
        biases->create_acc();
    }

    LOG("acc_zeros(biases)");
    acc_zeros(biases.get());

    LOG("biases init print");
    biases->print();
}

t4d *Weighted::forward_output(t4d &input) { 
    t4d *output_preact = this->_forward_output(input);

    AddVecDim<double, 1>(output_preact, *biases.get());

    return output_preact;
}

void Weighted::_backprop(double learning_rate, t4d &drv_error_output_preact, t4d &input) {
    unique_ptr<t4d> drv_error_weights(this->backprop_delta_weights(drv_error_output_preact, input)), drv_error_biases(this->backprop_delta_biases(drv_error_output_preact));

    this->update_weights(learning_rate, drv_error_weights.get(), drv_error_biases.get());
}

t4d * Weighted::backprop_delta_biases(t4d &drv_error_output_preact) {
    LOG("drv_error_biases = make_unique<t4d>(" + biases_shape.to_string() + ", 1, " + to_string(_acc) + ")");
    t4d * drv_error_biases = new t4d(biases_shape);
    
    if (_acc) {
        drv_error_biases->create_acc();
    }

    LOG("[drv_error_output_preact]");
    drv_error_output_preact.print();

    LOG("acc_accumulate(*drv_error_output_preact, drv_error_biases)");
    acc_accumulate(drv_error_output_preact, drv_error_biases);

    LOG("[drv_error_biases]");
    drv_error_biases->print();

    double mltp = (1.0f) / drv_error_output_preact.shape()[0];
    acc_mltp(drv_error_biases, mltp);

    return drv_error_biases;
}

void Weighted::update_weights(double learning_rate, t4d *drv_error_weights, t4d *drv_error_biases) {
    LOG(gph() + "Weighted::_backward");

    double mltp = -1.0f * learning_rate;

    acc_mltp(drv_error_weights, mltp);

    LOG("acc_add(weights, *drv_error_weights)");
    acc_add(weights.get(), *drv_error_weights);

    LOG("acc_mltp(drv_error_biases, mltp)");
    acc_mltp(drv_error_biases, mltp);

    //update
    LOG("acc_add(biases, *drv_error_biases)");
    acc_add(biases.get(),  *drv_error_biases);
}

/////////////////////////////////////////////////////////////////

/////////////////////////// <Fc> //////////////////////////////////////
/*
 */

Fc::Fc(Shape4D &prev_shape, int features, string activation_fn) : Weighted(prev_shape, features, activation_fn) {
    layerType = "fc";
    layerOp = "acc_matrix_multiply";
    // TODO function pointers? prev->input, op, error_in->prev_error_out functions

    weights_shape = Shape4D(prev_shape[1]*prev_shape[2]*prev_shape[3], features, 1, 1);
    LOG("weights_shape = " + weights_shape.to_string());

    biases_shape = Shape4D(1, features, 1, 1);
    LOG("biases_shape = " + biases_shape.to_string());
}

Fc::~Fc() {
    cout << gph() + " Fc destructor" << endl;
}

t4d * Fc::forward_input(t4d &prev_output) {
    t4d *input = new t4d(prev_output.shape().flat(1));
    acc_copy(prev_output, input);
    return input;
}

t4d * Fc::_forward_output(t4d &input) {
    t4d * output_preact = new t4d(input.shape()[0], features, 1, 1);
    acc_matrix_multiply(input, *weights.get(), output_preact);
    return output_preact;
}

t4d * Fc::backprop_delta_weights(t4d &drv_error_output_preact, t4d &input) {
    LOG(gph() + "Fc::_backward_weights");

    LOG("unique_ptr<t4d> input_tranposed(acc_transposed<double, 0, 1>(*input.get()))");
    unique_ptr<t4d> input_tranposed(acc_transposed<double, 0, 1>(input));
    LOG("[input_tranposed]");
    input_tranposed->print();
    
    t4d * drv_error_weights = new t4d(weights->shape());

    // DRV ERROR_WEIGHTS = (DRV ERROR_OUTPUT OP) * INPUT prototype
    LOG("acc_matrix_multiply(*input_tranposed.get(), *drv_error_output_preact, drv_error_weights)");
    acc_matrix_multiply(*input_tranposed.get(), drv_error_output_preact, drv_error_weights);

    double mltp =  1.0f/input.shape()[0];
    acc_mltp(drv_error_weights, mltp);
    return drv_error_weights;
}

t4d * Fc::backprop_delta_prev_output(t4d &drv_error_output_preact, t4d &input) {
    LOG(gph() + "Fc::_backward_input");

    LOG("unique_ptr<t4d> weights_transposed(acc_transposed<double,  0, 1>(*weights))");
    unique_ptr<t4d> weights_transposed(acc_transposed<double, 0, 1>(*weights.get()));
    
    LOG("[weights_transposed]");
    weights_transposed->print();

    unique_ptr<t4d> drv_error_input = make_unique<t4d>(input.shape());
    
    LOG("acc_matrix_multiply(*drv_error_output_preact, *weights_transposed.get(), drv_error_input)");
    acc_matrix_multiply(drv_error_output_preact, *weights_transposed.get(), drv_error_input.get());

    t4d * prev_drv_error_output = new t4d(Shape4D(input.shape()[0], prev_shape_proto[1], prev_shape_proto[2], prev_shape_proto[3]));

    LOG("acc_copy(*drv_error_input, *prev_drv_error_output)");
    acc_copy(*drv_error_input.get(), prev_drv_error_output);

    return prev_drv_error_output;
}

/////////////////////////// <Conv> //////////////////////////////////////
/*
 */
Conv::Conv(Shape4D &prev_shape, int features, string activation_fn, vector<int> _filter_size, vector<int> _stride, string _padding_type) : Weighted(prev_shape, features, activation_fn), filter_size{_filter_size[0], _filter_size[1]}, stride{_stride[0], _stride[1]}, padding_type{_padding_type} {
    layerType = "conv";
    layerOp = "acc_convolution2D";

    weights_shape = Shape4D(features, prev_shape[1], filter_size[0], filter_size[1]);
    biases_shape = Shape4D(1, features, 1, 1);

    helper_CalcConvPaddingOutSize(padding_type, prev_shape[2], prev_shape[3], filter_size, stride, out_height, out_width, padding);
}

Conv::~Conv() {
    cout << gph() + "Conv destructor" << endl;
}

t4d * Conv::forward_input(t4d &prev_output) {
    Shape4D prev_shape = prev_output.shape();

    Shape4D input_shape(prev_shape[0], prev_shape[1], prev_shape[2] + padding[0] + padding[1], prev_shape[3] + padding[2] + padding[3]);

    t4d *input = new t4d(input_shape);

    if(is_padded()) {
        acc_zeros(input);
        acc_pad2D(prev_output, input, padding[0], padding[1], padding[2], padding[3]);
    }
    else {
        acc_copy(prev_output, input);
    }

    return input;
}

t4d * Conv::_forward_output(t4d &input) {
    Shape4D output_shape(input.shape()[0], features, out_height, out_width);

    t4d *output_preact = new t4d(output_shape);

    acc_convolution2D(input, *weights.get(), output_preact, stride);

    return output_preact;
}

t4d * Conv::backprop_delta_weights(t4d &drv_error_output_preact, t4d &input) {
    LOG(gph() + "_backward_weights");

    Shape4D input_shape = input.shape(), output_shape = drv_error_output_preact.shape();

    int rev_padding_h = filter_size[0] - 1 + input_shape[2] - output_shape[2], rev_padding_w = filter_size[1] - 1 + input_shape[3] - output_shape[3];

    // TODO pad same tensor? transpose same tensor?

    LOG("unique_ptr<t4d> drv_error_output_preact_transposed_flipped_padded(acc_transposed<double, 0, 1>(*drv_error_output_preact))");
    unique_ptr<t4d> drv_error_output_preact_transposed_flipped_padded(acc_transposed<double, 0, 1>(drv_error_output_preact));
    
    LOG("[drv_error_output_preact_transposed]");
    drv_error_output_preact_transposed_flipped_padded->print();

    LOG("acc_flip_spatial(drv_error_output_preact_transposed_flipped_padded.get())");
    acc_flip_spatial(drv_error_output_preact_transposed_flipped_padded.get());

    LOG("[drv_error_output_preact_transposed_flipped]");
    drv_error_output_preact_transposed_flipped_padded->print();

    LOG("drv_error_output_preact_transposed_flipped_padded.reset(acc_padded2D_inner(*drv_error_output_preact_transposed_flipped_padded.get(), rev_padding_h - rev_padding_h/2,  rev_padding_w - rev_padding_w/2, rev_padding_h/2,  rev_padding_w/2, 0, 0))");
    drv_error_output_preact_transposed_flipped_padded.reset(acc_padded2D_inner(*drv_error_output_preact_transposed_flipped_padded.get(), rev_padding_h - rev_padding_h / 2, rev_padding_w - rev_padding_w / 2, rev_padding_h / 2, rev_padding_w / 2, 0, 0));
    
    LOG("[drv_error_output_preact_transposed_flipped_padded]");
    drv_error_output_preact_transposed_flipped_padded->print();

    LOG("unique_ptr<t4d> input_transposed_flipped(acc_transposed<double, 0, 1>(*input))");
    unique_ptr<t4d> input_transposed_flipped(acc_transposed<double, 0, 1>(input));
    
    LOG("[input_transposed]");
    input_transposed_flipped->print();

    LOG("acc_flip_spatial(input_transposed_flipped)");
    acc_flip_spatial(input_transposed_flipped.get());

    LOG("[input_transposed_flipped]");
    input_transposed_flipped->print();

    t4d * drv_error_weights = new t4d(weights->shape());
    //TODO check if acc on anything new for return
    // clock_t start=clock();
    LOG("acc_convolution2D(*drv_error_output_preact_transposed_flipped_padded.get(), *input, drv_error_weights, {1, 1})");
    acc_convolution2D(*drv_error_output_preact_transposed_flipped_padded.get(), *input_transposed_flipped.get(), drv_error_weights, {1, 1});

    double mltp = 1.0f/input.shape()[0];
    acc_mltp(drv_error_weights, mltp);

    return drv_error_weights;
    // cout << "acc_convolution2D backward duration: " << dur(start) << std::setprecision(15) << std::fixed << endl;
    // Shape4D in_s = drv_error_output_preact_transposed_flipped_padded->shape(), out_s = drv_error_weights->shape(), w_s = input_transposed_flipped->shape();
    // int bs = in_s[0], stri = 1, fs = w_s[2];
    // // cout << "tparallel_conv5(drv_error_output_preact_transposed_flipped_padded->data(), input_transposed_flipped->data(),  drv_error_weights->data(), in_s[0], in_s[1], in_s[2], in_s[3], out_s[1], out_s[2], out_s[3], w_s[2], stri, false);" << endl;
    // start=clock();
    // tparallel_conv5(drv_error_output_preact_transposed_flipped_padded->data(), input_transposed_flipped->data(),  drv_error_weights->data(), in_s[0], in_s[1], in_s[2], in_s[3], out_s[1], out_s[2], out_s[3], w_s[2], stri, false);
    // cout << "tparallel_conv5 backward duration: " << dur(start) << std::setprecision(15) << std::fixed << endl;
    // TODO calculate error biases
}

// TODO input, drv_error_output not copies?
t4d * Conv::backprop_delta_prev_output(t4d &drv_error_output_preact, t4d &input) {
    LOG(gph() + "_backward_input");
    // formula is rev_p = 2*(F-1) + (S-1)*(O-1)
    // drv_error_output_preact[B D H2 W2]->drv_error_output_preact[B D (H2 + rev_p[0]) (W2 + rev_p[1]);
    LOG("unique_ptr<t4d> drv_error_output_preact_padded(acc_padded2D_inner(*drv_error_output_preact,  filter_size[0]-1,  filter_size[0]-1, filter_size[1]-1,  filter_size[1]-1, stride[0]-1, stride[1]-1))");
    unique_ptr<t4d> drv_error_output_preact_padded(acc_padded2D_inner(drv_error_output_preact, filter_size[0] - 1, filter_size[0] - 1, filter_size[1] - 1, filter_size[1] - 1, stride[0] - 1, stride[1] - 1));

    // weights[D C F1 F2]->[C D F1 F2], and flip F1, F2 elements
    LOG("unique_ptr<t4d> weights_transposed_flipped(acc_transposed<double, 0, 1>(*weights))");
    unique_ptr<t4d> weights_transposed_flipped(acc_transposed<double, 0, 1>(*weights.get()));

    LOG("acc_flip_spatial(weights_transposed_flipped.get())");
    acc_flip_spatial(weights_transposed_flipped.get());

    unique_ptr<t4d> drv_error_input = make_unique<t4d>(input.shape());
    // = ERROR_INPUT = ERROR_OUTPUT * WEIGHTS
    
    LOG("acc_convolution2D(*drv_error_output_preact_padded.get(), *weights_transposed_flipped.get(), drv_error_input, {1, 1})");
    acc_convolution2D(*drv_error_output_preact_padded.get(), *weights_transposed_flipped.get(), drv_error_input.get(), {1, 1});

    t4d *prev_drv_error_output = new t4d(Shape4D(input.shape()[0], prev_shape_proto[1], prev_shape_proto[2], prev_shape_proto[3]));
    // update prev drv error output act
    // TODO transfer responsibly, maybe no padding was applied etc..no memory traces, no duplicates
    LOG("acc_rev_pad2D(*drv_error_input, *prev_drv_error_output, padding[0], padding[1], padding[2], padding[3])");
    acc_rev_pad2D(*drv_error_input.get(), prev_drv_error_output, padding[0], padding[1], padding[2], padding[3]);

    return prev_drv_error_output;
}
/////////////////////////////////////////////////////////////////
