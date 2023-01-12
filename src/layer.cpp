#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cassert>
#include <iomanip>
#include "layer.hpp"
#include "utils.hpp"
#include "ops.hpp"

using Neural::Tensor4D;
using Neural::Shape4D;

typedef Tensor4D<double> t4d;

using namespace std;

void assert_shape(Shape4D actual, Shape4D proto) {
    assert((actual[0]!=-1) && (actual[1]==proto[1]) && (actual[2]==proto[2]) && (actual[3]==proto[3]));
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

Layer::Layer(Shape4D prev_shape, int features, string afn) : prev_shape_proto(Shape4D(-1, prev_shape[1], prev_shape[2], prev_shape[3])), features(features) {
    LOGD << "Layer::Layer";

    LOGD << "Layer::prev_shape: " << prev_shape.to_string() << ", prev_shape_proto: " << this->prev_shape_proto.to_string();
    LOGD << "Generating layer number from " << nl;
    id = ++nl;
    LOGD << "Generated " << id;

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

t4d * Layer::forward_activate(t4d &output_preact) {
    LOGD << gph() + "Activation: " + activation_fn.name();

    Shape4D output_shape = output_preact.shape();

    assert_shape(output_shape, output_shape_proto);

    LOGD << "output = make_unique<t4d>(" + output_shape.to_string() + ", 1, " + to_string(_acc) + ")";
    t4d * output = new t4d(output_shape);
    output->create_acc();

    // helper_InnerActivate(*output_preact, output, activation_fn);
    activation_fn.apply(output_preact, output);

    return output;
}

t4d * Layer::backprop_calc_drv_error_output_preact(string loss_fn, double &loss_value, t4d & output, Tensor4D<int> &labels_batch) {
    LOGD << gph() + "Layer::backprop_calc_loss";
    
    LOGD << "activation_type: " << this->get_activation_name();
    string activation_type = this->get_activation_name();

    Shape4D output_shape = output.shape();
    assert_shape(output_shape, output_shape_proto);
    assert(labels_batch.shape() == output_shape);

    LOGD << "drv_error_output_preact = new t4d(" + output_shape.to_string() + ", 1, " + to_string(_acc) + ")";
    t4d *drv_error_output_preact = new t4d(output_shape);
    drv_error_output_preact->create_acc();

    LOGD << "Layer::loss getting data pointers";
    double *output_data = output.data(), *drv_error_output_preact_data = drv_error_output_preact->data();
    int *labels_data = labels_batch.data();

    int B = output_shape[0], M = output_shape[1];
    
    _LLOG(debug, (&output));

    _LLOG(debug, (&labels_batch));

    int lsize = output_shape.size();

    loss_value = 0.0f;
    LOGD << "loss_value = " << loss_value;

    if ((loss_fn == "CrossEntropy") && (activation_type == "softmax")) {
        //calculating loss
        #pragma acc parallel loop collapse(2) reduction(+:loss_value) present(labels_data[:lsize], output_data[:lsize])
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < M; j++) {
                int lblint = labels_data[i * M + j];
                double lblval = lblint, oval = output_data[i * M + j];
                double loval = log(oval);
                double val = lblval * loval;
                #ifndef _OPENACC
                LOGD << "Loss value += " << lblval << " * log(" << oval << ") = " << val;
                #endif
                loss_value += val;
            }
        }
        LOGD << "loss_value /= -1/" << B;
        loss_value *= -1;
        loss_value /= B;

        //skiping de-derivation
        #pragma acc parallel loop collapse(2) present(labels_data[:lsize], output_data[:lsize], drv_error_output_preact_data[:lsize])
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < M; j++) {
                double d_lbl = (double)labels_data[i * M + j];
                drv_error_output_preact_data[i * M + j] = output_data[i * M + j] - d_lbl;
            }
        }
    }

    LOGD << "loss_value = " << loss_value;
    return drv_error_output_preact;
}

t4d * Layer::backprop_calc_drv_error_output_preact(t4d &drv_error_output, t4d &output) {
    LOGD << gph() + "backprop_delta_output";

    Shape4D output_shape = output.shape();
    assert_shape(output_shape, output_shape_proto);
    assert(drv_error_output.shape() == output_shape);

    // If not set from somewhere else calc here drv_error_output_preact
    
    _LLOG(debug, (&drv_error_output));

    _LLOG(debug, (&output));

    LOGD << "t4d * drv_error_output_preact = new t4d(" + output_shape.to_string() + ", 1, " + to_string(_acc) + ")";
    t4d * drv_error_output_preact = new t4d(output_shape);
    drv_error_output_preact->create_acc();

    activation_fn.backward(drv_error_output, output, drv_error_output_preact);

    _LLOG(debug, drv_error_output_preact);

    return drv_error_output_preact;
}

Weighted::Weighted(Shape4D prev_shape_proto, int features, string afn) : Layer(prev_shape_proto, features, afn) {
}

Weighted::~Weighted() {
    LOGD << gph() + "destructor";
}

void Weighted::init() {
    LOGI << gph() + "::init";
    LOGI << "features: " << features;
    LOGI << "prev_shape_proto: " << prev_shape_proto.to_string();
    LOGI << "input_shape_proto: " << input_shape_proto.to_string();
    LOGI << "output_shape_proto: " << output_shape_proto.to_string();

    LOGI << "weights = make_unique<t4d>(" << weights_shape.to_string() << ")";
    weights = make_unique<t4d>(weights_shape);
    weights->create_acc();
    LOGI << "weights rng";
    acc_rng(weights.get(), (double)0.01f);
    _LLOG(debug, weights);

    LOGI << "biases = make_unique<t4d>(" << biases_shape.to_string()<< ")";
    biases = make_unique<t4d>(biases_shape);
    biases->create_acc();
    LOGI << "acc_zeros(biases)";
    acc_zeros(biases.get());
    _LLOG(debug, biases);
}

void Weighted::backprop_update(double learning_rate, t4d &drv_error_output_preact, t4d &input) {
    LOGD << gph() + "Weighted::backprop_update";
    LOGD << "learning_rate: " << learning_rate;
    Shape4D output_shape = drv_error_output_preact.shape(), input_shape = input.shape();
    assert_shape(output_shape, output_shape_proto);
    assert_shape(input_shape, input_shape_proto);

    unique_ptr<t4d> drv_error_weights(this->backprop_calc_drv_error_weights(drv_error_output_preact, input)), drv_error_biases(this->backprop_calc_drv_error_biases(drv_error_output_preact));

    double mltp = -1.0f * learning_rate;

    _LLOG_A(debug, drv_error_weights, "drv_error_weights non learning-rate");
    acc_mltp(drv_error_weights.get(), mltp);
    _LLOG(debug, drv_error_weights);
    LOGD << "acc_add(weights, *drv_error_weights)";
    _LLOG_A(debug, weights, "weightes pre-add");
    acc_add(weights.get(), *drv_error_weights.get());
    _LLOG(debug, weights);

    _LLOG_A(debug, drv_error_biases, "drv_error_biases non learning_rate");
    LOGD << "acc_mltp(drv_error_biases, mltp)";
    acc_mltp(drv_error_biases.get(), mltp);
    _LLOG(debug, drv_error_biases);
    //update
    LOGD << "acc_add(biases, *drv_error_biases)";
    _LLOG_A(debug, biases, "biases pre-add");
    acc_add(biases.get(),  *drv_error_biases.get());
    _LLOG(debug, biases);
}

t4d * Weighted::backprop_calc_drv_error_biases(t4d &drv_error_output_preact) {
    LOGD << gph() + "Weighted::backprop_calc_drv_error_biases";
    Shape4D output_shape = drv_error_output_preact.shape();
    assert_shape(output_shape, output_shape_proto);

    LOGD << "drv_error_biases = make_unique<t4d>(" + biases_shape.to_string() + ", 1, " + to_string(_acc) + ")";
    t4d * drv_error_biases = new t4d(biases_shape); 
    drv_error_biases->create_acc();

    _LLOG(debug, (&drv_error_output_preact));
    LOGD << "acc_accumulate(*drv_error_output_preact, drv_error_biases)";
    acc_accumulate(drv_error_output_preact, drv_error_biases);
    
    _LLOG_A(debug, drv_error_biases, "drv_error_biases non batch-normalized");

    double mltp = (1.0f) / drv_error_output_preact.shape()[0];
    LOGD << "Normalizing biases by 1/" << drv_error_output_preact.shape()[0] << " = " << mltp;
    acc_mltp(drv_error_biases, mltp);
    _LLOG(debug, drv_error_biases); 

    return drv_error_biases;
}

/////////////////////////////////////////////////////////////////

/////////////////////////// <Fc> //////////////////////////////////////
/*
 */

Fc::Fc(Shape4D prev_shape_proto, int features, string activation_fn) : Weighted(prev_shape_proto, features, activation_fn) {
    layerType = "fc";
    layerOp = "acc_matrix_multiply";
    // TODO function pointers? prev->input, op, error_in->prev_error_out functions
    LOGD << "prev_shape_proto:" << prev_shape_proto.to_string();
    input_shape_proto = Shape4D(-1, prev_shape_proto[1]*prev_shape_proto[2]*prev_shape_proto[3], 1, 1);
    LOGD << "input_shape_proto:" << input_shape_proto.to_string();
    output_shape_proto = Shape4D(-1, features, 1, 1);
    LOGD << "output_shape_proto:" << output_shape_proto.to_string();
    weights_shape = Shape4D(input_shape_proto[1], output_shape_proto[1], 1, 1);
    LOGD << "weights_shape = " + weights_shape.to_string();
    biases_shape = Shape4D(1, output_shape_proto[1], 1, 1);
    LOGD << "biases_shape = " << biases_shape.to_string();
}

Fc::~Fc() {
    LOGD << gph() + " Fc destructor";
}

t4d * Fc::forward_calc_input(t4d &prev_output) {
    LOGD << gph() + "Fc::forward_calc_input";
    Shape4D prev_shape = prev_output.shape();
    assert_shape(prev_shape, prev_shape_proto);

    _LLOG(debug, (&prev_output));
    t4d *input = new t4d(prev_shape[0], input_shape_proto[1], input_shape_proto[2], input_shape_proto[3]);
    input->create_acc();
    LOGD << "acc_copy(prev_output, input)";
    acc_copy(prev_output, input);
    _LLOG(debug, input);
    return input;
}

t4d * Fc::forward_calc_output_preact(t4d &input) {
    LOGD << gph() + "Fc::forward_calc_output";
    Shape4D input_shape = input.shape();
    assert_shape(input_shape, input_shape_proto);
    t4d * output_preact = new t4d(input_shape[0], output_shape_proto[1], output_shape_proto[2], output_shape_proto[3]);
    output_preact->create_acc();

    _LLOG(debug, weights);
    LOGD << "acc_matrix_multiply(input, *weights.get(), output_preact)";
    acc_matrix_multiply(input, *weights.get(), output_preact);
    _LLOG_A(debug, output_preact, "output_preact non-biases");

    _LLOG(debug, biases);
    LOGD << "AddVecDim<double, 1>(output_preact, *biases.get())";
    AddVecDim<double, 1>(output_preact, *biases.get());
    return output_preact;
}

t4d * Fc::backprop_calc_drv_error_weights(t4d &drv_error_output_preact, t4d &input) {
    LOGD << gph() + "Fc::_backward_weights";
    Shape4D input_shape = input.shape(), output_shape = drv_error_output_preact.shape();
    assert_shape(input_shape, input_shape_proto);
    assert_shape(output_shape, output_shape_proto);

    _LLOG(debug, (&input));
    LOGD << "unique_ptr<t4d> input_tranposed(acc_transposed<double, 0, 1>(*input.get()))";
    unique_ptr<t4d> input_tranposed(acc_transposed<double, 0, 1>(input));
    
    _LLOG(debug, input_tranposed);
    
    t4d * drv_error_weights = new t4d(weights->shape());
    drv_error_weights->create_acc();
    // DRV ERROR_WEIGHTS = (DRV ERROR_OUTPUT OP) * INPUT prototype
    LOGD << "acc_matrix_multiply(*input_tranposed.get(), *drv_error_output_preact, drv_error_weights)";
    acc_matrix_multiply(*input_tranposed.get(), drv_error_output_preact, drv_error_weights);
    _LLOG_A(debug, drv_error_weights, "drv_error_weights non-batch-normalized");
    double mltp = 1.0f/input_shape[0];
    acc_mltp(drv_error_weights, mltp);
    _LLOG(debug, drv_error_weights);
    return drv_error_weights;
}

t4d * Fc::backprop_calc_drv_error_prev_output(t4d &drv_error_output_preact, t4d &input) {
    LOGD << gph() + "Fc::_backward_input";
    Shape4D input_shape = input.shape(), output_shape = drv_error_output_preact.shape();
    assert_shape(input_shape, input_shape_proto);
    assert_shape(output_shape, output_shape_proto);

    _LLOG(debug, weights);
    LOGD << "unique_ptr<t4d> weights_transposed(acc_transposed<double,  0, 1>(*weights))";
    unique_ptr<t4d> weights_transposed(acc_transposed<double, 0, 1>(*weights.get()));
    
    _LLOG(debug, weights_transposed);

    unique_ptr<t4d> drv_error_input = make_unique<t4d>(input_shape);
    drv_error_input->create_acc();
    LOGD << "acc_matrix_multiply(*drv_error_output_preact, *weights_transposed.get(), drv_error_input)";
    acc_matrix_multiply(drv_error_output_preact, *weights_transposed.get(), drv_error_input.get());
    _LLOG(debug, drv_error_input);

    t4d * prev_drv_error_output = new t4d(Shape4D(output_shape[0], prev_shape_proto[1], prev_shape_proto[2], prev_shape_proto[3]));
    prev_drv_error_output->create_acc();
    LOGD << "acc_copy(*drv_error_input, *prev_drv_error_output)";
    acc_copy(*drv_error_input.get(), prev_drv_error_output);
    _LLOG(debug, prev_drv_error_output);
    return prev_drv_error_output;
}

/////////////////////////// <Conv> //////////////////////////////////////
/*
 */
Conv::Conv(Shape4D prev_shape, int features, string activation_fn, vector<int> _filter_size, vector<int> _stride, string _padding_type) : Weighted(prev_shape, features, activation_fn), filter_size{_filter_size[0], _filter_size[1]}, stride{_stride[0], _stride[1]}, padding_type{_padding_type} {
    layerType = "conv";
    layerOp = "acc_convolution2D";

    int in_h = prev_shape_proto[2], in_w = prev_shape_proto[3];

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
        padding[2] = padding_width - padding_width / 2;
        padding[3] = padding_width / 2;

        out_height = in_h;
        out_width = in_w;
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
        out_height = nom_h / stride[0] + 1;
        out_width = nom_w / stride[1] + 1;
    }
    else
    {
        throw(std::invalid_argument("Padding type not compatible."));
    }
    LOGD << "prev_shape_proto:" << prev_shape_proto.to_string();
    input_shape_proto = Shape4D(-1, prev_shape_proto[1], prev_shape_proto[2] + padding[0] + padding[1], prev_shape_proto[3] + padding[2] + padding[3]);
    LOGD << "input_shape_proto:" << input_shape_proto.to_string();
    output_shape_proto = Shape4D(-1, features, out_height, out_width);
    LOGD << "output_shape_proto:" << output_shape_proto.to_string();
    weights_shape = Shape4D(output_shape_proto[1], input_shape_proto[1], filter_size[0], filter_size[1]);
    LOGD << "weights_shape = " << weights_shape.to_string();
    biases_shape = Shape4D(1, output_shape_proto[1], 1, 1);
    LOGD << "biases_shape = " << biases_shape.to_string();
}

Conv::~Conv() {
    LOGD << gph() + "Conv destructor";
}

t4d * Conv::forward_calc_input(t4d &prev_output) {
    LOGD << gph() + "forward_calc_input";

    Shape4D prev_shape = prev_output.shape();
    assert_shape(prev_shape, prev_shape_proto);

    t4d *input = new t4d(prev_shape[0], input_shape_proto[1], input_shape_proto[2], input_shape_proto[3]);
    input->create_acc();
    _LLOG(debug, (&prev_output));
    if(is_padded()) {
        acc_zeros(input);
        LOGD.printf("acc_pad2D(prev_output, input, %d, %d, %d, %d", padding[0], padding[1], padding[2], padding[3]);
        acc_pad2D(prev_output, input, padding[0], padding[1], padding[2], padding[3]);
    }
    else {
        LOGD << "acc_copy(prev_output, input)";
        acc_copy(prev_output, input);
    }
    _LLOG(debug, input);
    return input;
}

t4d * Conv::forward_calc_output_preact(t4d &input) {
    LOGD << gph() + "forward_calc_output_preact";
    Shape4D input_shape = input.shape();
    assert_shape(input_shape, input_shape_proto);

    t4d *output_preact = new t4d(input_shape[0], output_shape_proto[1], output_shape_proto[2], output_shape_proto[3]);
    output_preact->create_acc();

    _LLOG(debug, (&input));
    _LLOG(debug, weights);
    LOGD.printf("acc_convolution2D(input, *weights.get(), output_preact, stride={%d, %d})", stride[0], stride[1]);
    acc_convolution2D(input, *weights.get(), output_preact, stride);
    _LLOG_A(debug, output_preact, "output_preact non-biases");
    _LLOG(debug, biases);
    LOGD << "AddVecDim<double, 1>(output_preact, *biases.get())";
    AddVecDim<double, 1>(output_preact, *biases.get());
    _LLOG(debug, output_preact);
    return output_preact;
}

t4d * Conv::backprop_calc_drv_error_weights(t4d &drv_error_output_preact, t4d &input) {
    LOGD << gph() + "_backward_weights";

    Shape4D input_shape = input.shape(), output_shape = drv_error_output_preact.shape();
    assert_shape(input_shape, input_shape_proto);
    assert_shape(output_shape, output_shape_proto);

    int rev_padding_h = filter_size[0] - 1 + input_shape[2] - output_shape[2], rev_padding_w = filter_size[1] - 1 + input_shape[3] - output_shape[3];

    // TODO pad same tensor? transpose same tensor?
    _LLOG(debug, (&drv_error_output_preact));
    LOGD << "unique_ptr<t4d> drv_error_output_preact_transposed_flipped_padded(acc_transposed<double, 0, 1>(*drv_error_output_preact))";
    unique_ptr<t4d> drv_error_output_preact_transposed_flipped_padded(acc_transposed<double, 0, 1>(drv_error_output_preact));
    _LLOG_A(debug, drv_error_output_preact_transposed_flipped_padded, "drv_error_output_preact_transposed");

    LOGD << "acc_flip_spatial(drv_error_output_preact_transposed_flipped_padded.get())";
    acc_flip_spatial(drv_error_output_preact_transposed_flipped_padded.get());
    _LLOG_A(debug, drv_error_output_preact_transposed_flipped_padded, "drv_error_output_preact_transposed_flipped");

    LOGD << "drv_error_output_preact_transposed_flipped_padded.reset(acc_padded2D_inner(*drv_error_output_preact_transposed_flipped_padded.get(), rev_padding_h - rev_padding_h/2,  rev_padding_w - rev_padding_w/2, rev_padding_h/2,  rev_padding_w/2, 0, 0))";
    drv_error_output_preact_transposed_flipped_padded.reset(acc_padded2D_inner(*drv_error_output_preact_transposed_flipped_padded.get(), rev_padding_h - rev_padding_h / 2, rev_padding_w - rev_padding_w / 2, rev_padding_h / 2, rev_padding_w / 2, 0, 0));
    _LLOG(debug, drv_error_output_preact_transposed_flipped_padded);

    _LLOG(debug, (&input));
    LOGD << "unique_ptr<t4d> input_transposed_flipped(acc_transposed<double, 0, 1>(*input))";
    unique_ptr<t4d> input_transposed_flipped(acc_transposed<double, 0, 1>(input));
    _LLOG_A(debug, input_transposed_flipped, "input_transposed");

    LOGD << "acc_flip_spatial(input_transposed_flipped)";
    acc_flip_spatial(input_transposed_flipped.get());
    _LLOG(debug, input_transposed_flipped);

    t4d * drv_error_weights = new t4d(weights->shape());
    drv_error_weights->create_acc();
    
    //TODO check if acc on anything new for return

    LOGD << "acc_convolution2D(*drv_error_output_preact_transposed_flipped_padded.get(), *input_transposed_flipped.get(), drv_error_weights, {1, 1})";
    acc_convolution2D(*drv_error_output_preact_transposed_flipped_padded.get(), *input_transposed_flipped.get(), drv_error_weights, {1, 1});
    _LLOG_A(debug, drv_error_weights, "drv_error_weights_non_normalized");

    double mltp = 1.0f/input.shape()[0];
    acc_mltp(drv_error_weights, mltp);
    _LLOG(debug, drv_error_weights);
    
    return drv_error_weights;
}

// TODO input, drv_error_output not copies?
t4d * Conv::backprop_calc_drv_error_prev_output(t4d &drv_error_output_preact, t4d &input) {
    LOGD << gph() + "_backward_input";

    Shape4D input_shape = input.shape(), output_shape = drv_error_output_preact.shape();
    assert_shape(input_shape, input_shape_proto);
    assert_shape(output_shape, output_shape_proto);

    _LLOG(debug, (&drv_error_output_preact));
    // formula is rev_p = 2*(F-1) + (S-1)*(O-1)
    // drv_error_output_preact[B D H2 W2]->drv_error_output_preact[B D (H2 + rev_p[0]) (W2 + rev_p[1]);
    LOGD << "unique_ptr<t4d> drv_error_output_preact_padded(acc_padded2D_inner(*drv_error_output_preact,  filter_size[0]-1,  filter_size[0]-1, filter_size[1]-1,  filter_size[1]-1, stride[0]-1, stride[1]-1))";
    unique_ptr<t4d> drv_error_output_preact_padded(acc_padded2D_inner(drv_error_output_preact, filter_size[0] - 1, filter_size[0] - 1, filter_size[1] - 1, filter_size[1] - 1, stride[0] - 1, stride[1] - 1));
    _LLOG(debug, drv_error_output_preact_padded);

    _LLOG(debug, weights);
    // weights[D C F1 F2]->[C D F1 F2], and flip F1, F2 elements
    LOGD << "unique_ptr<t4d> weights_transposed_flipped(acc_transposed<double, 0, 1>(*weights))";
    unique_ptr<t4d> weights_transposed_flipped(acc_transposed<double, 0, 1>(*weights.get()));
    _LLOG_A(debug, weights_transposed_flipped, "weights_transposed");

    LOGD << "acc_flip_spatial(weights_transposed_flipped.get())";
    acc_flip_spatial(weights_transposed_flipped.get());
    _LLOG(debug, weights_transposed_flipped);

    unique_ptr<t4d> drv_error_input = make_unique<t4d>(input.shape());
    drv_error_input->create_acc();
    // = ERROR_INPUT = ERROR_OUTPUT * WEIGHTS
    
    LOGD << "acc_convolution2D(*drv_error_output_preact_padded.get(), *weights_transposed_flipped.get(), drv_error_input, {1, 1})";
    acc_convolution2D(*drv_error_output_preact_padded.get(), *weights_transposed_flipped.get(), drv_error_input.get(), {1, 1});
    _LLOG(debug, drv_error_input);

    t4d *prev_drv_error_output = new t4d(Shape4D(input.shape()[0], prev_shape_proto[1], prev_shape_proto[2], prev_shape_proto[3]));
    prev_drv_error_output->create_acc();
    // update prev drv error output act
    // TODO transfer responsibly, maybe no padding was applied etc..no memory traces, no duplicates
    LOGD << "acc_rev_pad2D(*drv_error_input, *prev_drv_error_output, padding[0], padding[1], padding[2], padding[3])";
    acc_rev_pad2D(*drv_error_input.get(), prev_drv_error_output, padding[0], padding[1], padding[2], padding[3]);
    _LLOG(debug, prev_drv_error_output);
    return prev_drv_error_output;
}
/////////////////////////////////////////////////////////////////
