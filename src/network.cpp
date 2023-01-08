#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "network.hpp"
#include "ops.hpp"

using namespace std;

using Neural::Network;
using Neural::Tensor4D;
using Neural::Shape4D;

typedef Tensor4D<double> t4d;

Neural::Network::Network(Shape4D in_sh_pr) : input_shape_proto(Shape4D(-1, in_sh_pr[1], in_sh_pr[2], in_sh_pr[3])) {
    LOGV << "Network::Network";
    LOGD << "input_shape_proto: " << input_shape_proto.to_string();
}

Network::~Network() {
    LOGV << "Network destructor";
    for(auto it : layers) {
        delete it;
    }
}

void Network::train(const Tensor4D<double> * train_dataset, const Tensor4D<int> * train_labels, const Tensor4D<double> * valid_dataset, const Tensor4D<int> * valid_labels, int batch_size, bool acc, double learning_rate, string loss_fn, int fepoch, int fsteps) {
    PLOGI << "Network::train | batch_size: " << batch_size;

    Shape4D train_shape = train_dataset->shape(), labels_shape = train_labels->shape();
    PLOGI  << "Train shape: " << train_shape.to_string();
    
    int train_num_samples = train_shape[0], train_size = train_dataset->size();
    int train_num_outputs = labels_shape[1];
    
    assert( (train_shape[1] == input_shape_proto[1]) && (train_shape[2] == input_shape_proto[2]) && (train_shape[3] == input_shape_proto[3]));
    assert(train_num_samples == labels_shape[0]);
    assert(batch_size <= train_num_samples);
    
    unique_ptr<t4d> batch_data;
    unique_ptr<Tensor4D<int>> batch_labels;
    
    /////////////////////////
    {    
        Shape4D batch_data_shape(batch_size, train_shape[1], train_shape[2], train_shape[3]);
        LOGD << "batch_data_shape = " << batch_data_shape.to_string();
        batch_data = make_unique<t4d>(batch_data_shape);
        batch_data->create_acc();
    }
    
    {
        Shape4D batch_labels_shape(batch_size, labels_shape[1], labels_shape[2], labels_shape[3]);
        LOGD << "batch_labels_shape = " << batch_labels_shape.to_string();
        batch_labels = make_unique<Tensor4D<int>>(batch_labels_shape);
        batch_labels->create_acc();
    }
    //////////////////////////
    
    PLOGI << "Calling Layer::init";
    int lnn = 0;
    for(auto it: layers) {
        PLOGD << "Layer " << ++lnn << " init";
        it->init();
    }
    int iters = train_num_samples/batch_size, batch_start;
    
    clock_t train_start = clock();
    
    PLOGI.printf("Steps per epoch: %d", iters);
    int e = 0;
    double epoch_loss;
    do {
        epoch_loss = 0.0f;
        clock_t epoch_start = clock();
        int iter=0;
        do {
            clock_t iter_start = clock();

            batch_start = (iter*batch_size)%(train_num_samples-batch_size+1);

            IF_PLOG(plog::debug) {
                printf("Step %d, batch_start: %d, batch_size: %d | ",iter, batch_start, batch_size);
            }

            clock_t op_start;
            string op_name;
            
            IF_PLOG(plog::debug) { op_name = "acc_make_batch"; PLOGV << op_name; op_start = clock(); }
            acc_make_batch<double>(*train_dataset, batch_data.get(),  batch_start);
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG(debug, batch_data);            

            IF_PLOG(plog::debug) { op_name = "acc_normalize_img"; PLOGV << op_name; op_start = clock(); }
            acc_normalize_img(batch_data.get());
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG_A(debug, batch_data, "batch_data_normalized")

            IF_PLOG(plog::debug) { op_name = "acc_make_batch[labels]"; PLOGV << op_name; op_start = clock(); }
            acc_make_batch<int>(*train_labels, batch_labels.get(), batch_start);
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG(debug, batch_labels);

            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            vector<t4d *> inputs, outputs;
            t4d *prev_output = batch_data.get();

            for(int i = 0; i < layers.size(); i++) {
                PLOGD.printf("Forward Layer %d", i);
                
                _LLOG(debug, prev_output);
                
                IF_PLOG(plog::debug) { op_name = "forward_calc_input"; PLOGV << op_name; op_start = clock(); }
                inputs.push_back(layers[i]->forward_calc_input(*prev_output));
                PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                _LLOG(debug, inputs[i]);

                {
                    IF_PLOG(plog::debug) { op_name = "forward_calc_output_preact"; PLOGV << op_name; op_start = clock(); }    
                    unique_ptr<t4d> output_preact(layers[i]->forward_calc_output_preact(*(inputs[i])));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                    _LLOG(debug, output_preact);

                    IF_PLOG(plog::debug) { op_name = "forward_activate"; PLOGV << op_name; op_start = clock(); }
                    outputs.push_back(layers[i]->forward_activate(*output_preact));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                    _LLOG(debug, outputs[i]);
                }

                prev_output = outputs[i];
            }                

            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";   
            
            PLOGD << "Backpropagating";
            double loss;
            unique_ptr<t4d> drv_error_output_preact, drv_error_prev_output;

            for(int i = layers.size()-1; i>=0; i--) {
                PLOGD.printf("Backward Layer %d", i);
                
                _LLOG(debug, outputs[i]);

                if(i==(layers.size()-1)) {
                    IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_output_preact(loss)"; PLOGV << op_name; op_start = clock(); }    
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(loss_fn, loss, *(outputs[i]), *batch_labels.get()));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                    
                    PLOGD << "Epoch loss: " << epoch_loss << " += " << loss;
                    epoch_loss += loss;
                }
                else {
                    IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_output_preact"; PLOGV << op_name; op_start = clock(); }    
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(*drv_error_prev_output.get(), *(outputs[i])));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                }

                PLOGV.printf("delete outputs[%d]", i);
                delete outputs[i];

                _LLOG(debug, drv_error_output_preact);
                
                IF_PLOG(plog::debug) { op_name = "backprop_update"; PLOGV << op_name; op_start = clock(); }    
                layers[i]->backprop_update(learning_rate, *drv_error_output_preact.get(), *inputs[i]);
                PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                
                IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_prev_output"; PLOGV << op_name; op_start = clock(); }    
                drv_error_prev_output.reset(layers[i]->backprop_calc_drv_error_prev_output(*drv_error_output_preact.get(), *inputs[i]));
                PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);

                PLOGV.printf("delete inputs[%d]", i);
                delete inputs[i];
            }
            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";

            PLOGI.printf("Step %d - loss: %11.6f, epoch_loss: %11.6f, duration: %20.15f", iter, loss, epoch_loss, dur(iter_start));
            iter++;
        }
        while((iter < iters) && ( (fsteps==0) || (iter<fsteps) ) );
        PLOGI << "Epoch [" << e << "] loss: " << epoch_loss << ", duration: " << dur(epoch_start);
        
        e++;
        
    }
    while( (epoch_loss < -0.05f) && ( (fepoch==0) || (e < fepoch)) );
    
    PLOGI << "Train duration: " <<  std::setprecision(15) << std::fixed << dur(train_start);

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

