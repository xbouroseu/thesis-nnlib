#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include "network.hpp"
#include "ops.hpp"

//TODO solve prev_l, weight initialization, can be on ctor? can pass network last layer as first argument? or network itself?

//TODO template layers double,float?

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

// void Network::set_acc(bool acc) {
//     cout << "Network::set_acc{" << acc << endl;
//     for(auto it: layers) {
//         it->set_acc(acc);
//     }
//     _acc = acc;
// }

// void Network::forward() {
//     int l = 1;
//     for(auto it : layers) {
//         LOG("***************************************************************** Layer " << l <<" ****************************************************************************\n\n");
//         it->forward();
//         LOG("\n******************************************************************************************************************************************************\n");
//         l++;
//     }
// }

void Network::train(const Tensor4D<double> * train_dataset, const Tensor4D<int> * train_labels, const Tensor4D<double> * valid_dataset, const Tensor4D<int> * valid_labels, int batch_size, bool acc, double learning_rate, string loss_fn) {
    LOGV << "Network::train | batch_size: " << batch_size;

    
    Shape4D train_shape = train_dataset->shape(), labels_shape = train_labels->shape();
    LOGD  << "Train shape: " << train_shape.to_string();
    
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
        batch_data->create_acc();
    }
    //////////////////////////
    
    LOGD << "Calling Layer::init";
    int lnn = 0;
    for(auto it: layers) {
        LOGD << "Layer " << ++lnn << " init";
        it->init();
    }
    int iters = train_num_samples/batch_size;
    int batch_start;
    double duration;
    clock_t start = clock();
    
    LOGI << "Epoch iterations : " << iters;
    int e = 1;
    double epoch_loss;
    do {
        epoch_loss = 0.0f;
        clock_t epoch_start = clock();
        for(int iter = 0; iter < iters; iter++) {
            clock_t start_iter = clock();
            batch_start = (iter*batch_size)%(train_num_samples-batch_size+1);

            LOGI.printf(">> Step %d, batch_start: %d, batch_size: %d<<",iter, batch_start, batch_size);
            clock_t op_start;

            LOGI << "acc_make_batch";
            op_start = clock();
            acc_make_batch<double>(*train_dataset, batch_data.get(),  batch_start);
            LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);

            IF_PLOG(plog::debug) {
                LOGD << "[batch_data]";
                cout << batch_data->to_string() << endl;
            }
            
            LOGI << "acc_normalize_img";
            op_start = clock();
            acc_normalize_img(batch_data.get());
            LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);

            IF_PLOG(plog::debug) {
                LOGD << "[batch_data_normalized]";
                cout << batch_data->to_string() << endl;
            }   
            
            LOGI << "acc_make_batch[labels]";
            op_start = clock();
            acc_make_batch<int>(*train_labels, batch_labels.get(), batch_start);
            LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);

            IF_PLOG(plog::debug) {
                LOGD << "[batch_labels]";
                cout << batch_labels->to_string() << endl;
            }

            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            vector<t4d *> inputs, outputs;
            t4d *prev_output = batch_data.get();


            for(int i = 0; i < layers.size(); i++) {
                PLOGI.printf("Forward Layer %d", i);
                IF_PLOG(plog::debug) {
                    LOGD << "Layer " << i << " prev_output";
                    cout << prev_output->to_string() << endl;
                }
                
                PLOGI << "forward_calc_input";
                op_start = clock();
                t4d * tinp = layers[i]->forward_calc_input(*prev_output);
                LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);

                IF_PLOG(plog::debug) {
                    LOGD << "Layer " << i << " input";
                    cout << tinp->to_string() << endl;
                }
                inputs.push_back(tinp);
                
                LOGI << "forward_calc_output_preact";
                op_start = clock();
                t4d *output_preact = layers[i]->forward_calc_output_preact(*(inputs[i]));
                LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);
                IF_PLOG(plog::debug) {
                    LOGD << "Layer " << i << " output_preact";
                    cout << output_preact->to_string() << endl;
                }

                LOGI << "forward_activate";
                op_start = clock();
                t4d *output = layers[i]->forward_activate(*output_preact);
                LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);
                delete output_preact;
                IF_PLOG(plog::debug) {
                    LOGD << "Layer " << i << " output";
                    cout << output->to_string() << endl;
                }
                outputs.push_back(output);
                prev_output = outputs[i];
            }                

            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";   
            
            LOGD << "Backpropagating";
            double loss;
            unique_ptr<t4d> drv_error_output_preact, drv_error_prev_output;

            for(int i = layers.size()-1; i>=0; i--) {
                PLOGI.printf("Backward Layer %d", i);
                IF_PLOG(plog::debug) {
                    PLOGD << "Output";
                    cout << outputs[i]->to_string() << endl;
                }
                if(i==(layers.size()-1)) {
                    LOGI << "backprop_calc_drv_error_output_preact(loss)";
                    op_start = clock();
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(loss_fn, loss, *(outputs[i]), *batch_labels.get()));
                    LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);
                    
                    LOGD << "Epoch loss: " << epoch_loss << " += " << loss;
                    epoch_loss += loss;
                }
                else {
                    LOGI << "backprop_calc_drv_error_output_preact";
                    op_start = clock();
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(*drv_error_prev_output.get(), *(outputs[i])));
                    LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);
                }
                delete outputs[i];

                
                IF_PLOG(plog::debug) {
                    PLOGD << "drv_error_output_preact";
                    cout << drv_error_output_preact->to_string() << endl;
                }
                
                LOGI << "backprop_update";
                op_start = clock();
                layers[i]->backprop_update(learning_rate, *drv_error_output_preact.get(), *inputs[i]);
                LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);
                
                LOGI << "backprop_calc_drv_error_prev_output";
                op_start = clock();
                drv_error_prev_output.reset(layers[i]->backprop_calc_drv_error_prev_output(*drv_error_output_preact.get(), *inputs[i]));
                LOGI << "duration: " <<  std::setprecision(15) << std::fixed << dur(op_start);

                delete inputs[i];
            }
            double iter_duration = dur(start_iter);
            LOGI << "Iteration " << iter << " loss: " << loss << ", duration: " <<  std::setprecision(15) << std::fixed << iter_duration;

            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
        }
        double epoch_duration = dur(epoch_start);
        LOGI << "Epoch [" << e << "] loss: " << epoch_loss << ", duration: " << epoch_duration << endl;
        
        // //calculate loss over validation set
        // //if it stops dropping then stop
        // //TODO split dataset in validation, test, train
        e++;
    }
    while( epoch_loss > 0.05f );
    
    duration = dur(start);
    LOGI << "Train duration: " <<  std::setprecision(15) << std::fixed << duration;

 }

void hello() {

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

