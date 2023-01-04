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
    LOGV << "input_shape_proto: " << input_shape_proto.to_string();
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
    LOGV  << "Train shape: " << train_shape.to_string();
    
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
        LOGV << "batch_data_shape = " << batch_data_shape.to_string();
        batch_data = make_unique<t4d>(batch_data_shape);
        batch_data->create_acc();
    }
    
    {
        Shape4D batch_labels_shape(batch_size, labels_shape[1], labels_shape[2], labels_shape[3]);
        LOGV << "batch_labels_shape = " << batch_labels_shape.to_string();
        batch_labels = make_unique<Tensor4D<int>>(batch_labels_shape);
        batch_data->create_acc();
    }
    //////////////////////////
    
    LOGV << "Calling Layer::init";
    for(auto it: layers) {
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
        for(int iter = 0; iter < iters; iter++) {
            batch_start = (iter*batch_size)%(train_num_samples-batch_size+1);

            string fors; 
            sprintf(&fors[0], ">> Step %d, batch_start: %d<< \n",iter, batch_start);
            LOGD << fors;

            acc_make_batch<double>(*train_dataset, batch_data.get(),  batch_start); 
            LOGD << "[batch_data]";
            LOGD << (batch_data->to_string());
            
            acc_normalize_img(batch_data.get());
            LOGD << "[batch_data normalized]";
            LOGD << batch_data->to_string();           
            
            acc_make_batch<int>(*train_labels, batch_labels.get(), batch_start);
            
            LOGD << "[batch_labels]";
            LOGD << batch_labels->to_string();
            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            vector<t4d *> inputs, outputs;
            t4d *prev_output = batch_data.get();

            LOGV << "for(int i = 0; i < layers.size(); i++)";
            for(int i = 0; i < layers.size(); i++) {
                inputs.push_back(layers[i]->forward_input(*prev_output));
                t4d *output_preact = layers[i]->forward_output(*(inputs[i]));
                t4d *output = layers[i]->activate(*output_preact);
                outputs.push_back(output);
                delete output_preact;
                prev_output = outputs[i];
            }                

            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";   
            
            LOGI << "Calculating loss at output";
            double loss;
            LOGD << outputs.back()->to_string();
            unique_ptr<t4d> output_preact_loss(layers.back()->backprop_calc_loss(loss_fn, loss, *outputs.back(), *batch_labels.get()));
            delete outputs.back();
            epoch_loss += loss;

            unique_ptr<t4d> prev_output_loss(layers.back()->backprop(learning_rate, *output_preact_loss.get(), *inputs.back()));
            delete inputs.back();

            LOGD << "Backpropagating";
            int j = 0;

            for(int i = layers.size()-2; i>=0; i--) {
                output_preact_loss.reset(layers[i]->backprop_delta_output(*prev_output_loss.get(), *(outputs[i])));
                delete outputs[i];
                prev_output_loss.reset(layers[i]->backprop(learning_rate, *output_preact_loss.get(), *inputs[i]));
                delete inputs[i];
            }

            LOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
        }
        LOGI << "Epoch [" << e << "] loss: " << epoch_loss << endl;
        
        // //calculate loss over validation set
        // //if it stops dropping then stop
        // //TODO split dataset in validation, test, train
        e++;
    }
    while( epoch_loss > 0.05f );
    
    duration = dur(start);
    LOGI << "Train duration: " <<  std::setprecision(15) << std::fixed << duration;
    LOGI << "Epoch duration: " <<  std::setprecision(15) << std::fixed << duration/e;
    LOGI << "Step duration: " <<  std::setprecision(15) << std::fixed << duration/(iters * e);
    clock_t now = clock();
    duration = now - start;
    LOGI << "Time now: " << clock() << " | start: " << start << " | duration: " << duration << " | ms: " << (duration/CLOCKS_PER_SEC) << std::setprecision(15) << std::fixed;
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

