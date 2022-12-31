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
    LOG("Network::Network");
    LOG("input_shape_proto: " << input_shape_proto.to_string());
}

Network::~Network() {
    cout << "Network destructor" << endl;
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

void Network::train(const LabeledData<double> &train_data, const LabeledData<double> &valid_data, int batch_size, bool acc, double learning_rate, string loss_fn) {
    cout << "Network::train | batch_size: " << batch_size << endl;
    
    Tensor4D<double> * train_dataset = train_data.data, *valid_dataset = valid_data.data;
    Tensor4D<int> * train_labels = train_data.labels, *valid_labels = valid_data.labels;
    
    Shape4D train_shape = train_dataset->shape(), labels_shape = train_labels->shape();
    cout  << "Train shape: " << train_shape.to_string() << endl;
    
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
        cout << "batch_data_shape = " << batch_data_shape.to_string() << endl;
        batch_data = make_unique<t4d>(batch_data_shape);
        batch_data->create_acc();
    }
    
    {
        Shape4D batch_labels_shape(batch_size, labels_shape[1], labels_shape[2], labels_shape[3]);
        cout << "batch_labels_shape = " << batch_labels_shape.to_string() << endl;
        batch_labels = make_unique<Tensor4D<int>>(batch_labels_shape);
        batch_data->create_acc();
    }
    //////////////////////////
    
    cout << "Calling Layer::init" << endl;
    for(auto it: layers) {
        it->init();
    }
    int iters = train_num_samples/batch_size;
    int batch_start;
    double duration;
    clock_t start = clock();
    
    printf("Epoch iterations : %d\n", iters);
    int e = 1;
    double epoch_loss;
    do {
        epoch_loss = 0.0f;
        cout << "Epoch: " << e;
        // for(int iter = 0; iter < iters; iter++) {
        //     batch_start = (iter*batch_size)%(train_num_samples-batch_size+1);
                    
        //     // printf(">> Step %d, batch_start: %d<< \n",iter, batch_start);
            
        //     acc_make_batch<double>(*train_dataset, batch_data.get(),  batch_start); 
        //     LOG("[batch_data]");
        //     XLOG(batch_data->print());
            
        //     acc_normalize_img(batch_data.get());
        //     LOG("[batch_data normalized]");
        //     XLOG(batch_data->print());           
            
        //     acc_make_batch<int>(*train_labels, batch_labels.get(), batch_start);
            
        //     LOG("[batch_labels]");
        //     XLOG(batch_labels->print());
        //     LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            
        //     vector<t4d *> inputs, outputs;
        //     t4d *prev_output = batch_data.get();

        //     LOG("for(int i = 0; i < layers.size(); i++)");
        //     for(int i = 0; i < layers.size(); i++) {
        //         inputs.push_back(layers[i]->forward_input(*prev_output));
        //         t4d *output_preact = layers[i]->forward_output(*(inputs[i]));
        //         t4d *output = layers[i]->activate(*output_preact);
        //         outputs.push_back(output);
        //         delete output_preact;
        //         prev_output = outputs[i];
        //     }                

        //     LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            
        //     LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            
        //     LOG("Calculating loss at output");
        //     double loss;
        //     XLOG(outputs.back()->print());
        //     unique_ptr<t4d> output_preact_loss(layers.back()->backprop_calc_loss(loss_fn, loss, *outputs.back(), *batch_labels.get()));
        //     delete outputs.back();
        //     epoch_loss += loss;

        //     unique_ptr<t4d> prev_output_loss(layers.back()->backprop(learning_rate, *output_preact_loss.get(), *inputs.back()));
        //     delete inputs.back();

        //     LOG("Backpropagating");
        //     int j = 0;

        //     for(int i = layers.size()-2; i>=0; i--) {
        //         output_preact_loss.reset(layers[i]->backprop_delta_output(*prev_output_loss.get(), *(outputs[i])));
        //         delete outputs[i];
        //         prev_output_loss.reset(layers[i]->backprop(learning_rate, *output_preact_loss.get(), *inputs[i]));
        //         delete inputs[i];
        //     }

        //     LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /BACKWARD "  << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        // }
        cout << " loss: " << epoch_loss << endl;
        
        // //calculate loss over validation set
        // //if it stops dropping then stop
        // //TODO split dataset in validation, test, train
        e++;
    }
    while( epoch_loss > 0.05f );
    
    duration = dur(start);
    cout << "Train duration: " <<  std::setprecision(15) << std::fixed << duration << endl;
    cout << "Epoch duration: " <<  std::setprecision(15) << std::fixed << duration/e << endl;
    cout << "Step duration: " <<  std::setprecision(15) << std::fixed << duration/(iters * e) << endl;
    clock_t now = clock();
    duration = now - start;
    cout << "Time now: " << clock() << " | start: " << start << " | duration: " << duration << " | ms: " << (duration/CLOCKS_PER_SEC) << std::setprecision(15) << std::fixed << endl;
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

