#include "network.hpp"
#include "layer.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>

//TODO solve prev_l, weight initialization, can be on ctor? can pass network last layer as first argument? or network itself?

//TODO template layers double,float?

typedef Tensor4D<double> t4d;

using namespace std;
using Neural::Network;
using Neural::Layers::Layer;

Network::Network() {}

Network::~Network() {
    cout << "Network destructor" << endl;
    for(auto it : layers) {
        delete it;
    }
}

void Network::set_acc(bool acc) {
    cout << "Network::set_acc{" << acc << endl;
    for(auto it: layers) {
        it->set_acc(acc);
    }
    _acc = acc;
}

void Network::attach_input(unique_ptr<Tensor4D<double>> * _input) {
    cout << "Network::attach_input" << endl;
    layers[0]->apply_to(_input);
    
    for(int i = 1; i < layers.size(); i++) {
        layers[i]->apply_to(layers[i-1]);
        cout << "shapes done" << endl;
    }
    cout << "/Network::attach_input" << endl;
}

void Network::alloc() {
    cout << "Network::alloc" << endl;
    for(auto it : layers) {
        it->alloc();
    }
}

void Network::forward() {
    int l = 1;
    for(auto it : layers) {
        LOG("***************************************************************** Layer " << l <<" ****************************************************************************\n\n");
        it->forward();
        LOG("\n******************************************************************************************************************************************************\n");
        l++;
    }
}

void Network::train(const LabeledData<double> &train_data, const LabeledData<double> &valid_data, int batch_size, int epochs, bool acc, double learning_rate, string loss_fn) {
    cout << "Network::train | batch_size: " << batch_size << " | epochs: " << epochs << endl;
    
    auto train_dataset = train_data.data;
    auto train_labels = train_data.labels;
    
    Shape4D train_shape = train_dataset->shape(), labels_shape = train_labels->shape();
    cout  << "Train shape: " << train_shape.to_string() << endl;
    
    int num_samples = train_shape[0], train_size = train_dataset->size();
    int num_outputs = labels_shape[1];
    
    assert(num_samples == labels_shape[0]);
    assert(batch_size <= num_samples);
    
    unique_ptr<t4d> batch_data;
    unique_ptr<Tensor4D<int>> batch_labels;
    
    /////////////////////////
    {    
        Shape4D bshape(batch_size, train_shape[1], train_shape[2], train_shape[3]);
        cout << "bshape = " << bshape.to_string() << endl;
        batch_data = make_unique<t4d>(bshape);
        if(acc) {
            batch_data->create_acc();
        }
    }
    
    {
        Shape4D lbl_bshape(batch_size, num_outputs, 1, 1);
        cout << "lbl_bshape = " << lbl_bshape.to_string() << endl;
        batch_labels = make_unique<Tensor4D<int>>(lbl_bshape);
        if(acc) {
            batch_data->create_acc();
        }
    }
    //////////////////////////
    
    this->attach_input(&batch_data);
    this->set_acc(acc);
    this->alloc();
    
    int iters = train_shape[0]/batch_size;
    int batch_start;
    double duration;
    clock_t start = clock();
    
    {
        printf("Iterations : %d\n", iters);
        for(int e = 0; e < epochs; e++) {
            double epoch_loss = 0.0f;
            
            for(int i = 0; i < iters; i++) {
                batch_start = (i*batch_size)%(num_samples-batch_size+1);
                        
                printf(">>>>>>> Step %d, batch_start: %d<<<<<<<< ",i, batch_start);
                
                acc_make_batch<double>(*train_dataset, batch_data.get(),  batch_start); 
                LOG("[batch_data]");
                batch_data->print();
                
                acc_normalize_img(batch_data.get());
                LOG("[batch_data normalized]");
                batch_data->print();           
                
                acc_make_batch<int>(*train_labels, batch_labels.get(), batch_start);
                
                
                LOG("[batch_labels]");
                batch_labels->print();
                
                LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
                
                this->forward();
                

                LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
                
                LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
                
                LOG("Calculating loss at output");
                double loss = layers[layers.size()-1]->loss(loss_fn, batch_labels.get());
                epoch_loss += loss;
                
                cout << "Loss = " << loss << endl;
                    
                LOG("Backpropagating");
                for(int i = layers.size()-1; i>=0; i--) {
                    layers[i]->backward(learning_rate);
                }
                LOG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /BACKWARD "  << i <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            }
            
            //calculate loss over validation set
            //if it stops dropping then stop
            //TODO split dataset in validation, test, train
            
        }
    }
    
    duration = dur(start);
    cout << "Train duration: " <<  std::setprecision(15) << std::fixed << duration << endl;
    cout << "Step duration: " <<  std::setprecision(15) << std::fixed << duration/iters << endl;
    clock_t now = clock();
    duration = now - start;
    cout << "Time now: " << clock() << " | start: " << start << " | duration: " << duration << " | ms: " << (duration/CLOCKS_PER_SEC) << std::setprecision(15) << std::fixed << endl;
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

