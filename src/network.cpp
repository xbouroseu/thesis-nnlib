#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "network.hpp"
#include "ops.hpp"


using namespace std;

using Neural::Network;
using Neural::Tensor4D;
using Neural::Shape4D;

typedef Tensor4D<double> t4d;

Neural::Network::Network(const Shape4D &in_sh_pr) : __input_shape_proto(Shape4D(-1, in_sh_pr[1], in_sh_pr[2], in_sh_pr[3])) {
    LOGD << "Network::Network";
    LOGD << "input_shape_proto: " << __input_shape_proto.to_string();
}

Network::~Network() {
    LOGD << "Network destructor";
    for(int i=0; i < layers.size(); i++) {
        delete layers[i];
    }
}

t4d * Network::forward(t4d &init_input) {
    clock_t op_start;
    string op_name;
    
    t4d *prev_output = &init_input;
    
    for(int i = 0; i < layers.size(); i++) {
        PLOGD.printf("Forward Layer %d", i);
        
        _LLOG(debug, prev_output);

        _LOGXPC(debug, "forward_calc_input",  t4d *input_i = layers[i]->forward_calc_input(*prev_output));
        _LLOG(debug, input_i);

        if(i>0) {
            delete prev_output;
        }

        _LOGXPC(debug, "forward_calc_output_preact",  unique_ptr<t4d> output_preact(layers[i]->forward_calc_output_preact(*input_i)) );
        _LLOG(debug, output_preact);

        delete input_i;

        _LOGXPC(debug, "forward_activate", t4d * output_i = layers[i]->forward_activate(*output_preact.get()));
        _LLOG(debug, output_i);
        
        prev_output = output_i;

        // IF_PLOG(plog::debug) { op_name = "forward_calc_input"; PLOGD << op_name; op_start = clock(); }
        // t4d *input_i = layers[i]->forward_calc_input(*prev_output);
        // PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);

        // IF_PLOG(plog::debug) { op_name = "forward_calc_output_preact"; PLOGD << op_name; op_start = clock(); }    
        // unique_ptr<t4d> output_preact(layers[i]->forward_calc_output_preact(*input_i));
        // PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);

        // IF_PLOG(plog::debug) { op_name = "forward_activate"; PLOGD << op_name; op_start = clock(); }
        // t4d * output_i = layers[i]->forward_activate(*output_preact.get());
        // PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);

    }

    return prev_output;
}

void Network::forward(t4d &init_input, vector<t4d *> &inputs, vector<t4d *> &outputs) {
    t4d *prev_output = &init_input;

    clock_t op_start;
    string op_name;

    for(int i = 0; i < layers.size(); i++) {
        PLOGD.printf("Forward Layer %d", i);
        
        _LLOG(debug, prev_output);

        IF_PLOG(plog::debug) { op_name = "forward_calc_input"; PLOGD << op_name; op_start = clock(); }
        inputs.push_back(layers[i]->forward_calc_input(*prev_output));
        PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
        _LLOG(debug, inputs[i]);

        IF_PLOG(plog::debug) { op_name = "forward_calc_output_preact"; PLOGD << op_name; op_start = clock(); }    
        unique_ptr<t4d> output_preact(layers[i]->forward_calc_output_preact(*(inputs[i])));
        PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
        _LLOG(debug, output_preact);
        
        IF_PLOG(plog::debug) { op_name = "forward_activate"; PLOGD << op_name; op_start = clock(); }
        outputs.push_back(layers[i]->forward_activate(*output_preact.get()));
        PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
        _LLOG(debug, outputs[i]);
        
        prev_output = outputs[i];
    }
}

void Network::init() {
    PLOGI << "Network::init";
    int lnn = 0;

    for(auto it: layers) {
        PLOGD << "Layer " << ++lnn << " init";
        it->init();
    }
}

void Network::eval(const Tensor4D<double> &eval_dataset, const Tensor4D<int> &eval_labels, double &recall, double &precision, double &accuracy, double &f1_score) {
    Shape4D eval_data_shape = eval_dataset.shape(), eval_labels_shape = eval_labels.shape();

    vector<Tensor4D<int> *> confusion_matrices;
    int eval_batch_size = eval_data_shape[0]/100;
    int iters_eval = eval_data_shape[0]/eval_batch_size;

    LOGI.printf("eval_batch_size: %d, iters_eval: %d", eval_batch_size, iters_eval);
    for(int v=0; v < iters_eval; v++) {
        LOGI_IF((v%10)==0) << v;
        int eval_batch_start = (v*eval_batch_size)%(eval_data_shape[0]-eval_batch_size+1);

        unique_ptr<t4d> eval_batch_data = make_unique<t4d>(eval_batch_size, eval_data_shape[1], eval_data_shape[2], eval_data_shape[3]);
        eval_batch_data->create_acc();

        unique_ptr<Tensor4D<int>> eval_batch_labels = make_unique<Tensor4D<int>>(eval_batch_size, eval_labels_shape[1], eval_labels_shape[2], eval_labels_shape[3]);
        eval_batch_labels->create_acc();

        acc_make_batch(eval_dataset, eval_batch_data.get(), eval_batch_start);
        acc_normalize_img(eval_batch_data.get());
        acc_make_batch<int>(eval_labels, eval_batch_labels.get(), eval_batch_start);
        
        t4d *eval_batch_output = this->forward(*eval_batch_data.get());
        Tensor4D<int> *batch_conf_matrix = acc_calc_confusion_matrix(*eval_batch_output, *eval_batch_labels.get());
        confusion_matrices.push_back(batch_conf_matrix);
        delete eval_batch_output;
    }

    Tensor4D<int> *confusion_matrix_final = confusion_matrices[0];
    for(int i = 1; i < confusion_matrices.size(); i++) {
        acc_add(confusion_matrix_final, *confusion_matrices[i]);
    }

    _LLOG(info, confusion_matrix_final);

    LOGI << "Calculating precision/recall per class";
    vector<Tensor4D<double> *> precision_recall_class = calc_metrics(*confusion_matrix_final);
    Tensor4D<double> * precision_class = precision_recall_class[0];
    Tensor4D<double> * recall_class = precision_recall_class[1];
    Tensor4D<double> * accuracy_class = precision_recall_class[2];
    Tensor4D<double> * f1_class = precision_recall_class[3];

    _LLOG(info, precision_recall_class[0]);
    _LLOG(info, precision_recall_class[1]);
    _LLOG(info, precision_recall_class[2]);
    _LLOG(info, precision_recall_class[3]);

    LOGI << "Calculating [precision, recall, accuracy, f1_score] macro average over classes";

    precision = 0.0f;
    recall = 0.0f;
    accuracy = 0.0f;
    f1_score = 0.0f;

    for(int m = 0; m < precision_class->size(); m++) {
        precision += precision_class->iat(m);
        recall += recall_class->iat(m);
        accuracy += accuracy_class->iat(m);
        f1_score += f1_class->iat(m);
    }
    precision /= precision_class->size();
    recall /= precision_class->size();
    accuracy /= accuracy_class->size();
    f1_score /= f1_class->size();

    delete confusion_matrix_final;
}
void Network::train(const Tensor4D<double> &train_dataset, const Tensor4D<int> &train_labels, const Tensor4D<double> &valid_dataset, const Tensor4D<int> &valid_labels,  int batch_size, bool acc, double learning_rate, string loss_fn, int fepoch, int fsteps) {
    PLOGI << "Network::train | batch_size: " << batch_size;

    Shape4D train_shape = train_dataset.shape(), train_labels_shape = train_labels.shape(), valid_shape = valid_dataset.shape(), valid_labels_shape = valid_labels.shape();

    assert(train_shape[0] == train_labels_shape[0]);
    assert(batch_size <= train_shape[0]);
    assert_shape(train_shape, valid_shape);
    assert_shape(train_labels_shape, valid_labels_shape);
    assert_shape(train_shape, __input_shape_proto);

    this->init();
    
    int iters = train_shape[0]/batch_size, batch_start;
    
    PLOGI.printf("Steps per epoch: %d", iters);
    int e = 0;
    vector<double> vec_epoch_recall, vec_epoch_precision, vec_epoch_accuracy, vec_epoch_f1;
    clock_t train_start = clock();

    do {
        double epoch_loss = 0.0f, precision_epoch_macro = 0.0f, recall_epoch_macro = 0.0f, accuracy_epoch_macro = 0.0f, f1_epoch_macro = 0.0f;
        clock_t epoch_start = clock();
        int iter=0;

        do {
            clock_t iter_start = clock();

            batch_start = (iter*batch_size)%(train_shape[0]-batch_size+1);
            
            LOGD << "-------------------------------------------------------------------------------------------------------------------------------------";
            
            IF_PLOG(plog::debug) {

                printf("Step %d, batch_start: %d, batch_size: %d | ",iter, batch_start, batch_size);
            }

            unique_ptr<t4d> batch_data = make_unique<t4d>(batch_size, train_shape[1], train_shape[2], train_shape[3]);
            batch_data->create_acc();
            
            unique_ptr<Tensor4D<int>> batch_labels = make_unique<Tensor4D<int>>(batch_size, train_labels_shape[1], train_labels_shape[2], train_labels_shape[3]);
            batch_labels->create_acc(); 
    
            clock_t op_start;
            string op_name;
            
            IF_PLOG(plog::debug) { op_name = "acc_make_batch"; PLOGD << op_name; op_start = clock(); }
            acc_make_batch<double>(train_dataset, batch_data.get(),  batch_start);
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG(debug, batch_data);            

            IF_PLOG(plog::debug) { op_name = "acc_normalize_img"; PLOGD << op_name; op_start = clock(); }
            acc_normalize_img(batch_data.get());
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG_A(debug, batch_data, "batch_data_normalized")

            IF_PLOG(plog::debug) { op_name = "acc_make_batch[labels]"; PLOGD << op_name; op_start = clock(); }
            acc_make_batch<int>(train_labels, batch_labels.get(), batch_start);
            PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
            _LLOG(debug, batch_labels);

            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            vector<t4d *> inputs, outputs;
            this->forward(*batch_data.get(), inputs, outputs);

            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /FORWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            
            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            double loss;
            unique_ptr<t4d> drv_error_output_preact, drv_error_prev_output;

            for(int i = layers.size()-1; i>=0; i--) {
                PLOGD.printf("Backward Layer %d", i);
                
                _LLOG(debug, outputs[i]);

                if(i==(layers.size()-1)) {
                    IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_output_preact(loss)"; PLOGD << op_name; op_start = clock(); }    
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(loss_fn, loss, *(outputs[i]), *batch_labels.get()));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                    
                    if(iter == 500) {
                        acc_calc_confusion_matrix(*(outputs[i]), *batch_labels.get());
                    }
                    PLOGD << "Epoch loss: " << epoch_loss << " += " << loss;
                    epoch_loss += loss;
                    
                }
                else {
                    IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_output_preact"; PLOGD << op_name; op_start = clock(); }    
                    drv_error_output_preact.reset(layers[i]->backprop_calc_drv_error_output_preact(*drv_error_prev_output.get(), *(outputs[i])));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                }

                PLOGD.printf("delete outputs[%d]", i);
                delete outputs[i];

                _LLOG(debug, drv_error_output_preact);

                if(i!=0) {
                    IF_PLOG(plog::debug) { op_name = "backprop_calc_drv_error_prev_output"; PLOGD << op_name; op_start = clock(); }   
                    drv_error_prev_output.reset(layers[i]->backprop_calc_drv_error_prev_output(*drv_error_output_preact.get(), *inputs[i]));
                    PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);
                }

                IF_PLOG(plog::debug) { op_name = "backprop_update"; PLOGD << op_name; op_start = clock(); }    
                layers[i]->backprop_update(learning_rate, *drv_error_output_preact.get(), *inputs[i]);
                PLOGD << "Execution time: " << op_name << " = " <<  std::setprecision(15) << std::fixed << dur(op_start);

                PLOGD.printf("delete inputs[%d]", i);
                delete inputs[i];
            }

            PLOGD << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< /BACKWARD " << iter <<" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";

            PLOGI_IF((iter%100)==0).printf("[Epoch: %d] Step %d | batch_start:%d | step_loss: %11.6f | epoch_loss: %11.6f | duration: %20.15f", e, iter, batch_start, loss, epoch_loss, dur(iter_start));
            iter++;
        }
        while((iter < iters) && ( (fsteps==0) || (iter<fsteps) ) );
        
        //TODO overload operator+ Tensor?
        //TODO make ops return?
        //TODO chain create_acc etc?
        LOGW << "Calculating metrics for valid_dataset";
        LOGW << "this->eval(valid_dataset, valid_labels, recall_epoch_macro, recall_epoch_macro)";
        this->eval(valid_dataset, valid_labels, recall_epoch_macro, recall_epoch_macro, accuracy_epoch_macro, f1_epoch_macro);
        vec_epoch_recall.push_back(recall_epoch_macro);
        vec_epoch_precision.push_back(precision_epoch_macro);
        vec_epoch_accuracy.push_back(accuracy_epoch_macro);
        vec_epoch_f1.push_back(f1_epoch_macro);

        PLOGI << "[Epoch " << e << "] epoch_loss: " << epoch_loss << " | precision_avg: " << precision_epoch_macro << " | recall_avg: " << recall_epoch_macro << " | accuracy_avg: " << accuracy_epoch_macro << " | f1_avg: " << f1_epoch_macro << " | duration: " << dur(epoch_start);
        e++;
        
    }
    while( (e>0 && ( (vec_epoch_f1[e-1]-vec_epoch_f1[e-2]) >= 0.0005 ) ) && ( (fepoch==0) || (e < fepoch)) );
    
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

