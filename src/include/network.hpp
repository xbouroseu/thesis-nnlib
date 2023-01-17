#pragma once
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "utils.hpp"
#include "tensor.hpp"
#include "layer.hpp"

//TODO weights is Network property?
//TODO layer::forward is variadic?
//TODO batch_size at train time? resize whole network? so init then? 33

namespace Neural {
    class Network {
    private:    
        std::vector<Neural::Layers::Layer *> layers;
        Neural::Shape4D input_shape_proto;
        bool _debug{false}, _acc{false};
        
    public:
        Network(Neural::Shape4D);
        ~Network();

        void init();
        void forward(Neural::Tensor4D<double> &, std::vector<Neural::Tensor4D<double> *> &, std::vector<Neural::Tensor4D<double> *> &);
        Neural::Tensor4D<double> *forward(Neural::Tensor4D<double> &init_input);

        template<class L, class ... Args>
        void add_layer(Args ...args) {
            LOGV << "Network::add_layer";
            Neural::Layers::Layer *newl;
            Neural::Shape4D prev_sh;          
            
            if(layers.size()==0) {
                prev_sh = input_shape_proto;
            }
            else {
                prev_sh = layers[layers.size()-1]->get_output_shape_proto();
            }
            
            newl = new L(prev_sh, args...);
            layers.push_back(newl);
        }
        
        void set_debug(bool);
        void set_acc(bool);
        void alloc();
        void forward();
        void eval(Tensor4D<double> &eval_dataset, Tensor4D<int> &eval_labels, double &recall, double &precision);
        void train(Tensor4D<double> &, Tensor4D<int> &, Tensor4D<double> &, Tensor4D<int> &, int, bool, double, std::string, int fepochs = 0, int fsteps = 0);
    };
}


void param2file_al(double *, std::string , std::string , int  );
void param2file_csv(double *, std::string , int , int , int );
