#pragma once
#include "neural.hpp"
#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

using Neural::Layers::Layer;

//TODO weights is Network property?
//TODO layer::forward is variadic?
//TODO batch_size at train time? resize whole network? so init then?

namespace Neural {
    class Network {
    private:    
        std::vector<Layer *> layers;
        
        bool _debug{false}, _acc{false};
        
    public:
        Network();
        ~Network();
                
        template<class L, class ... Args>
        void add_layer(Args ...args) {
            LOG("Network::add_layer");
            L *newl = new L(args...);
                
            layers.push_back(newl);
        }
        
        void attach_input(std::unique_ptr<Tensor4D<double>> *);
        void set_debug(bool);
        void set_acc(bool);
        void alloc();
        void forward();
        void train(const LabeledData<double> &, const LabeledData<double> &, int, int, bool, double, std::string);
    };
}


void param2file_al(double *, std::string , std::string , int  );
void param2file_csv(double *, std::string , int , int , int );
