#pragma once
#include "neural.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
//TODO: require prev_l on ctor? eliminate init_params
//TODO: :overload Tensor4D operations? +-,
//TODO: pooling layer, dropout layer, batch normalization layer, deconvolution layer
//TODO: weight layers, non-weight layers OR weight_size = 0
//TODO: require prev layer on constructor, set weight sizes
//TODO: test batch now vs batch normal vs online (duration, accuracy)
// TODO: Weights class including biases, dimensionality


typedef Tensor4D<double> t4d;
typedef Neural::Activations::Base<double> actd;
using std::string;
using std::vector;

namespace Neural::Layers {

    ////////////////////////////// <Layer> /////////////////////////////////////////////////
    class Layer {
        friend class Neural::Network;
        
    protected:
        Layer() {}
        Layer(int, string);
        virtual ~Layer() = default;
        
        string layerType{""}, layerOp{""};
        
        actd activation_fn;
        
        int features, id;
        
        std::unique_ptr<t4d> input, output, output_preact, drv_error_input, drv_error_output_preact, drv_error_output;
        std::unique_ptr<t4d> *prev_drv_error_output{nullptr};

        bool _acc{false};
                
        Shape4D input_shape, output_shape;
        
        string gph();
        // void log(string);
        // void log_gph(string);
        // void log_gph();
        
        void apply_to(Layer *);
        void apply_to(std::unique_ptr<t4d> *);
        double loss(string, Tensor4D<int> *);
        
        virtual void init_shape() = 0;
        virtual void alloc() {}
        
        void forward();
        virtual void _forward() = 0;
       
        void backward(double);
        
        virtual void _backward_input() = 0;
        
        virtual void _backward(double) = 0;
        
        static int nl;
    public:
        auto type() const { return layerType; }
        auto get_shape() const { return output_shape; };
        auto & get_output() { return output; };
        string get_activation_name() { return activation_fn.name(); }
        
        void set_acc(bool acc) { _acc = acc; }
    };
    
    class BatchNormal : public Layer {
    };
    ////////////////////////////// </Layer> //////////////////////////////////////////////////
    
    
    ////////////////////////////// <Weighted> ////////////////////////////////////////////////
    class Weighted : public Layer {
    friend class Neural::Network;
    protected:
        Weighted() {}
        Weighted(int, string);
        virtual ~Weighted();
        
        Shape4D weights_shape, biases_shape;
        
        std::unique_ptr<t4d> weights, biases, drv_error_weights, drv_error_biases;

        void alloc() override;
        
        void _forward();
        void _backward(double);
                
        virtual void _forward_weights() = 0;
        virtual void _backward_weights() = 0;
    };
    
    class Fc: public Weighted {
    friend class Neural::Network;        
    
    protected:
        Fc(int, string);
        ~Fc();
        
        void init_shape();
        
        void _forward_weights();
        void _backward_weights();
        void _backward_input();  
    };
    
    class Conv: public Weighted {
    friend class Neural::Network;
    
    private:
        vector<int> stride{0,0}, filter_size{0,0}, padding{0,0,0,0}, stride_bp_weights{0,0};
        
        string padding_type{""};

    protected:
        
        Conv(int, string, vector<int>, vector<int>, string);
        Conv(int, string, int, vector<int>, string);
        Conv(int, string, vector<int>, int, string);
        Conv(int, string, int, int, string);
        ~Conv();
       
        void init_shape();
        
        void _forward_weights();
        void _backward_weights();
        void _backward_input();
        
        bool is_padded() { return padding[0] != 0 || padding[1] != 0 || padding[2]!=0 || padding[3]!=0; }
    };
    ////////////////////////////// </Weighted> /////////////////////////////////////////////////

}
