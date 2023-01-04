#pragma once
#include "tensor.hpp"
#include "ops.hpp"
#include <memory>
#include <string>
#include <iostream>
#include <sstream>

//TODO: require prev_l on ctor? eliminate init_params
//TODO: :overload Neural::Tensor4D operations? +-,
//TODO: pooling layer, dropout layer, batch normalization layer, deconvolution layer
//TODO: weight layers, non-weight layers OR weight_size = 0
//TODO: require prev layer on constructor, set weight sizes
//TODO: test batch now vs batch normal vs online (duration, accuracy)
// TODO: Weights class including biases, dimensionality
namespace Neural::Layers {

    ////////////////////////////// <Layer> //////////////////////////////////////////////////
    class Layer {        
    protected:        
        std::string layerType{""}, layerOp{""};
        Neural::Activations::Base<double> activation_fn;
        Neural::Shape4D prev_shape_proto, input_shape_proto, output_shape_proto;
        int features, id;
        bool _acc{false};

        std::string gph();

        static int nl;
    public:
        Layer() {}
        Layer(Neural::Shape4D , int, std::string);
        virtual ~Layer() = default;

        auto type() const { return layerType; }
        std::string get_activation_name() { return activation_fn.name(); }
        void set_acc(bool acc) { _acc = acc; }
        Shape4D get_output_shape_proto() { return output_shape_proto; }
        virtual void init() = 0;

        virtual Neural::Tensor4D<double> * forward_input(Neural::Tensor4D<double> &) = 0;
        virtual Neural::Tensor4D<double> * forward_output(Neural::Tensor4D<double> &) = 0;
        Neural::Tensor4D<double> * activate(Neural::Tensor4D<double> &);

        Neural::Tensor4D<double> * backprop_calc_loss(std::string, double &, Neural::Tensor4D<double> &, Neural::Tensor4D<int> &);
        Neural::Tensor4D<double> * backprop_delta_output(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);
        virtual Neural::Tensor4D<double> * backprop_delta_prev_output(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &) = 0;
        virtual Neural::Tensor4D<double> * backprop(double, Neural::Tensor4D<double> &, Neural::Tensor4D<double> &) = 0;
    };
    
    class BatchNormal : public Layer {
    };
    ////////////////////////////// </Layer> //////////////////////////////////////////////////
    
    ////////////////////////////// <Weighted> ////////////////////////////////////////////////
    class Weighted : public Layer {
    protected:
        Weighted() {}
        Weighted(Neural::Shape4D , int, std::string);
        virtual ~Weighted();
        
        Neural::Shape4D weights_shape, biases_shape;
        
        std::unique_ptr<Neural::Tensor4D<double>> weights, biases;

        void init();
        
        Neural::Tensor4D<double> * backprop(double, Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);
        virtual Neural::Tensor4D<double> * backprop_delta_weights(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &) = 0;
        Neural::Tensor4D<double> * backprop_delta_biases(Neural::Tensor4D<double> &);
    };
    
    class Fc: public Weighted {
    
    public:
        Fc(Neural::Shape4D , int, std::string);
        ~Fc();
        
        Neural::Tensor4D<double> * forward_input(Neural::Tensor4D<double> &);
        Neural::Tensor4D<double> * forward_output(Neural::Tensor4D<double> &);

        Neural::Tensor4D<double> * backprop_delta_weights(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);
        Neural::Tensor4D<double> * backprop_delta_prev_output(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);  
    };
    
    class Conv: public Weighted {
    
    private:
        std::vector<int> stride{0,0}, filter_size{0,0}, padding{0,0,0,0}, stride_bp_weights{0,0};
        int out_height, out_width;
        std::string padding_type{""};

    protected:
        Neural::Tensor4D<double> * forward_input(Neural::Tensor4D<double> &);
        Neural::Tensor4D<double> * forward_output(Neural::Tensor4D<double> &);

        Neural::Tensor4D<double> * backprop_delta_weights(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);
        Neural::Tensor4D<double> * backprop_delta_prev_output(Neural::Tensor4D<double> &, Neural::Tensor4D<double> &);  
        
        bool is_padded() { return padding[0] != 0 || padding[1] != 0 || padding[2]!=0 || padding[3]!=0; }

    public:
        Conv(Neural::Shape4D , int, std::string, std::vector<int>, std::vector<int>, std::string);
        ~Conv();
    };
       
    ////////////////////////////// </Weighted> /////////////////////////////////////////////////

}
