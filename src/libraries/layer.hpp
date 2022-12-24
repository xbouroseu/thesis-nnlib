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

    ////////////////////////////// <Layer> //////////////////////////////////////////////////
    class Layer {
        friend class Neural::Network;
        
    protected:
        Layer() {}
        Layer(Shape4D &, int, string);
        virtual ~Layer() = default;
        
        string layerType{""}, layerOp{""};
        actd activation_fn;
        Shape4D prev_shape_proto;
        int features, id;
        bool _acc{false};

        string gph();

        virtual t4d * forward_input(t4d &) = 0;
        virtual t4d * forward_output(t4d &) = 0;
        t4d * activate(t4d &);

        t4d * backprop_calc_loss(string, double &, t4d &, Tensor4D<int> &);
        t4d * backprop_delta_output(t4d &, t4d &);
        virtual t4d * backprop_delta_prev_output(t4d &, t4d &) = 0;
        t4d * backprop(double, t4d &, t4d &);
        virtual void _backprop(double, t4d &, t4d &);

        virtual void alloc() {}

        static int nl;
    public:
        auto type() const { return layerType; }
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
        Weighted(Shape4D &, int, string);
        virtual ~Weighted();
        
        Shape4D weights_shape, biases_shape;
        
        std::unique_ptr<t4d> weights, biases;

        void alloc() override;
        
        t4d * forward_output(t4d &);
        virtual t4d * _forward_output(t4d &) = 0;

        void _backprop(double, t4d &, t4d &);
        virtual t4d * backprop_delta_weights(t4d &, t4d &) = 0;
        t4d * backprop_delta_biases(t4d &);

        void update_weights(double , t4d *, t4d *);
    };
    
    class Fc: public Weighted {
    friend class Neural::Network;        
    
    protected:
        Fc(Shape4D &, int, string);
        ~Fc();
        
        t4d * forward_input(t4d &);
        t4d * _forward_output(t4d &);

        t4d * backprop_delta_weights(t4d &, t4d &);
        t4d * backprop_delta_prev_output(t4d &, t4d &);  
    };
    
    class Conv: public Weighted {
    friend class Neural::Network;
    
    private:
        vector<int> stride{0,0}, filter_size{0,0}, padding{0,0,0,0}, stride_bp_weights{0,0};
        int out_height, out_width;
        string padding_type{""};

    protected:
        
        Conv(Shape4D &, int, string, vector<int>, vector<int>, string);
        ~Conv();
       
        t4d * forward_input(t4d &);
        t4d * _forward_output(t4d &);

        t4d * backprop_delta_weights(t4d &, t4d &);
        t4d * backprop_delta_prev_output(t4d &, t4d &);  
        
        bool is_padded() { return padding[0] != 0 || padding[1] != 0 || padding[2]!=0 || padding[3]!=0; }
    };
    ////////////////////////////// </Weighted> /////////////////////////////////////////////////

}
