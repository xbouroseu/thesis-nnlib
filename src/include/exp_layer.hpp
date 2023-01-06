
namespace nLayer {
    namespace Type {
        struct input;
        struct output;
        struct hidden;
    };
    
    namespace Operation {
        struct conv;
        struct fc;
        struct pooling;
        struct dropout;
        struct identity;
        struct None;
    };
    
    namespace Activation {
        struct sigmoid;
        struct relu;
        struct softmax;
        struct identity;
        struct None;
    };
    
    namespace Types {
        const int input = 1;
        const int hidden = 2;
        const int output = 3;
    };

    namespace Operations {
        const int conv = 2;
        const int fc = 1;
        const int pooling = 3;
    };

    namespace Activations {
        const int sigmoid = 1;
        const int relu = 2;
        const int softmax = 3;
        const int None = 4;
    };
};

template<class T> struct op_group : std::integral_constant<int, 0> {};
template<> struct op_group<nLayer::Operation::pooling> : std::integral_constant<int, 1> {};
template<> struct op_group<nLayer::Operation::identity> : std::integral_constant<int, 1> {};
template<> struct op_group<nLayer::Operation::fc> : std::integral_constant<int, 2> {};
template<> struct op_group<nLayer::Operation::conv> : std::integral_constant<int, 2> {};

template<class T> struct op_group_valid : std::integral_constant<int, 0> {};
template<> struct op_group_valid<nLayer::Operation::fc> : std::integral_constant<int, 1> {};
template<> struct op_group_valid<nLayer::Operation::conv> : std::integral_constant<int, 1> {};
template<> struct op_group_valid<nLayer::Operation::pooling> : std::integral_constant<int, 1> {};
template<> struct op_group_valid<nLayer::Operation::identity> : std::integral_constant<int, 1> {};

template<class A> struct act_group : std::integral_constant<int, 0> {};
template<> struct act_group<nLayer::Activation::sigmoid> : std::integral_constant<int, 1> {};
template<> struct act_group<nLayer::Activation::relu> : std::integral_constant<int, 1> {};
template<> struct act_group<nLayer::Activation::identity> : std::integral_constant<int, 1> {};
template<> struct act_group<nLayer::Activation::softmax> : std::integral_constant<int, 2> {};

template<class T, class O, class A, int GroupO = op_group_valid<O>::value, int GroupA = act_group<A>::value> struct type_group : std::integral_constant<int, 0> {};
template<>
struct type_group<nLayer::Type::input, nLayer::Operation::None, nLayer::Activation::None, 0, 0> : std::integral_constant<int, 1> {};

template<class O, class A>
struct type_group<nLayer::Type::hidden, O, A, 1, 1> : std::integral_constant<int, 2> {};

template<class A>
struct type_group<nLayer::Type::output, nLayer::Operation::fc, A, 1, 2> : std::integral_constant<int, 3> {};

class BaseLay {
public:
    void set_debug(bool dbg) { debug = dbg; }
    Shape4D get_shape() const { return shape; };
    
protected:
    BaseLay();
    BaseLay(int);
    ~BaseLay();
    std::shared_ptr<Tensor4D<double>> op_input{nullptr};
    
    int features, id;
    bool debug{false};
    Shape4D shape;
        
    static int nl;
    static int genId();
};

template<class O, int GroupO = op_group<O>::value>
class BaseOpLay : public BaseLay {
    
};

template<class O>
class BaseOpLay<O, 1> : public BaseLay {
protected:
    std::shared_ptr<Tensor4D<double>> op_output{nullptr};
    void alloc_op(); //constexpr?
};

template<class O>
class BaseOpLay<O, 2> : public BaseOpLay<nLayer::Operation::identity> {
protected:
    std::shared_ptr<Tensor4D<double>> op_weights{nullptr}, op_biases{nullptr};
    Shape4D weights_shape;
};

template<class O>
class OpLay : public BaseOpLay<O> {
};

template<>
class OpLay<nLayer::Operation::fc> : public BaseOpLay<nLayer::Operation::fc> {
protected:
    void forward();
};

template<>
class OpLay<nLayer::Operation::conv> : public BaseOpLay<nLayer::Operation::conv> {
protected:
    void forward();
};

template<>
class OpLay<nLayer::Operation::pooling> : public BaseOpLay<nLayer::Operation::pooling> {
protected:
    void forward();
};

template<>
class OpLay<nLayer::Operation::identity> : public BaseOpLay<nLayer::Operation::identity> {
protected:
    void forward();
};

template<class O, class A, int GroupO = op_group_valid<O>::value>
class ActLay : public OpLay<O> {
};

template<class O, class A>
class ActLay<O, A, 1> : public OpLay<O> {
protected:
    std::shared_ptr<Tensor4D<double>> act_output{nullptr}, act_output_drv{nullptr};
    
    void alloc_act();
    
    void activate(); //constexpr
    
    void alloc(); // constexpr layer
    void accel();// constexpr
    void dealloc();// constexpr
    void deaccel();// constexpr
};


template<class T, class O, class A, int GroupO = op_group_valid<O>::value, int GroupA = act_group<A>::value>
class Lay {
};

// input layer
template<class O, class A>
class Lay<nLayer::Type::input, O, A, 0, 0> : public ActLay<nLayer::Operation::None, nLayer::Operation::None> {
    BaseLay *next_l{nullptr};
    void init(std::shared_ptr<Tensor4D<double>>);
};

class nInputLayer: public Lay<nLayer::Type::input, nLayer::Operation::None, nLayer::Activation::None> {
};

//hidden layer
template<class O, class A>
class Lay<nLayer::Type::hidden, O, A, 1, 1> : public ActLay<O, A> {
    BaseLay *next_l{nullptr}, *prev_l{nullptr};
    void init(BaseLay *);
};

template<class O, class A>
class nHiddenLayer : public Lay<nLayer::Type::hidden, O, A> {
};

//output layer
template<class A>
class Lay<nLayer::Type::output, nLayer::Operation::fc, A, 1, 2> : public ActLay<nLayer::Operation::fc, A> {
    BaseLay *prev_l{nullptr};
    void init(BaseLay *);
};

template<class A>
class nOutputLayer : public Lay<nLayer::Type::output, nLayer::Operation::fc, A> {
};

////////////////////////////
/*
class BaseTypeLayer {
friend class Network;
friend class InnerLayer;
friend class Layer_FC;
friend class ConvLayer;
friend class InputLayer;

public:
    void set_debug(bool dbg) { debug = dbg; }
    Shape4D get_shape() const { return shape; };
    
protected:
    BaseTypeLayer();
    BaseTypeLayer(int);
    ~BaseTypeLayer();
    std::shared_ptr<Tensor4D<double>> prev_output{nullptr};
    
    int features, id;
    bool debug{false};
    Shape4D shape;
    
    static int nl;
    static int genId();
};

template<int layer_type>
class TypeLayer : public BaseTypeLayer {
};

template<>
class TypeLayer<nLayer::Types::input>: public BaseTypeLayer {
protected:
    BaseTypeLayer *next_l{nullptr};
    void init(std::shared_ptr<Tensor4D<double>>);
};

template<>
class TypeLayer<nLayer::Types::output>: public BaseTypeLayer {
protected:
    BaseTypeLayer *prev_l{nullptr};
    void init(BaseTypeLayer *);
};

template<>
class TypeLayer<nLayer::Types::hidden>: public BaseTypeLayer {
protected:
    BaseTypeLayer *next_l{nullptr}, *prev_l{nullptr};
    void init(BaseTypeLayer *);
};

template<int layer_type, int layer_op>
class OpTypeLayer: public TypeLayer<layer_type> {
protected:
    std::shared_ptr<Tensor4D<double>> op_input{nullptr};
    std::shared_ptr<Tensor4D<double>> weights{nullptr}, biases{nullptr};
    Shape4D weights_shape;
    
    void forward(); //constexpr with Tensors is (prev_output, op_input, weights, biases)
};

template<int layer_type>
class OpTypeLayer<layer_type, nLayer::Operations::pooling> : public TypeLayer<layer_type> {
protected:
    std::shared_ptr<Tensor4D<double>> op_input{nullptr};
    void forward(); //no weights, biases
};

template<int layer_type, int layer_op, int activation_type>
class ActOpTypeLayer : public OpTypeLayer<layer_type, layer_op> {
protected:
    std::shared_ptr<Tensor4D<double>> op_output{nullptr}, op_output_drv{nullptr};
    
    void alloc(); // constexpr layer
    void accel();// constexpr
    void dealloc();// constexpr
    void deaccel();// constexpr
    void activate(); //constexpr
};

//

template<int layer_op, int activation_type>
class HiddenLayer : public ActOpTypeLayer<nLayer::Types::hidden, layer_op, activation_type> {
protected:
};

class InLayer : public TypeLayer<nLayer::Types::input>  {
public:
    InLayer();
    ~InLayer();
};

class OutLayer : public ActOpTypeLayer<nLayer::Types::output, nLayer::Operations::fc, nLayer::Activations::softmax> {
public:
    OutLayer();
    ~OutLayer();
protected:
};
*/
