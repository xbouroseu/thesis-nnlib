template <typename T> 
constexpr auto type_name() {
    std::string_view name, prefix, suffix;
    
    #ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
    #elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
    #elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
    #endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    
    return name;
}

Tensor4D<double> * read_sample_data(string data_path) {
    ifstream file_data(data_path);

    if(file_data.is_open()) {
        int magic_n, ns, sh, sw;

        file_data >> magic_n >> ns >> sh >> sw;

        if(magic_n != 3344) {
            throw runtime_error("Invalid SAMPLE data file!");
        }
        
        LOGI.printf("Magic number: %d, num_samples: %d, sample_height: %d, sample_width: %d\n", magic_n, ns, sh, sw);

        Tensor4D<double> * _dataset = new Tensor4D<double>(ns, 1, sh, sw);

        for(int i = 0; i < _dataset->size(); i++) {
            file_data >> _dataset->iat(i);
        }

        file_data.close();

        return _dataset;
    }
    else {
        throw runtime_error("Unable to open file " + data_path + "!");
    }

}

Tensor4D<int> * read_sample_labels(string data_path) {
    ifstream file_data(data_path);

    if(file_data.is_open()) {
        int magic_n, ns;

        file_data >> magic_n >> ns;

        if(magic_n != 3345) {
            throw runtime_error("Invalid SAMPLE labels file!");
        }

        LOGI.printf("Magic number: %d, num_labels\n", magic_n, ns);

        Tensor4D<int> * _dataset = new Tensor4D<int>(ns, 10, 1, 1);

        for(int i = 0; i < ns; i++) {
            int lblfull;
            file_data >> lblfull;
            for(int m = 0; m < 10; m++) {
                int lbl1hot=0;
                if(m == lblfull) {
                    lbl1hot = 1;
                }
                _dataset->iat(i*10 + m) = lbl1hot;
            }
        }

        file_data.close();

        return _dataset;
    }
    else {
        throw runtime_error("Unable to open file " + data_path + "!");
    }
}

int main() {
    printf("Current working dir: %s\n", get_current_dir_name());
    cout << "__FILE__" << __FILE__ << endl;
    cout << "Neural::is_acc " << Neural::is_acc << endl;
    cout << "Neural::get_device_type() " << Neural::get_device_type() << endl;

    Tensor4D<double> *train_data, *valid_data, *test_data;
    Tensor4D<int> *train_labels, *valid_labels, *test_labels;

    vector<int> filter_size_conv1, filter_size_conv2, stride_conv1, stride_conv2;
    int depth_conv1, depth_conv2, num_hidden_nodes, num_outputs;
    string padding_conv1, padding_conv2;
    
    train_data = read_sample_data("data/sample_data.txt");
    train_labels = read_sample_labels("data/sample_labels.txt");
    valid_data = train_data;
    valid_labels = train_labels;
    
    test_data = train_data;
    test_labels = train_labels;

    LOGI << "Train_data->to_string()";
    IF_PLOG(plog::info) {
        cout << train_data->to_string() << endl;
    }
    
    padding_conv1 = "valid";
    padding_conv2 = "valid";
    filter_size_conv1 = {3, 3};
    filter_size_conv2 = {2,2};
    stride_conv1 = {1,1};
    stride_conv2 = {1,1};
    depth_conv1 = 2;
    depth_conv2 = 2;
    num_hidden_nodes = 5;
    num_outputs = 10;

    
}   