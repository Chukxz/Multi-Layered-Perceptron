#include <dlfcn.h>
#include <iostream>

int main()
{
  const char *train_image_file = "./train-images.idx3-ubyte";
  const char *train_label_file = "./train-labels.idx1-ubyte";
  unsigned
  train_len = 60000,
  layers_len = 4;
  unsigned layers_list[layers_len] = {784, 32, 32, 10};

  double learning_rate = 0.01;
  unsigned batch_size = 16;
  unsigned epoch_num = 5;
  double momentum = 0.9;
  double weight_decay = 0.8;
  double i_dropout = 0.2;
  double h_dropout = 0.5;
  bool verbose = true;

  using TrainNN1Type = void (*)
  (unsigned *layers_list, unsigned layers_len, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose);

  // Load the SO
  void* handle = dlopen("./mlp_arm64-v8a_d.so", RTLD_LAZY);
  if(!handle)
  {
    std::cerr << "Cannot open library 'mlp_arm64-v8a_d.so' : " << dlerror() << "\n";
    return 1;
  }

  // Reset errors
  dlerror();

  // Load the function
  TrainNN1Type const trainNN1Func = reinterpret_cast<TrainNN1Type>(dlsym(handle, "trainNN1"));
  const char* dlsym_error = dlerror();
  if (dlsym_error)
  {
    std::cerr << "Cannot load symbol 'trainNN1': " << dlsym_error << "\n";
    dlclose(handle);
    return 1;
  }

  // Use the function
  trainNN1Func(layers_list, layers_len, train_len, learning_rate, batch_size, epoch_num, momentum, weight_decay, i_dropout, h_dropout, train_label_file, train_image_file, verbose);

  // Free the SO
  dlclose(handle);
  return 0;
}