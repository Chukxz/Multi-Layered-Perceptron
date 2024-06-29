#include <Windows.h>
#include <vector>
#include <iostream>

int main()
{
  const char *train_image_file = "./train-images.idx3-ubyte";
  const char *train_label_file = "./train-labels.idx1-ubyte";
  unsigned
  train_len = 60000,
  layers_len = 4;
  unsigned layers_list[4] = {784, 32, 32, 10};

  double learning_rate = 0.001;
  unsigned batch_size = 16;
  unsigned epoch_num = 5;
  double momentum = 0.7;
  double weight_decay = 0.02;
  double i_dropout = 0.2;
  double h_dropout = 0.5;
  bool verbose = true;

  using TrainNN1Type = void (__cdecl*)
  (unsigned *layers_list, unsigned layers_len, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose);

  // Load the DLL
  HMODULE const hinstLib = LoadLibraryExW(L"./mlp_x86-64_avx_d.dll", nullptr, 0);
  if(!hinstLib)
  {
    std::cerr << "Cannot load DLL 'mlp_x86-64_avx_d.dll' : " << GetLastError() << "\n";
    return 1;
  }

  // Load the function
  TrainNN1Type const trainNN1Func = reinterpret_cast<TrainNN1Type>(GetProcAddress(hinstLib, "trainNN1"));
  if(!trainNN1Func)
  {
    std::cerr << "Cannot find function 'trainNN1' : " << GetLastError() << "\n";
    FreeLibrary(hinstLib);
    return 1;
  }

  // Use the function
  trainNN1Func(layers_list, layers_len, train_len, learning_rate, batch_size, epoch_num, momentum, weight_decay, i_dropout, h_dropout, train_label_file, train_image_file, verbose);

  // Free the DLL
  FreeLibrary(hinstLib);
  return 0;
}