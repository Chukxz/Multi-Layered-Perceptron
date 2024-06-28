#include <Windows.h>
#include <iostream>

int main()
{
  const char *mlp_filename = "./TRAINED_NN/NN_1716772158.bin";
  const char *train_image_file = "./train-images.idx3-ubyte";
  const char *train_label_file = "./train-labels.idx1-ubyte";
  int train_len = 60000;

  double learning_rate = 0.1;
  unsigned batch_size = 16;
  unsigned epoch_num = 10;
  double momentum = 0.9;
  double weight_decay = 0.7;
  double i_dropout = 0.2;
  double h_dropout = 0.5;
  bool verbose = true;

  using TrainNN2Type = void (__cdecl*)
  (const char *mlp_filename, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose);

  // Load the DLL
  HMODULE const hinstLib = LoadLibraryExW(L"./mlp_x86-64_avx_d.dll", nullptr, 0);
  if(!hinstLib)
  {
    std::cerr << "Cannot load DLL 'mlp_x86-64_avx_d.dll' : " << GetLastError() << "\n";
    return 1;
  }

  // Load the function
  TrainNN2Type const trainNN2Func = reinterpret_cast<TrainNN2Type>(GetProcAddress(hinstLib, "trainNN2"));
  if(!trainNN2Func)
  {
    std::cerr << "Cannot find function 'trainNN2' : " << GetLastError() << "\n";
    FreeLibrary(hinstLib);
    return 1;
  }

  // Use the function
  trainNN2Func(mlp_filename, train_len, learning_rate, batch_size, epoch_num, momentum, weight_decay, i_dropout, h_dropout, train_label_file, train_image_file, verbose);

  // Free the DLL
  FreeLibrary(hinstLib);
  return 0;
}