#include <Windows.h>
#include <iostream>
#include <string.h>

int main(int argv, char **argc)
{
  if (argv != 2) return 1;
  int len = strlen(argc[1]);

  const char *test_image_file = "./t10k-images.idx3-ubyte";
  const char *test_label_file = "./t10k-labels.idx1-ubyte";
  unsigned test_len = 10000;
  char *mlp_filename;
  bool verbose = false;
  bool validate = false;

  if(mlp_filename = (char*) malloc(len * sizeof(char)))
  {
    strcpy(mlp_filename, argc[1]);

    using TestNNType = void (__cdecl*)
    (const char *mlp_filename, unsigned _test_len, double _i_dropout, double _h_dropout, const char *test_label_file,
    const char *test_image_file, bool _verbose, bool validate);

    // Load the DLL
    HMODULE const hinstLib = LoadLibraryExW(L"./mlp_x86-64_avx_d.dll", nullptr, 0);
    if(!hinstLib)
    {
      std::cerr << "Cannot load DLL 'mlp_x86-64_avx_d.dll' : " << GetLastError() << "\n";
      return 1;
    }

    // Load the function
    TestNNType const testNNFunc = reinterpret_cast<TestNNType>(GetProcAddress(hinstLib, "testNN"));
    if(!testNNFunc)
    {
      std::cerr << "Cannot find function 'testNN' : " << GetLastError() << "\n";
      FreeLibrary(hinstLib);
      return 1;
    }

    // Use the function
    testNNFunc(mlp_filename, test_len, 0.0, 0.0, test_label_file, test_image_file, verbose, validate);

    // Free the DLL
    FreeLibrary(hinstLib);
    return 0;
  }

  return 1;
}