#include <dlfcn.h>
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

    using TestNNType = void (*)
    (const char *mlp_filename, unsigned _test_len, double _i_dropout, double _h_dropout, const char *test_label_file,
    const char *test_image_file, bool _verbose, bool validate);

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
    TestNNType const testNNFunc = reinterpret_cast<TestNNType>(dlsym(handle, "testNN"));
    const char* dlsym_error = dlerror();
    if (dlsym_error)
    {
      std::cerr << "Cannot load symbol 'testNN': " << dlsym_error << "\n";
      dlclose(handle);
      return 1;
    }

    // Use the function
    testNNFunc(mlp_filename, test_len, 0.0, 0.0, test_label_file, test_image_file, verbose, validate);

    // Free the SO
    dlclose(handle);
    return 0;
  }

  return 1;
}