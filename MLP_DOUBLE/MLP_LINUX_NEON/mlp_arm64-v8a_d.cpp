// Includes

  #include <iostream>
  #include <vector>
  #include <algorithm>
  #include <cmath>
  #include <cstdlib>
  #include <ctime>
  #include <cstring>
  #include <arm_neon.h>
  #include <memory>
  #include <fstream>
  #include <sstream>
  #include <iomanip>
  #include <chrono>
  #include <thread>
  #include <ncurses.h>
  #include <functional>
  #include <filesystem>
  #include <mutex>
  #include <atomic>
  #include <random>
  
  

namespace mlp_d
{
  // Global variables
  
  std::vector<std::size_t> rand_list;
  std::atomic<bool> running(true);
  std::atomic<bool> leave(true);
  std::mutex randMutex;
  std::mutex genMutex;
  double epsilon = 0.00000008;



  // Structs and Classes

    struct offsets
    {
      std::size_t p_offset;
      std::size_t a_offset;
      std::size_t p_a_offset;
    };


    class Layer
    {
      private : 
      bool isFirst;
      std::size_t num;
      std::size_t prev_num;
      std::size_t weights_len;
      void setStatus(std::size_t layer_index,const std::vector<std::size_t> &layers_list);
      void randLayerArr(std::vector<double> &arr, std::size_t arrlen, long start, long end, std::uint16_t dp);
      void initLayer(bool randomize);

      protected:

      public:
      std::vector<double> weights;
      std::vector<double> biases;
      std::vector<double> preactvs;
      std::vector<double> actvs;
      Layer(const std::vector<std::size_t> &layers_list, std::size_t layer_index, bool randomize);
      ~Layer();
      std::size_t getNum();
      void randLayer();
    };


    class NeuralNetwork
    {
      private:

      protected:
      
      public:
      offsets &_offset;
      const std::vector<std::size_t> &layers_list;
      NeuralNetwork(const std::vector<std::size_t> &layers_list, offsets &_offset, bool randomize);
      ~NeuralNetwork();
      void addLayer(std::size_t index, bool randomize);

      void forwardPassExt(const std::vector<std::uint8_t> &image_data, double _i_dropout, double _h_dropout, std::vector<double> &preactvs, std::vector<double> &actvs);
      void clearNN();
      void randNN();
      Layer **NN_layers;
    };


    class SGD
    {
      private:
      std::size_t train_len;
      std::vector<std::size_t> &layers_list;
      std::size_t batch_size;
      double eta;
      double alpha;
      double lambda;
      double i_dropout;
      double h_dropout;
      const char *label_file;
      const char *image_file;
      NeuralNetwork *NN;
      std::size_t mark;

      std::vector<double>

      gradientCrossEntWthSftMax(const std::vector<double> &logits,
      const std::vector<std::uint8_t> &trueLabels),

      layersDerivs(std::size_t label, std::size_t p_offset, std::size_t a_offset, std::size_t layer, std::vector<double> velocities,
      const std::vector<double> &preactvs, const std::vector<double> &actvs, std::vector<double> &cActMat);

      void backProp(std::size_t label, std::vector<double> &gradients, std::vector<double> &velocities,
      const std::vector<double> &preactvs, const std::vector<double> &actvs);

      protected:
    
      public:
      SGD(std::vector<std::size_t> &_layers_list, std::size_t _train_len, std::size_t _batch_size,
      double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *_label_file, const char *_image_file,
      NeuralNetwork *_NN);
      ~SGD();
      void initSGD(std::size_t batch_offset, double _eta, std::vector<double> &previous_weights);
      void initSGDThread(std::size_t batch_offset, std::size_t i, std::vector<double> &gradients, std::vector<double> &velocities, std::size_t pix_size);
    };


    class AALR
    {
      private:
      std::size_t train_len;
      std::vector<std::size_t> &layers_list;
      std::size_t batch_size;
      std::size_t epoch_num;
      std::size_t epoch;
      double eta;
      double alpha;
      double lambda;
      double i_dropout;
      double h_dropout;
      const char *image_file;
      const char *label_file;
      std::vector<double> prev_gradient_deltas;
      std::vector<double> checkpoint;

      void trainEpoch(SGD &_SGD, std::size_t count, bool _verbose);

      protected:

      public:
      AALR(std::vector<std::size_t> &_layers_list, std::size_t _train_len, double _eta, std::size_t _batch_size, std::size_t _epoch_num,
      double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file);
      ~AALR();
      void runAALR(NeuralNetwork *preNN, bool _verbose);
    };



  // Main Functions Declaration

    void checkPtr(void *ptr);
    void checkDir(std::string dir_name);
    bool checkIndices(const std::vector<double> &vec, std::size_t start, std::size_t end);
    void escapeHelpExit();
    void escapeHelp();
    void escape(std::mutex &input_mutex, std::string &input_string);

    void add_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end);
    void sub_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end);
    void mul_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end);
    void div_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end);

    std::vector<double> sliceDblVec(const std::vector<double> &vec, std::size_t start, std::size_t end);
    std::vector<double> copyMergeDblVecs(const std::vector<double> &vecA, const std::vector<double> &vecB);
    std::vector<double> moveMergeDblVecs(const std::vector<double> &&vecA, const std::vector<double> &&vecB);

    void NN2Vec(NeuralNetwork* NN, std::vector<double> &arr, int offset, int flag);
    void Vec2NN(std::vector<double> &arr, NeuralNetwork* NN, int offset, int flag);

    std::size_t readLabel(const char *filename, std::size_t row);
    std::vector<std::uint8_t> readImage(const char *filename, std::size_t row, std::size_t pix_size);
    void saveNNVec(std::vector<double>& arr, double loss, std::size_t total);
    void saveNNLossLog(const char* pre_path, double loss, bool train, std::size_t suc, std::size_t total);
    void saveNNTestLabelLog(const char* pre_path, std::string label_data);
    void readNNVec(std::vector<double> &arr, std::vector<std::size_t> &layers_list, const char *filename);

    double custom_round(double num, std::uint16_t);
    double custom_rand(long long lb, long long ub, std::uint16_t);
    template <typename Container> void mt19937shuffle(Container& data);
    std::vector<double> dropoutList(std::size_t num, double dropout);

    std::size_t getLabel(std::vector<double> logits, std::size_t layers_len);
    void getOffsets(offsets &_offset, std::vector<size_t> &layers_list);
    double relu(double x);
    double reluDerivative(double x);
    std::vector<double> softmax(const std::vector<double>& logits, std::size_t start, std::size_t end);

    std::vector<std::uint8_t> getTrueLabels(std::size_t logits_size, std::size_t label);
    void meanCatCrsEntThread(std::vector<std::size_t> &rand_list, size_t num, const std::vector<size_t> &layers_list, double i_dropout, double h_dropout, 
    const char *label_file, const char *image_file, NeuralNetwork *NN, std::size_t index, double *losses, std::size_t pix_size);
    double meanCatCrossEntropy(std::vector<std::size_t> &rand_list, size_t len, const std::vector<size_t> &layers_list,
    double _i_dropout, double _h_dropout, const char *label_file, const char *image_file, NeuralNetwork *NN);



  // Main Functions Implementation

    void checkPtr(void *ptr)
    {
      if (ptr == nullptr)
      {
        std::cerr << "Invalid pointer." << std::endl;
        running.store(false);
      }
    }

    void checkDir(std::string dir_name)
    {
      namespace fs = std::filesystem;

      if (!fs::exists(dir_name))
      {
        if (!fs::create_directory(dir_name)) 
        {
          std::cerr << "Failed to create directory" << dir_name << "." << std::endl;
          running.store(false);
        }
      }
    }

    bool checkIndices(const std::vector<double> &vec, std::size_t start, std::size_t end)
    {
      if (start > end || (end - start) > vec.size())
      {
        std::cerr << "Invalid slice indices." << std::endl;
        running.store(false);
        return false;
      }

      else return true;
    }

    void escapeHelpExit()
    {
      std::size_t n = 6000;

      while (running.load() && n > 0)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        --n;
      }

      if (running.load() && leave.load()) std::exit(EXIT_SUCCESS);
    }

    void escapeHelp()
    {
      std::cout << "\nExit Help:\nPress escape key (esc) or 'exit' (any of the letters can be upper or lower case,\
 doesn't matter) + enter to exit.\n";
      std::cout << "If you have previously pressed other keys it also doesn't matter (as long as esc was not one of them),\
 what matters is that the last input was 'exit' + enter key or escape key.\n";
      std::cout << "Press any key (except esc) to continue. If no key is pressed, this program automatically exits after 1 minute.";

      // Initialize ncurses
      initscr();
      cbreak();
      noecho();
      nodelay(stdscr, TRUE); // Non-blocking input
      keypad(stdscr, TRUE); // Enable special keys like arrow keys

      int char_code = getch();
      if (char_code == 27) running.store(false);

      std::cout << "\n\n";     
      leave.store(false);
      endwin(); // End ncurses mode
    }

    void escape(std::mutex &input_mutex, std::string &input_string)
    {
      int ch;

      // Initialize ncurses
      initscr();
      cbreak();
      noecho();
      nodelay(stdscr, TRUE); // Non-blocking input
      keypad(stdscr, TRUE); // Enable special keys like arrow keys

      while (running.load())
      {
        ch = getch();
        if(ch != ERR)
        {
          if (ch == 27)
          {
            running.store(false);
            std::cout << "\nEscape key pressed. Exiting.\n";
          }

          else
          {
            std::lock_guard<std::mutex> lock(input_mutex);
            input_string.push_back(static_cast<char>(ch));

            if (input_string.size() >= 4)
            {
              std::string str = input_string.substr(input_string.size() - 4);
              std::transform(str.begin(), str.end(), str.begin(), ::tolower);

              if (str == "exit")
              {
                running.store(false);
                std::cout << "\n'Exit' + enter pressed. Exiting.\n";
              }
            }
          }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Sleep for 10ms to avoid busy waiting
      }
      
      endwin(); // End ncurses mode
    }
    
    void add_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end)
    {
      bool are_indices_valid = checkIndices(a, start, end);
      if (!are_indices_valid)
      {
        res = a;
        return;
      }

      std::size_t n = end - start;
      std::size_t i = 0;
      
      // Process groups of 2 doubles using NEON        
      for (i = 0; i < n; i += 2) 
      {
        float64x2_t va = vld1q_f64(&a[i]); // Load 2  doubles from vector a
        float64x2_t vb = vld1q_f64(&b[i]); // Load 2 doubles from vector b
        float64x2_t vr = vaddq_f64(va, vb); // Add the 2 doubles
        vst1q_f64(&res[i], vr); // Store the result back to result vector
      }

      // Handle remaining elements
      for (; i < n; ++i) res[i] = a[i] + b[i];
    }

    void sub_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end)
    {
      bool are_indices_valid = checkIndices(a, start, end);
      if (!are_indices_valid)
      {
        res = a;
        return;
      }

      std::size_t n = end - start;
      std::size_t i = 0;

      // Process groups of 2 doubles using NEON        
      for (i = 0; i < n; i += 2) 
      {
        float64x2_t va = vld1q_f64(&a[i]); // Load 2  doubles from vector a
        float64x2_t vb = vld1q_f64(&b[i]); // Load 2 doubles from vector b
        float64x2_t vr = vsubq_f64(va, vb); // Subtract the 2 doubles
        vst1q_f64(&res[i], vr); // Store the result back to result vector
      }

      // Handle remaining elements
      for (; i < n; ++i) res[i] = a[i] - b[i];
    }

    void mul_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end)
    {
      bool are_indices_valid = checkIndices(a, start, end);
      if (!are_indices_valid)
      {
        res = a;
        return;
      }

      std::size_t n = end - start;
      std::size_t i = 0;

      // Process groups of 2 doubles using NEON        
      for (i = 0; i < n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]); // Load 2  doubles from vector a
        float64x2_t vb = vld1q_f64(&b[i]); // Load 2 doubles from vector b
        float64x2_t vr = vmulq_f64(va, vb); // Multiply the 2 doubles
        vst1q_f64(&res[i], vr); // Store the result back to result vector
      }

      // Handle remaining elements
      for (; i < n; ++i) res[i] = a[i] * b[i];
    }

    void div_vectors_neon_double(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &res, std::size_t start, std::size_t end)
    {
      bool are_indices_valid = checkIndices(a, start, end);
      if (!are_indices_valid)
      {
        res = a;
        return;
      }

      std::size_t n = end - start;
      std::size_t i = 0;

      // Process groups of 2 doubles using NEON        
      for (i = 0; i < n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]); // Load 2  doubles from vector a
        float64x2_t vb = vld1q_f64(&b[i]); // Load 2 doubles from vector b
        float64x2_t vr = vdivq_f64(va, vb); // Divide the 2 doubles
        vst1q_f64(&res[i], vr); // Store the result back to result vector
      }

      // Handle remaining elements
      for (; i < n; ++i) res[i] = a[i] / b[i];
    }

    std::vector<double> sliceDblVec(const std::vector<double> &vec, std::size_t start, std::size_t end)
    {
      if (start > end || (end - start) > vec.size())
      {
        std::cerr << "Invalid slice indices." << std::endl;
        running.store(false);
        std::vector<double> zero_vector(vec.size());
        return zero_vector;
      }

      std::vector<double> sliced(vec.begin() + start, vec.begin() + end);
      return sliced;
    }

    std::vector<double> copyMergeDblVecs(const std::vector<double> &vecA, const std::vector<double> &vecB)
    {
      std::vector<double> merged_vector;
      merged_vector.reserve(vecA.size() + vecB.size());
      merged_vector.insert(merged_vector.end(), vecA.begin(), vecA.end());
      merged_vector.insert(merged_vector.end(), vecB.begin(), vecB.end());

      return merged_vector;
    }

    std::vector<double> moveMergeDblVecs(const std::vector<double> &&vecA, const std::vector<double> &&vecB)
    {
      std::vector<double> merged_vector;
      merged_vector.reserve(vecA.size() + vecB.size());

      merged_vector.insert(merged_vector.end(),
      std::make_move_iterator(vecA.begin()),
      std::make_move_iterator(vecA.end()));

      merged_vector.insert(merged_vector.end(),
      std::make_move_iterator(vecB.begin()),
      std::make_move_iterator(vecB.end()));

      return merged_vector;
    }

    void NN2Vec(NeuralNetwork* NN, std::vector<double> &arr, int offset, int flag)
    {
      const std::size_t layers_len = NN->layers_list.size();
      const std::vector<std::size_t> layers_list = NN->layers_list;
      std::size_t arr_offset = offset;

      for (std::size_t i = 1; i < layers_len; ++i)
      {
        std::size_t 
        l_len = layers_list[i],
        p_len = layers_list[i - 1],
        span = l_len * p_len + arr_offset,
        end = span + l_len;

        std::vector<double>
        weights = NN->NN_layers[i]->weights,
        biases = NN->NN_layers[i]->biases,
        parameters = copyMergeDblVecs(NN->NN_layers[i]->weights, NN->NN_layers[i]->biases),
        sliced_arr = sliceDblVec(arr, arr_offset, end);

        if (flag >= 0)
        {
          if (flag == 0)
          {
            sliced_arr.assign(sliced_arr.size(), 0.0);
          }
          add_vectors_neon_double(sliced_arr, parameters, sliced_arr, 0, sliced_arr.size());
        }

        else
        {
          sub_vectors_neon_double(sliced_arr, parameters, sliced_arr, 0, sliced_arr.size());
        }

        std::copy(sliced_arr.begin(), sliced_arr.end(), arr.begin() + arr_offset);
        arr_offset = end;
      }
    }

    void Vec2NN(std::vector<double> &arr, NeuralNetwork* NN, int offset, int flag)
    {
      const std::size_t layers_len = NN->layers_list.size();
      const std::vector<std::size_t> layers_list = NN->layers_list;
      std::size_t arr_offset = offset;

      for (std::size_t i = 1; i < layers_len; ++i)
      {
        std::size_t 
        l_len = layers_list[i],
        p_len = layers_list[i - 1],
        span = l_len * p_len + arr_offset,
        end = span + l_len;

        std::vector<double>
        parameters = copyMergeDblVecs(NN->NN_layers[i]->weights, NN->NN_layers[i]->biases),
        sliced_arr = sliceDblVec(arr, arr_offset, end);

        if(flag >= 0)
        {
          if (flag == 0)
          {
            parameters.assign(parameters.size(), 0.0);
          }
          add_vectors_neon_double(parameters, sliced_arr, parameters, 0, parameters.size());
        }

        else
        {
          sub_vectors_neon_double(parameters, sliced_arr, parameters, 0, parameters.size());
        }

        std::size_t mid = NN->NN_layers[i]->weights.size();
        std::copy(parameters.begin(), parameters.begin() + mid, NN->NN_layers[i]->weights.begin());
        std::copy(parameters.begin() + mid, parameters.end(), NN->NN_layers[i]->biases.begin());   
        arr_offset = end;
      }
    }

    std::size_t readLabel(const char *filename, std::size_t row)
    {
      char x[1];
      FILE *fptr;

      fptr = fopen(filename, "rb");
      if(!fptr) 
      {
        std::cerr << "File " << filename << " could not be opened." << std::endl;
        running.store(false);
        return 0;
      }

      else
      {
        fseek(fptr, 8 + row, SEEK_SET);
        fread(x, sizeof(char), 1, fptr);
        fclose(fptr);
        return static_cast<std::size_t>(x[0]);
      }
    }

    std::vector<std::uint8_t> readImage(const char *filename, std::size_t row, std::size_t pix_size)
    {
      std::vector<char> x(pix_size);
      FILE *fptr;
      std::vector<std::uint8_t> image_data(pix_size);

      fptr = fopen(filename, "rb");
      if(!fptr) 
      {
        std::cerr << "File " << filename << " could not be opened." << std::endl;
        running.store(false);
      }

      else
      {
        fseek(fptr, 16 + (row * pix_size), SEEK_SET);
        fread(x.data(), sizeof(char), pix_size, fptr);
        fclose(fptr);

        for (std::size_t i = 0; i < pix_size; ++i)
        {
          std::uint8_t k = static_cast<std::uint8_t>(x[i]);
          if (k < 0) k = 256 + k;
          image_data[i] = k;
        }
      }

      return image_data;
    }

    void saveNNVec(std::vector<double>& arr, double loss, std::size_t total)
    {
      FILE *fptr;

      std::string dir_name = "TRAINED_NN";
      checkDir(dir_name);

      std::string path_prefix = dir_name + "/NN_" + std::to_string(time(nullptr));
      std::string path = path_prefix + ".bin";
      const char *filename = path.c_str();

      std::size_t len = arr.size();
      double *temp_arr = new double[len];
      checkPtr(temp_arr);
      
      for(std::size_t i = 0; i < len; ++i) temp_arr[i] = arr[i];

      fptr = fopen(filename, "wb");
      fwrite(&temp_arr, __SIZEOF_DOUBLE__, len, fptr);
      fclose(fptr);
      delete[] temp_arr;

      const char* pre_path = path_prefix.c_str();
      std::cout << "Neural Network saved to " << filename << ".\n";
      saveNNLossLog(pre_path, loss, true, 0, total);
    }

    void saveNNLossLog(const char* pre_path, double loss, bool train, std::size_t suc, std::size_t total)
    {
      std::ofstream TextFile;
      std::string path_prefix = pre_path;
      std::string t_path = path_prefix + "_LossLog.txt";
      std::string log_prefix = "Neural network loss on ";
      std::string log;
      if (train) log = "training: ";
      else log = "testing: ";

      TextFile.open(t_path);
      TextFile << log_prefix << log << loss << ".\n";
      if (!train) 
      {
        TextFile << "Number of images successfully identified: " << suc << ".\n";
        TextFile << "Percentage success: " << ((double) suc / total) * 100 << "%.\n";
      }
      TextFile << "Number of images: " << total << ".\n";
      TextFile.close();

      std::cout << "Neural Network loss log saved to " << t_path << ".\n";
    }

    void saveNNTestLabelLog(const char* pre_path, std::string label_data)
    {
      std::ofstream TextFile;
      std::string path_prefix = pre_path;
      std::string t_path = path_prefix + "_LabelLog.txt";

      TextFile.open(t_path);
      TextFile << "Log for expected and actual labels:\n\n";
      TextFile << label_data;
      TextFile.close();

      std::cout << "Neural Network label log saved to " << t_path << ".\n";
    }

    void readNNVec(std::vector<double> &arr, std::vector<std::size_t> &layers_list, const char *filename)
    {
      FILE *fptr;
      double temp[2];

      fptr = fopen(filename, "rb");
      if(!fptr) 
      {
        std::cerr << "File " << filename << " could not be opened." << std::endl;
        running.store(false);
      }

      else
      {
        fread(temp, __SIZEOF_DOUBLE__, 2, fptr);
        std::size_t num = static_cast<std::size_t>(temp[0]);
        std::size_t len = static_cast<std::size_t>(temp[1]);

        double *temp_layers_list = new double[len], *temp_arr = new double[num];
        checkPtr(temp_layers_list);
        checkPtr(temp_arr);

        if(running.load())
        {
          fread(temp_layers_list, __SIZEOF_DOUBLE__, len, fptr);
          fread(temp_arr, __SIZEOF_DOUBLE__, num, fptr);
          fclose(fptr);

          for (std::size_t i = 0; i < len; ++i) layers_list.push_back(temp_layers_list[i]);
          for(std::size_t i = 0; i < num; ++i) arr.push_back(temp_arr[i]);

          // Free memory.
          delete[] temp_layers_list;
        }
      }
    }

    double custom_round(double num, std::uint16_t dp)
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(dp) << num;
      std::istringstream iss(oss.str());
      double rounded;
      iss >> rounded;
      return rounded;
    }

    double custom_rand(long long lb, long long ub, std::uint16_t dp)
    {
      if (lb >= ub) 
      {
        std::cerr << "Lower bound greater than or equal to upper bound." << std::endl;
        running.store(false);
        return 0.0;
      }

      static std::uint16_t count = 1;
      std::unique_lock<std::mutex> lock(randMutex);
      srand(time(0)*count);
      double whole = ((static_cast<long long int>(rand()) + LONG_MAX) % (ub - lb)) + lb;
      double decimal = 0;

      for (std::uint16_t i = 0; i < dp; ++i)
      {
        decimal *= 10;
        decimal += rand()%10;
      }
      lock.unlock();

      double num;
      if (dp == 0) num = whole;
      else num = whole + (decimal / pow(10, dp));
      
      ++count;
      if (count == UINT16_MAX) count = 1;

      return num;
    }

    template <typename Container>
    void mt19937shuffle(Container& data)
    {
      std::lock_guard<std::mutex> lock(genMutex);
      std::random_device rd;
      std::mt19937 generator(rd());
      std::shuffle(data.begin(), data.end(), generator);
    }

    std::vector<double> dropoutList(std::size_t num, double dropout)
    {
      std::vector<double> dropout_list(num);
      std::size_t stop = static_cast<std::size_t>(num * dropout);
      double mult = 1.0 / (1.0 - dropout);

      for(std::size_t i = 0; i < num; ++i)
      {
        if (i < stop) dropout_list[i] = 0.0;
        else dropout_list[i] = mult;
      }
      mt19937shuffle(dropout_list); 

      return dropout_list;
    }

    std::size_t getLabel(std::vector<double> logits, std::size_t last_layer_len)
    {
      double max = std::numeric_limits<double>::min();
      std::size_t max_index = -1;

      for (std::size_t i = 0; i < last_layer_len; ++i)
      {
        bool check = logits[i] > (max);
        max = check ? logits[i] : max;
        max_index = check ? i : max_index;
      }

      return max_index;
    }

    void getOffsets(offsets &_offset, std::vector<size_t> &layers_list)
    {
      std::size_t p_num = 0;
      std::size_t a_num = 0;
      std::size_t p_a_num = 0;
      std::size_t p_buf = 0;
      const std::size_t layers_len = layers_list.size();

      std::size_t n = 0;
      for (std::size_t i = 0; i < layers_len; ++i)
      {
        const std::size_t len = layers_list[i];

        if (n == 0)
        {
          p_buf = len;
          a_num = len;
        }

        else
        {
          p_buf *= len;
          p_num += p_buf + len;
          p_buf = len;
          a_num += len;
          p_a_num += len;
        }

        ++n;
      }

      _offset.p_offset = p_num;
      _offset.a_offset = a_num;
      _offset.p_a_offset = p_a_num;

      std::size_t w = p_num - p_a_num;
      std::cout << "Number of weights: " << w << "\n";
      std::cout << "Number of biases: " << p_num - w << "\n";
      std::cout << "Total number of parameters (weights + biases): " << p_num << "\n";
      std::cout << "Number of activations: " << a_num << "\n";
    }

    double relu(double x)
    {
      return std::max(0.0, x);
    }

    double reluDerivative(double x)
    {
      return x > 0 ? 1.0 : 0.0;
    }

    std::vector<double> softmax(const std::vector<double>& logits, std::size_t start, std::size_t end)
    {
      bool are_indices_valid = checkIndices(logits, start, end);
      if (!are_indices_valid) 
      {
        return std::vector<double> (logits.size());                
      }

      std::size_t len = end - start;
      std::vector<double> exp_values(len);
      double sum_exp = 0.0;

      // Calculate exponentials and their sum
      for (std::size_t i = 0; i < len; ++i)
      {
        exp_values[i] = std::exp(logits[i]);
        sum_exp += exp_values[i];
      }

      // Normalize to get probabilities
      for(std::size_t i = 0; i < len; ++i) exp_values[i] /= sum_exp;

      return exp_values;
    }

    std::vector<std::uint8_t> getTrueLabels(std::size_t logits_size, std::size_t label)
    {
      std::vector<std::uint8_t> trueLabels(logits_size);

      // Set the one-hot encoded labels 
      for(std::size_t i = 0; i < logits_size; ++i)
      {
        trueLabels[i] = i == label ? 1 : 0; 
      }

      return trueLabels;
    }

    void meanCatCrsEntThread(std::vector<std::size_t> &rand_list, size_t num, const std::vector<size_t> &layers_list, double i_dropout, double h_dropout, 
    const char *label_file, const char *image_file, NeuralNetwork *NN, std::size_t index, double *losses, std::size_t pix_size)
    {    
      losses[index] = 0.0;

      for (std::size_t i = 0; i < num; ++i)
      {        
        if(!running.load()) return;
        std::size_t 
        rand_index = rand_list[index * num + i],        
        label = readLabel(label_file, rand_index);
        std::vector<std::uint8_t> image_data = readImage(image_file, rand_index, pix_size);

        std::size_t last_layer_len = layers_list[layers_list.size() - 1];
        std::vector<double>
        preactvs(NN->_offset.p_a_offset),
        actvs(NN->_offset.a_offset);
        NN->forwardPassExt(image_data, i_dropout, h_dropout, preactvs, actvs);
        std::vector<double> prob = softmax(actvs, actvs.size() - last_layer_len, actvs.size());
        losses[index] -= std::log(prob[label] + epsilon);
      }

      std::lock_guard<std::mutex> lock(randMutex);
      losses[index] /= num;
    }

    double meanCatCrossEntropy(std::vector<std::size_t> &rand_list, size_t len, const std::vector<size_t> &layers_list, 
    double i_dropout, double h_dropout, const char *label_file, const char *image_file, NeuralNetwork *NN)
    {
      std::cout << "Getting loss...\n";

      using namespace std::chrono;
      const auto start{steady_clock::now()};

      std::size_t const thread_num = 8;
      std::vector<std::thread*> thr(thread_num);
      double loss = 0.0,
      losses[thread_num];
      
      std::size_t
      pix_size = layers_list[0],
      num = len / thread_num,
      last = len - ((thread_num - 1) * num);

      for(std::size_t i = 0; i < thread_num; ++i)
      {
        std::size_t n = i < (thread_num - 1) ? num : last;  
        if (!running.load()) break;
        thr[i] = new std::thread(&meanCatCrsEntThread, std::ref(rand_list), n, std::ref(layers_list), i_dropout, h_dropout, label_file, image_file, NN, i, losses, pix_size);
      }

      if(running.load())
      {
        for(std::size_t i = 0; i < thread_num; ++i) thr[i]->join();
        for(std::size_t i = 0; i < thread_num; ++i) loss += losses[i];
        loss /= thread_num; 

        const auto stop{steady_clock::now()};
        const duration<double> elapsed_time(stop - start);
        std::cout << "Time taken: " << elapsed_time.count() << " second(s).\n\n";
        
        return custom_round(loss, 4);
      }

      else return 0.0;
    }

    Layer::Layer(const std::vector<std::size_t> &layers_list, std::size_t layer_index, bool randomize)
    {
      num = layers_list[layer_index];
      setStatus(layer_index, layers_list);
      initLayer(randomize);
    }

    Layer::~Layer(){}

    std::size_t Layer::getNum()
    {
      return num;
    }

    void Layer::setStatus(std::size_t layer_index, const std::vector<std::size_t> &layers_list)
    {
      if (layer_index == 0) 
      {
        isFirst = true;
        prev_num = 0;
      }
      else {
        isFirst = false;
        prev_num = layers_list[layer_index-1];
      }
    }

    void Layer::initLayer(bool randomize)
    {
      actvs.resize(num);

      if (!isFirst)
      {
        weights_len = prev_num * num;
        preactvs.resize(num);
        weights.resize(weights_len);
        biases.resize(num);
      }
      
      if (randomize) randLayer();
    }

    void Layer::randLayer()
    {
      if (!isFirst)
      {
        randLayerArr(weights, weights_len, -1, 1, 5);
        randLayerArr(biases, num, -10, 11, 0);
      }
    }

    void Layer::randLayerArr(std::vector<double> &arr, std::size_t arrlen, long start, long end, std::uint16_t dp)
    {
      for (std::size_t i = 0; i < arrlen; ++i)
      {
        if(!running.load()) return;
        arr[i] = custom_rand(start, end, dp);
      }
    }


    NeuralNetwork::NeuralNetwork(const std::vector<std::size_t> &layers, offsets &__offset, bool randomize)
    :layers_list(layers), _offset(__offset)
    {
      const size_t layers_num = layers_list.size();
      NN_layers = new Layer*[layers_num];// Initialize the layers pointer array.
      for (std::size_t i = 0; i < layers_num; ++i)
      {
        if(!running.load()) return;
        addLayer(i, randomize); // Add the layers
      }
    }

    NeuralNetwork::~NeuralNetwork(){}

    void NeuralNetwork::addLayer(std::size_t index, bool randomize)
    {
      NN_layers[index] = new Layer(layers_list, index, randomize); // Create a new layer and add its address to the layers pointer array.
    }

    void NeuralNetwork::clearNN()
    {
      // Free memory.
      for (std::size_t i = 0; i<layers_list.size(); ++i)
      {
        if(!running.load()) return;
        delete NN_layers[i];
      }
      delete [] NN_layers;
    }

    void NeuralNetwork::randNN()
    {
      for (std::size_t i = 0; i < layers_list.size(); ++i) NN_layers[i]->randLayer();
    }

    void NeuralNetwork::forwardPassExt(const std::vector<std::uint8_t> &image_data,  double i_dropout,
    double h_dropout, std::vector<double> &preactvs, std::vector<double> &actvs) 
    {
      const std::size_t layers_len = layers_list.size();
      std::size_t start = 0, ref = layers_list[0];

      for (std::size_t i = 0; i < layers_len; ++i)
      {
        if(!running.load()) return;
        std::size_t l_len = layers_list[i];
        std::vector<double> dropoutVals;

        if (i == 0) 
        {
          dropoutVals = dropoutList(l_len, i_dropout);
          for (std::size_t j = 0; j < l_len; ++j)
          {
            actvs[j] = (double) image_data[j] / 255 * dropoutVals[j];
          }  
        } 

        else
        { 
          if (i != layers_len - 1) dropoutVals = dropoutList(l_len, h_dropout); // don't perform on the last layer
          std::size_t dif = start - ref, p_len = layers_list[i - 1];

          std::vector<double> last_actvs = NN_layers[i - 1]->actvs,
          weights = NN_layers[i]->weights,
          biases = NN_layers[i]->biases;

          for (size_t j = 0; j < l_len; ++j)
          {
            for (std::size_t k = 0; k < p_len; ++k) preactvs[j + dif] += weights[j * p_len + k] * last_actvs[k];
            preactvs[j + dif] += biases[j];
          }

          if (i == layers_len - 1)
          {
            std::vector<double> 
            last_layer_actvs = softmax(preactvs, preactvs.size() - l_len, preactvs.size()),
            last_layer_actvs_padding(actvs.size() - l_len),
            stored_actvs = moveMergeDblVecs(std::move(last_layer_actvs_padding), std::move(last_layer_actvs));
            add_vectors_neon_double(actvs, stored_actvs, actvs, 0, actvs.size());
          } 
          
          else
          {
            for (std::size_t j = 0; j < l_len; ++j) 
            {
              preactvs[j + dif] *= dropoutVals[j]; // don't perform on the last layer
              actvs[j + start] = relu(preactvs[j + dif]);  
            } 
          }       
        } 

        start += l_len;
      }
    }


    SGD::SGD(std::vector<std::size_t> &_layers_list, std::size_t _train_len, std::size_t _batch_size,
    double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *_label_file, const char *_image_file,
    NeuralNetwork* _NN)
    :layers_list(_layers_list), train_len(_train_len), batch_size(_batch_size), alpha(_alpha), lambda(_lambda), 
    i_dropout(_i_dropout), h_dropout(_h_dropout), label_file(_label_file), image_file(_image_file), NN(_NN)
    {};

    SGD::~SGD(){};

    void SGD::initSGD(std::size_t batch_offset, double _eta, std::vector<double> &previous_weights)
    {
      eta = _eta;
      std::size_t pix_size = layers_list[0];
      std::vector<double> gradients(NN->_offset.p_offset);
      std::vector<std::thread*> thread_ptrs(batch_size);

      std::vector<std::vector<double>> 
      thread_gradients(batch_size, std::vector<double>(NN->_offset.p_offset)),
      thread_velocities(batch_size, std::vector<double>(NN->_offset.p_offset));

      for (std::size_t i = 0; i < batch_size; ++i)
      {
        add_vectors_neon_double(thread_velocities[i], previous_weights, thread_velocities[i], 0, thread_velocities[i].size());
      }

      for(std::size_t i = 0; i < batch_size; ++i)
      {
        if(!running.load()) return;
        thread_ptrs[i] = new std::thread(&SGD::initSGDThread, this, batch_offset, i, std::ref(thread_gradients[i]), std::ref(thread_velocities[i]), pix_size);
      }

      if(running.load())
      {
        // Terminate threads
        for (std::size_t i = 0; i < batch_size; ++i) thread_ptrs[i]->join();

        // Sum up thread outputs and store
        for (std::size_t i = 1; i < batch_size; ++i)
        {
          add_vectors_neon_double(thread_gradients[0], thread_gradients[i], thread_gradients[0], 0, thread_gradients[0].size());
          add_vectors_neon_double(thread_velocities[0], thread_velocities[i], thread_velocities[0], 0, thread_velocities[0].size());
        }

        // Divide summed up thread outputs by the batch size to get the mean value and update the gradients with the mean value.
        std::vector<double> batch_size_vec(NN->_offset.p_offset);
        batch_size_vec.assign(batch_size_vec.size(), static_cast<double>(batch_size));
        div_vectors_neon_double(thread_gradients[0], batch_size_vec, thread_gradients[0], 0, thread_gradients[0].size());
        add_vectors_neon_double(gradients, thread_gradients[0], gradients, 0, gradients.size());

        // Update the previous weights
        div_vectors_neon_double(thread_velocities[0], batch_size_vec, thread_velocities[0], 0, thread_velocities[0].size());
        previous_weights.assign(previous_weights.size(), 0.0);
        add_vectors_neon_double(previous_weights, thread_velocities[0], previous_weights, 0, previous_weights.size());

        // Free thread pointers
        for (std::size_t i = 0; i < batch_size; ++i)
        {
          free(thread_ptrs[i]);
        }

        Vec2NN(gradients, NN, 0, 1); // Increment the values of the neural network's weights and biases by the values of the gradients double array.
      }
    }

    void SGD::initSGDThread(std::size_t batch_offset, std::size_t i, std::vector<double> &gradients, std::vector<double> &velocities, std::size_t pix_size)
    {
      std::size_t index = rand_list[batch_offset + i];
      std::size_t label = readLabel(label_file, index);
      std::vector<std::uint8_t> image_data = readImage(image_file, index, pix_size);
      std::vector<double>
      preactvs(NN->_offset.p_a_offset),
      actvs(NN->_offset.a_offset);
      NN->forwardPassExt(image_data, i_dropout, h_dropout, preactvs, actvs);
      backProp(label, gradients, velocities, preactvs, actvs);
    }

    void SGD::backProp(std::size_t label, std::vector<double> &gradients, std::vector<double> &velocities,
    const std::vector<double> &preactvs, const std::vector<double> &actvs)
    { 
      std::size_t layer = layers_list.size() - 1;
      std::vector<double> emptyCActMat;
      gradients = layersDerivs(label, NN->_offset.p_offset, NN->_offset.a_offset, layer, velocities, preactvs, actvs, emptyCActMat);
    }

    std::vector<double> SGD::gradientCrossEntWthSftMax(const std::vector<double> &logits,
    const std::vector<std::uint8_t> &trueLabels)
    {
      // Function to calculate gradient of loss function wrt logits
      std::vector<double> prob = softmax(logits, 0, logits.size());

      // Gradient calculation
      std::vector<double> gradient (prob.size());
      for(size_t i = 0; i < prob.size(); ++i)
      {
        gradient[i] = prob[i] - trueLabels[i];
      }

      return gradient;
    }

    std::vector<double> SGD::layersDerivs(std::size_t label, std::size_t p_offset, std::size_t a_offset, std::size_t layer,
    std::vector<double> velocities, const std::vector<double> &preactvs, const std::vector<double> &actvs, std::vector<double> &cActMat)
    {
      std::size_t
      l = layers_list[layer],
      p = layers_list[layer - 1],
      new_p_offset = p_offset - (l * (1 + p)),
      bias_offset = p_offset - l,
      new_a_offset = a_offset - l,
      p_a_offset = new_a_offset - layers_list[0];

      std::vector<double> 
      layer_weights = NN->NN_layers[layer]->weights,
      gradients(l * (1 + p));

      if (layer == layers_list.size() - 1)
      {
        std::vector<double> l_actvs = sliceDblVec(actvs, a_offset - l, a_offset);
        std::vector<std::uint8_t> trueLabels = getTrueLabels(l_actvs.size(), label);
        cActMat = gradientCrossEntWthSftMax(l_actvs, trueLabels);
      }

      else
      {
        std::size_t n = layers_list[layer + 1];
        std::vector<double>
        n_l_preactvs = sliceDblVec(preactvs, p_a_offset + l, p_a_offset + l + n),
        next_cActMat = cActMat,
        next_layer_weights = NN->NN_layers[layer + 1]->weights; 
        cActMat.resize(l);

        for(std::size_t i = 0; i < l; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
          {
            cActMat[i] += next_layer_weights[i * n + j] * reluDerivative(n_l_preactvs[j]) * next_cActMat[j];
          }
        }
      }

      for (std::size_t j = 0; j < l; ++j)
      {
        double z_vals = 1.0;
        if(layer < layers_list.size() - 1) z_vals = reluDerivative(preactvs[j + p_a_offset]);

        for (std::size_t k = 0; k < p; ++k)
        {
          std::size_t index_g = j * p + k;
          std::size_t index_v = new_p_offset + index_g;
          double gradient_delta = actvs[k + new_a_offset - p] * z_vals * cActMat[j];
          velocities[index_v] = alpha * velocities[index_v] + (1 - alpha) * gradient_delta;
          double decay = lambda * layer_weights[index_g];
          gradients[index_g] = -eta * (velocities[index_v] + decay);        
        }

        gradients[l * p + j] = -eta * z_vals * cActMat[j];
      }

      if (layer > 1) 
      {
        std::vector<double> prev_gradients = layersDerivs(label, new_p_offset, new_a_offset, layer - 1, velocities, preactvs, actvs, cActMat);
        gradients = copyMergeDblVecs(prev_gradients, gradients);
      }
      
      return gradients;
    }


    AALR::AALR(std::vector<std::size_t> &_layers_list, std::size_t _train_len, double _eta, std::size_t _batch_size, std::size_t _epoch_num,
    double _alpha, double _lambda, double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file)
    :layers_list(_layers_list), train_len(_train_len), eta(_eta), batch_size(_batch_size), epoch_num(_epoch_num),
    alpha(_alpha), lambda(_lambda), i_dropout(_i_dropout), h_dropout(_h_dropout), label_file(train_label_file), image_file(train_image_file)
    {}

    AALR::~AALR(){}

    void AALR::runAALR(NeuralNetwork *preNN, bool _verbose)
    {
      std::size_t layers_len = layers_list.size();
      if (layers_len < 2)
      {
        std::cerr << "Number of Neural Network Layers less than 2." << std::endl;
        running.store(false);
      }

      else
      {
        std::cout << "Initializing Neural Network training with " << train_len << " images(s), [";
        for (std::size_t i = 0; i < layers_len; ++i) 
        {
          if (i == layers_len - 1) std::cout << layers_list[i];
          else std::cout << layers_list[i] << ", ";
        }
        std::cout << "] layers, and " << epoch_num << " sessions (limiting epochs).\n";
        std::cout << "Learning rate: " << eta << ", Momentum: " << alpha << ", Weight decay: " << lambda << ", Dropout rate: "
        << i_dropout << " (Input) - " << h_dropout << " (Hidden)" << ", Batch size: " << batch_size << 
        ", Labels source file: " << label_file << " and Images source file: " << image_file << ".\n\n";

        rand_list.resize(train_len);
        std::iota(rand_list.begin(), rand_list.end(), 0);
        mt19937shuffle(rand_list);

        offsets _offset;
        NeuralNetwork *NeuralNet;  

        if (preNN == nullptr) 
        {
          getOffsets(_offset, layers_list);
          NeuralNetwork NeuralNetInit(layers_list, _offset, true);  // Randomize the weight and values upon initiating the neural network
          NeuralNet = &NeuralNetInit;
        }
        else NeuralNet = preNN;

        SGD _SGD(layers_list, train_len, batch_size, alpha, lambda, i_dropout, h_dropout, label_file, image_file, NeuralNet);

        double best_loss = meanCatCrossEntropy(rand_list, train_len, layers_list, i_dropout, h_dropout, label_file, image_file, NeuralNet);
        while(std::isnan(best_loss))
        {
          if (!running.load()) break;
          NeuralNet->randNN();
          std::cout << "Loss is NaN, retrying... \n";
          best_loss = meanCatCrossEntropy(rand_list, train_len, layers_list, i_dropout, h_dropout, label_file, image_file, NeuralNet);
        } 

        double loss;
        prev_gradient_deltas.resize(_offset.p_offset);
        checkpoint.resize(_offset.p_offset);

        // PHASE 1: Start Initial Learning Rate Exploration.
        std::cout << "Phase 1 of learning rate exploration started. Initial loss: " << best_loss << ".\n\n";
        
        std::size_t p = 10; // Patience.
        std::size_t i = 0; // Patience counter.
        std::size_t t = 0; // Epoch counter.

        while (i < p)
        {
          if (!running.load()) break;
          trainEpoch(_SGD, t, _verbose);
          loss = meanCatCrossEntropy(rand_list, train_len, layers_list, i_dropout, h_dropout, label_file, image_file, NeuralNet);  
          ++i;
          ++t;
          std::cout << "Cumulative number of epoch(s) trained: " << t << ", Patience counter: " << i <<  ", Patience: " << p << ", Loss: " << loss << ", Best loss: "
          << best_loss << ", Learning Rate: " << eta << ".\n\n";

          if (std::isnan(loss) or (loss > best_loss))
          {
            eta /= 2;
            i = 0;
            Vec2NN(prev_gradient_deltas, NeuralNet, 0, -1); // Decrement the values of the neural network's weights and biases by the values of the previous delta array.
          }

          else best_loss = loss;
        }
        NN2Vec(NeuralNet, checkpoint, 0, 0); // Set the values of the checkpoint's double array to the values of the neural network's weights and biases.

        std::cout << "Phase 1 of learning rate exploration ended. Final loss: " << best_loss << ".\n\n";

        // PHASE 2: Start Optimistic Binary Exploration
        std::cout << "Phase 2 of optimistic binary exploration started. Initial loss: " << best_loss << ".\n\n";
        
        eta *= 2;
        p = 1;
        t = 0;

        while (t < epoch_num)
        {
          if (!running.load()) break;
          for (std::size_t i = 0; i < p; ++i) trainEpoch(_SGD, t, _verbose);
          loss = meanCatCrossEntropy(rand_list, train_len, layers_list, i_dropout, h_dropout, label_file, image_file, NeuralNet);  
          ++t;
          std::cout << "Session: " << t << ", Number of epochs trained: " << p << ", Loss: " << loss << ", Best loss: " << best_loss << ", Learning Rate: " << eta << ".\n\n";

          if (std::isnan(loss))
          {
            eta /= 2;
            p *= 2;
            Vec2NN(checkpoint, NeuralNet, 0, 0); // Set the values of the neural network's weights and biases to the values of the checkpoint's double array.
            continue;
          }

          if (loss < best_loss)
          {
            best_loss = loss;
            NN2Vec(NeuralNet, checkpoint, 0, 0); // Set the values of the checkpoint's double array to the values of the neural network's weights and biases.
            eta *= 2;
            p = 1;
          }

          else
          {
            if (!running.load()) break;
            for (std::size_t i = 0; i < p; ++i) trainEpoch(_SGD, t, _verbose);
            ++t;
            loss = meanCatCrossEntropy(rand_list, train_len, layers_list, i_dropout, h_dropout, label_file, image_file, NeuralNet);  
            std::cout << "Session: " << t << ", Number of epochs trained: " << p << ", Loss: " << loss << ", Best loss: " << best_loss << ", Learning Rate: " << eta << ".\n\n";

            if (loss < best_loss)
            {
              best_loss = loss;
              NN2Vec(NeuralNet, checkpoint, 0, 0); // Set the values of the checkpoint's double array to the values of the neural network's weights and biases.
              eta *= 2;
              p = 1;
            }

            else
            {
              eta /= 2;
              p *= 2;

              if (std::isnan(loss))
              {
                Vec2NN(checkpoint, NeuralNet, 0, 0); // Set the values of the neural network's weights and biases to the values of the checkpoint's double array.
              }
            }
          }
        }

        std::cout << "Phase 2 of optimistic binary exploration ended. Final loss: " << best_loss << ".\n\n";

        std::size_t pad = layers_len + 2;
        std::vector<double> output(_offset.p_offset + pad);

        // Set the intial part of the output's double array which stores
        // the array size, layers array length and each layer's length values.
        int q = 0;
        for (int i = 0; i < pad; ++i)
        {
          if (i == 0) output[i] = _offset.p_offset;
          else if (i == 1) output[i] = layers_len;
          else
          {
            output[i] = layers_list[q];
            ++q;
          }
        }
        NN2Vec(NeuralNet, output, pad, 0); // Set the values of the output's double array to the values of the neural network's weights and biases.
        NeuralNet->clearNN();
        
        // Save the output's double array values to a BIN file.
        saveNNVec(output, best_loss, train_len);

        running.store(false);
      }
    }

    void AALR::trainEpoch(SGD &_SGD, std::size_t count, bool _verbose)
    {
      if (_verbose) std::cout << "Training epoch...\n";
      using namespace std::chrono;  
      const auto start{steady_clock::now()};

      mt19937shuffle(rand_list);
      std::size_t divs = train_len / batch_size; // The number of batches to train for.
      for (std::size_t i = 0; i < divs; ++i) 
      {
        std::size_t batch_offset = i * batch_size;
        _SGD.initSGD(batch_offset, eta, prev_gradient_deltas); // Runs SGD for the batch.
      }
      
      const auto stop{steady_clock::now()};
      const duration<double> elapsed_time(stop - start);
      if (_verbose) std::cout << "Epoch trained. Time taken: " << elapsed_time.count() << " second(s).\n";
    }
}



// DLL Helper Functions Declaration

  extern "C" int NN(const char *mlp_filename,  double _i_dropout, double _h_dropout, std::uint8_t *_image_data, unsigned _image_data_size, bool _verbose);
  extern "C" void trainNN1(unsigned *layers_list, unsigned layers_len, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda,  double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose);
  extern "C" void trainNN2(const char *mlp_filename, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda,  double _i_dropout, double _h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose);
  void testNNThread(const char *mlp_filename, std::size_t _test_len,  double _i_dropout, double _h_dropout, const char *test_label_file,
  const char *test_image_file, bool _verbose, bool validate);
  extern "C" void testNN(const char *mlp_filename, unsigned _test_len,  double _i_dropout, double _h_dropout, const char *test_label_file,
  const char *test_image_file, bool _verbose, bool validate);



// DLL Helper Functions Implementation

  using namespace mlp_d;

  extern "C" int NN(const char *mlp_filename, double i_dropout, double h_dropout, uint8_t *_image_data, unsigned _image_data_size, bool _verbose)
  {
    using namespace std::chrono;
    const auto start{steady_clock::now()};

    std::vector<double> input;
    std::vector<std::size_t> layers_list;
    std::vector<std::uint8_t> image_data(_image_data, _image_data + _image_data_size);
    readNNVec(input, layers_list, mlp_filename);

    if(running.load())
    {
      offsets _offset;
      getOffsets(_offset, layers_list);
      NeuralNetwork NeuralNet(layers_list, _offset, false); // Don't randomize the weight and values upon initiating the neural network.
      Vec2NN(input, &NeuralNet, 0, 0); // Set the values of the neural network's weights and biases to the values of the input double array.

      std::size_t last_layer_len = layers_list[layers_list.size() - 1];
      std::vector<double>
      preactvs(_offset.p_a_offset),
      actvs(_offset.a_offset),
      logits;
      NeuralNet.forwardPassExt(image_data, i_dropout, h_dropout, preactvs, actvs);
      logits = sliceDblVec(actvs, actvs.size() - last_layer_len, actvs.size());
      std::size_t label = getLabel(logits, last_layer_len);

      // Free memory.
      NeuralNet.clearNN();

      if (_verbose) std::cout << "Classified image label: " << label << ".\n";

      const auto stop{steady_clock::now()};
      const duration<double> elapsed_time(stop - start);
      if (_verbose) std::cout << "Time taken: " << elapsed_time.count() << " second(s).\n\n";

      return label;
    }

    else return 0.0;   
  }


  extern "C" void trainNN1(unsigned *layers_list, unsigned layers_len, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda, double i_dropout, double h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose)
  {
    using namespace std::chrono;
    const auto start{steady_clock::now()};

    std::mutex input_mutex;
    std::string input_string;

    std::thread escHelpExit(&escapeHelpExit);
    std::thread escHelp(&escapeHelp);
    escHelp.join();

    std::thread esc_thread(&escape, std::ref(input_mutex), std::ref(input_string));  
    std::vector<std::size_t> _layers_list(layers_list, layers_list + layers_len);
    AALR aalr(_layers_list, _train_len, _eta, _batch_size, _epoch_num, _alpha, _lambda, i_dropout, h_dropout, train_label_file, train_image_file);
    std::thread aalr_thread(&AALR::runAALR, aalr, nullptr, _verbose);

    escHelpExit.join();
    esc_thread.join();
    aalr_thread.join();

    const auto stop{steady_clock::now()};
    const duration<double> elapsed_time(stop - start);
    std::cout << "Time taken: " << elapsed_time.count() << " second(s).\n\n";

    std::exit(EXIT_SUCCESS);
  }


  extern "C" void trainNN2(const char *mlp_filename, unsigned _train_len, double _eta, unsigned _batch_size, unsigned _epoch_num,
  double _alpha, double _lambda, double i_dropout, double h_dropout, const char *train_label_file, const char *train_image_file, bool _verbose)
  {
    using namespace std::chrono;
    const auto start{steady_clock::now()};

    std::mutex input_mutex;
    std::string input_string;

    std::thread escHelpExit(&escapeHelpExit);
    std::thread escHelp(&escapeHelp);
    escHelp.join();
    std::thread esc_thread(&escape, std::ref(input_mutex), std::ref(input_string));
    
    std::vector<double> input;
    std::vector<std::size_t> layers_list;
    readNNVec(input, layers_list, mlp_filename);

    if(running.load())
    {
      offsets _offset;
      getOffsets(_offset, layers_list);
      NeuralNetwork NeuralNet(layers_list, _offset, false); // Don't randomize the weight and values upon initiating the neural network.
      Vec2NN(input, &NeuralNet, 0, 0); // Set the values of the neural network's weights and biases to the values of the input double array.
      
      AALR aalr(layers_list, _train_len, _eta, _batch_size, _epoch_num, _alpha, _lambda, i_dropout, h_dropout, train_label_file, train_image_file);
      std::thread aalr_thread(&AALR::runAALR, aalr, nullptr, _verbose);
      aalr_thread.join();
    }
    escHelpExit.join();
    esc_thread.join();

    const auto stop{steady_clock::now()};
    const duration<double> elapsed_time(stop - start);
    std::cout << "Time taken: " << elapsed_time.count() << " second(s).\n\n";
    
    std::exit(EXIT_SUCCESS);
  }


  void testNNThread(const char *mlp_filename, std::size_t _test_len, double i_dropout, double h_dropout, const char *test_label_file,
  const char *test_image_file, bool _verbose, bool validate)
  {
    std::vector<double> input;
    std::vector<std::size_t> layers_list;
    readNNVec(input, layers_list, mlp_filename);
    if(!running.load()) return;

    double overall_loss = 0.0;
    std::size_t pix_size = layers_list[0],
    n = 0, layers_len = layers_list.size();

    std::string label_data;
    std::ostringstream pre_label;

    std::cout << "Initializing Neural Network testing with " << _test_len << " images(s), [";
    for (std::size_t i = 0; i < layers_len; ++i) 
    {
      if (i == layers_len - 1) std::cout << layers_list[i];
      else std::cout << layers_list[i] << ", ";
    }
    std::cout << "] layers and Neural Network model source file: " << mlp_filename << ".\n";
    std::cout << "Labels source file: " << test_label_file << " and Images source file: " << test_image_file << ".\n\n";

    offsets _offset;
    getOffsets(_offset, layers_list);
    NeuralNetwork NeuralNet(layers_list, _offset, true); // Randomize the weight and values upon initiating the neural network.
    Vec2NN(input, &NeuralNet, 0, 0); // Set the values of the neural network's weights and biases to the values of the input double array.

    for (std::size_t i = 0; i < _test_len; ++i) // Process each testing image.
    {
      if(!running.load()) return;
      double loss;
      std::size_t last_layer_len = layers_list[layers_list.size() - 1];
      std::vector<double>
      preactvs(_offset.p_a_offset),
      actvs(_offset.a_offset),
      logits;

      std::size_t label = readLabel(test_label_file, i); // Read the image's label
      std::vector<std::uint8_t> image_data = readImage(test_image_file, i, pix_size); // Read the image's data.

      if(!running.load()) return;
      NeuralNet.forwardPassExt(image_data, i_dropout, h_dropout, preactvs, actvs); // Set the preactivations and activations.
      logits = sliceDblVec(actvs, actvs.size() - last_layer_len, actvs.size());
      std::vector<double> prob = softmax(logits, 0, logits.size());
      
      loss -= std::log(prob[label] + epsilon);
      std::size_t c_label = getLabel(logits, last_layer_len);
      overall_loss += loss;

      if (c_label == label) ++n;
      if (_verbose) std::cout << "Image: " << i << ", Image label: " << label << ", Classified image label: " << c_label << ", Loss: " << loss << ".\n\n";
      pre_label << "Image: " << i << ", Image label: " << label << ", Classified image label: " << c_label << ", Loss: " << loss << ".\n\n";
    }
    if(!running.load()) return;

    overall_loss /= _test_len;
    label_data = pre_label.str();
    NeuralNet.clearNN();

    std::cout << "Loss: " << overall_loss << ", Number of successfully identified images: " << n << "/" << _test_len << ".\n";
    std::cout << "Percentage success: " << ((double) n / _test_len) * 100 << "%.\n";
    std::cout << "Total number of images: " << _test_len << ".\n";
    
    std::string path_prefix;
    if(validate)
    {
      std::string dir_name = "VALIDATE_NN";
      checkDir(dir_name);
      path_prefix = dir_name + "/NN_" + std::to_string(time(nullptr));
    }
    else 
    {
      std::string dir_name = "TEST_NN";
      checkDir(dir_name);
      path_prefix = dir_name + "/NN_" + std::to_string(time(nullptr));
    }
    
    if(!running.load()) return;
    const char* pre_path = path_prefix.c_str();
    saveNNLossLog(pre_path, overall_loss, false, n, _test_len);
    saveNNTestLabelLog(pre_path, label_data);
  }

  extern "C" void testNN(const char *mlp_filename, unsigned _test_len, double i_dropout, double h_dropout, const char *test_label_file,
  const char *test_image_file, bool _verbose, bool validate)
  {
    using namespace std::chrono;
    const auto start{steady_clock::now()};

    std::mutex input_mutex;
    std::string input_string;

    std::thread escHelpExit(&escapeHelpExit);
    std::thread escHelp(&escapeHelp);
    escHelp.join();

    std::thread esc_thread(&escape, std::ref(input_mutex), std::ref(input_string));
    std::thread test_thread(&testNNThread, mlp_filename, _test_len, i_dropout, h_dropout, test_label_file, test_image_file, _verbose, validate);

    escHelpExit.join();
    esc_thread.join();
    test_thread.join();

    const auto stop{steady_clock::now()};
    const duration<double> elapsed_time(stop - start);
    std::cout << "Time taken: " << elapsed_time.count() << " second(s).\n\n";

    std::exit(EXIT_SUCCESS);
  }