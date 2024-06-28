#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>


double custom_rand(long long lb, long long ub, std::uint16_t dp)
{
  if (lb >= ub) 
  {
    std::cerr << "Lower bound greater than or equal to upper bound." << std::endl;
    return 0.0;
  }

  static std::uint16_t count = 1;
  srand(time(0)*count);
  double whole = ((static_cast<long long int>(rand()) + LONG_MAX) % (ub - lb)) + lb;
  double decimal = 0;

  for (std::uint16_t i = 0; i < dp; ++i)
  {
    decimal *= 10;
    decimal += rand()%10;
  }

  double num;
  if (dp == 0) num = whole;
  else num = whole + (decimal / pow(10, dp));
  
  ++count;
  if (count == UINT16_MAX) count = 1;

  return num;
}


double ro(double num, std::size_t dp)
{
  double factor = std::pow(10, dp);
  return std::round(num * factor / factor);
}

double rs(double num, std::uint16_t dp)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(dp) << num;
  std::istringstream iss(oss.str());
  double rounded;
  iss >> rounded;
  return rounded;
}

template <typename Container>
void benchmark_shuffle(Container& data)
{
  std::random_device rd;
  std::mt19937 generator(rd());

  auto start = std::chrono::high_resolution_clock::now();
  std::shuffle(data.begin(), data.end(), generator);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Shuffling " << data.size() << " elements took " << duration.count()
  << " seconds." << std::endl;
}

template <typename Container>
void benchmark_randSpec(Container* data, std::size_t num)
{
  std::size_t half = num / 2;

  auto start = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < half; ++i)
  {
    std::size_t
    a = static_cast<std::size_t>(custom_rand(0, static_cast<long long int>(num), 0)),
    b = static_cast<std::size_t>(custom_rand(0, static_cast<long long int>(num), 0)),
    temp = data[a];

    data[a] = data[b];
    data[b] = temp;
  }  
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Shuffling " << num << " elements took " << duration.count()
  << " seconds." << std::endl;
}

void createSpecArr(std::size_t *arr, std::size_t num)
{
  for (std::size_t i = 0; i < num; ++i)
  {
    arr[i] = i;
  }
}

struct offsets
{
  std::size_t p_offset;
  std::size_t a_offset;
  std::size_t p_a_offset;
};

std::size_t layers_list[4] = {784, 32, 32, 10};

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

// enum offset_manager
// {
//   o_weights,
//   o_biases,
//   o_preactvs,
//   o_actvs,
// };

// class OffsetManager
// {
//   private :
//   std::size_t layer;
//   std::size_t w_offset;
//   std::size_t b_offset;
//   std::size_t p_a_offset;
//   std::size_t a_offset;
//   offsets &offset;
//   std::vector<size_t> &layers_list;
//   std::size_t processOffset(std::size_t enum_code, std::size_t l);

//   protected:

//   public :
//   std::size_t getCurrentLayerOffset(std::size_t enum_code);
//   std::size_t getPreviousLayerOffset(std::size_t enum_code);
//   std::size_t getNextLayerOffset(std::size_t enum_code);

//   OffsetManager(offsets &_offset, std::size_t _layer, std::vector<size_t> &_layers_list);
//   OffsetManager(const OffsetManager &other);
//   OffsetManager& operator= (const OffsetManager& other);
//   ~OffsetManager();
// };

// OffsetManager::OffsetManager(offsets &_offset, std::size_t _layer, std::vector<size_t> &_layers_list)
// : offset(_offset), layer(_layer), layers_list(_layers_list)
// {}

// OffsetManager::OffsetManager(const OffsetManager &other)
// : offset(other.offset)
// {
//   layer = other.layer;
//   w_offset = other.w_offset;
//   b_offset = other.b_offset;
//   p_a_offset = other.p_a_offset;
// }

// OffsetManager& OffsetManager::operator= (const OffsetManager& other)
// {
//   // if (this == &other) return *this; // self-assignment check
//   return *this;
// }

// std::size_t OffsetManager::processOffset(std::size_t enum_code, std::size_t l)
// {
//   std::size_t _offset_;
//   switch (enum_code)
//   {
//   case 0:
//     _offset_ = w_offset + l;
//     break;
//   case 1:
//     _offset_ = b_offset + l;
//     break;
//   case 2:
//     _offset_ = p_a_offset + l;
//     break;
//   case 3:
//     _offset_ = a_offset + l;
//     break;

//   default:
//     _offset_ = l;
//     break;
//   }

//   return _offset_;
// }

// std::size_t OffsetManager::getCurrentLayerOffset(std::size_t enum_code)
// {
//   return processOffset(enum_code, layer);
// }

// std::size_t OffsetManager::getPreviousLayerOffset(std::size_t enum_code)
// {
//   return processOffset(enum_code, layer - 1);
// }

// std::size_t OffsetManager::getNextLayerOffset(std::size_t enum_code)
// {
//   return processOffset(enum_code, layer + 1);
// }

int main()
{

  // std::cout << rs(-2.92992998832987429837928394892739429283, 2) << std::endl;
  // std::size_t f = 1000;

  // for (int i = 0; i < f; ++ i)
  // {
  // double res = custom_rand(-1, 1, 5);
  // if (res > 0.5 & res < 0.6) std::cout << res << "\t";
  // }

  // for (std::size_t size = 10; size <= 1000000; size *= 10)
  // {
  //   std::vector<std::size_t> data(size);

  //   // Fill the vector with numbers from 0 to size-1
  //   std::iota(data.begin(), data.end(), 0);

  //   std::cout << "Benchmarking for vector size: " << size << std::endl;
  //   benchmark_shuffle(data);
  // }

  // std::cout << "\n\n\n\n";

  // for (std::size_t size = 10; size <= 1000000; size *= 10)
  // {
  //   std::size_t *data = (std::size_t*) malloc(size * __SIZEOF_SIZE_T__);

  //   // Fill the array with numbers from 0 to size-1
  //   createSpecArr(data, size);

  //   std::cout << "Benchmarking for array size: " << size << std::endl;
  //   benchmark_randSpec(data, size);

  //   // Free the array
  //   free(data);    
  // }

  std::cout << custom_rand(-1, 1, 5);
}
