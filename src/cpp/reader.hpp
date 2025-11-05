// 这个头文件的作用：
// 1. MappedFile 类：封装跨平台的内存映射文件操作。
// 2. IReader 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <array>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "basic.hpp"
// mmap 和 C++17
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include <charconv> // For std::from_chars, C++17 fast integer parsing
#include <cstring>  // For memchr

/*   MappedFile 类：封装跨平台的内存映射文件操作。 */
class MappedFile {
private:
    const char* file_ptr_ = nullptr;
    size_t file_size_ = 0;

#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMapFile_ = NULL;
#else
    int fd_ = -1;
#endif

public:
    /**
     * @brief 构造函数，打开文件并将其映射到内存。
     * @param filename 要打开的文件路径。
     */
    MappedFile(const std::string& filename);

    /**
     * @brief 析构函数，解除内存映射并关闭文件句柄。
     */
    ~MappedFile();

    // --- 禁用拷贝和赋值 ---
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // --- 访问器 ---
    const char* data() const { return file_ptr_; }
    size_t size() const { return file_size_; }
    bool is_open() const { return file_ptr_ != nullptr; }

};

/*   IReader 接口：定义读取器的基本操作。 */
class IReader {
protected:
    // 指向内存映射的文件 (所有子类共享)
    const MappedFile& mapped_file_; 
    const char* file_ptr_;
    size_t file_size_;

    // 索引 (Requirement 1a)
    std::vector<size_t> frame_indices_; 
    
    // 原子数量 (假设在所有帧中不变，如果可变则需修改)
    int n_atoms_ = 0;

    /**
     * @brief 纯虚函数：子类必须实现这个函数，来填充 frame_indices_
     */
    virtual void build_index() = 0;

public:
    IReader(const MappedFile& file) 
        : mapped_file_(file), 
          file_ptr_(file.data()), 
          file_size_(file.size()) {}

    virtual ~IReader() {} // 虚析构函数

    // --- 公共 API ---
    size_t get_nframes() const {
        return frame_indices_.size();
    }

    int get_natoms() const {
        return n_atoms_;
    }

    /**
     * @brief 纯虚函数：子类必须实现这个函数，来解析特定的一帧
     */
    virtual Frame read_frame(size_t frame_index) = 0;
    virtual Frame read_first_frame() = 0;
    virtual Frame read_selected_frames(const std::vector<size_t>& frame_indices) = 0;
};

// some placeholders for specific reader implementations
class XYZReader : public IReader {

};

class LAMMPSReader : public IReader {

};