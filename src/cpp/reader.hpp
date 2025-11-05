// 这个头文件的作用：
// 1. MappedFile 类：封装跨平台的内存映射文件操作。
// 2. IReader 
#pragma once
#include <iostream>
#include <fstream>
#include <memory>
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
#ifndef NOMINMAX
#define NOMINMAX
#endif
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

// -------------------------------------------------------------
// 1. MappedFile RAII 封装类 (来自你的 XYZ 示例)
// -------------------------------------------------------------
/**
 * @brief 使用 RAII 封装内存映射文件 (mmap)，确保资源自动释放。
 */
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
    MappedFile(const std::string& filename) {
#ifdef _WIN32
        hFile_ = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Error: Could not open file (Windows).");
        }

        LARGE_INTEGER size_li;
        if (!GetFileSizeEx(hFile_, &size_li)) {
            CloseHandle(hFile_);
            throw std::runtime_error("Error: Could not get file size (Windows).");
        }
        file_size_ = size_li.QuadPart;
        
        // 句柄检查：如果文件大小为0，则不创建映射
        if (file_size_ == 0) {
            CloseHandle(hFile_);
            // 允许空文件，但 file_ptr_ 将为 nullptr
            return;
        }

        hMapFile_ = CreateFileMapping(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapFile_ == NULL) {
            CloseHandle(hFile_);
            throw std::runtime_error("Error: Could not create file mapping (Windows).");
        }

        file_ptr_ = (const char*)MapViewOfFile(hMapFile_, FILE_MAP_READ, 0, 0, file_size_);
        if (file_ptr_ == NULL) {
            CloseHandle(hMapFile_);
            CloseHandle(hFile_);
            throw std::runtime_error("Error: Could not map view of file (Windows).");
        }
#else // POSIX
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("Error: Could not open file (POSIX).");
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close(fd_);
            throw std::runtime_error("Error: Could not get file size (POSIX).");
        }
        file_size_ = sb.st_size;

        // 句柄检查：如果文件大小为0，则不创建映射
        if (file_size_ == 0) {
            close(fd_);
            fd_ = -1;
            // 允许空文件，但 file_ptr_ 将为 nullptr
            return;
        }

        file_ptr_ = (const char*)mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (file_ptr_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Error: Could not map file to memory (POSIX).");
        }
        // mmap后可以立即关闭fd
        close(fd_);
        fd_ = -1;
#endif
    }

    /**
     * @brief 析构函数，解除内存映射并关闭文件句柄。
     */
    ~MappedFile() {
#ifdef _WIN32
        if (file_ptr_) UnmapViewOfFile(file_ptr_);
        if (hMapFile_) CloseHandle(hMapFile_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
        if (file_ptr_ && file_ptr_ != MAP_FAILED) {
            munmap((void*)file_ptr_, file_size_);
        }
        if (fd_ != -1) {
             // 理论上fd_在构造函数中已经关闭，但作为安全措施
            close(fd_);
        }
#endif
    }

    // --- 禁用拷贝和赋值 ---
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // --- 访问器 ---
    const char* data() const { return file_ptr_; }
    size_t size() const { return file_size_; }
    bool is_open() const { return file_ptr_ != nullptr; }
};

// -------------------------------------------------------------
// 3. Reader 接口 (IReader)
// -------------------------------------------------------------

/**
 * @brief Reader 接口
 * 定义了所有 Reader (XYZ, LAMMPS) 都必须实现的功能。
 */
class IReader {
public:
    virtual ~IReader() = default;

    /**
     * @brief 核心功能：索引整个文件。
     * 必须在任何 read_* 操作之前调用 (除非已 load_index)。
     * 负责填充内部的 frame_indices_ 向量。
     */
    virtual void index() = 0;

    /**
     * @brief 核心功能：读取指定索引的多个帧。
     * @param frame_indices 要读取的帧的索引 (基于0)。
     * @return 包含所请求帧的数据列表。
     */
    virtual std::vector<Frame> read_frames(
        const std::vector<size_t>& frame_indices
    ) = 0;

    // --- Getters ---

    /** @brief 获取索引到的总帧数 */
    virtual size_t get_num_frames() const = 0;

    /** @brief 获取（第一帧的）原子数 */
    virtual int get_num_atoms() const = 0;
};


// -------------------------------------------------------------
// 4. LAMMPSReader 完整声明
// -------------------------------------------------------------

/**
 * @brief 读取 LAMMPS dump 文件的具体实现。
 * 使用 mmap 高效处理，并支持文件索引缓存。
 */
class LAMMPSReader : public IReader {
private:
    std::string filename_;
    std::unique_ptr<MappedFile> mapped_file_; // mmap RAII 句柄
    size_t file_size_ = 0;
    const char* file_ptr_ = nullptr; // 指向 mmap 内存的开头

    // 索引数据 (将被序列化)
    std::vector<long long> frame_indices_;  // 每一帧 "ITEM: TIMESTEP" 的起始偏移量
    int n_atoms_ = -1;                      // 原子数
    std::vector<std::string> column_names_; // "ITEM: ATOMS" 后面的列名
    
    // 辅助 map，用于快速查找列索引
    // e.g., "x" -> 2, "y" -> 3, "fx" -> 5
    std::map<std::string, int> column_map_; 

    /**
     * @brief (私有) 内部函数：解析内存中的单个帧数据块。
     * @param frame_start 指向该帧在 mmap 内存中的起始指针。
     * @param frame_end 指向该帧的结束指针 (即下一帧的开始)。
     * @return 解析后的 Frame 结构体。
     */
    Frame parse_frame_chunk(const char* frame_start, const char* frame_end);

    /**
     * @brief (私有) 内部函数：真正的 mmap 扫描逻辑。
     * 被 index() 调用，负责填充 `frame_indices_`, `n_atoms_`, `column_names_`。
     * @param start_ptr 指向第一个帧的起始位置（在 mmap 内存中）
     */
    void scan_mmap_for_index(const char* start_ptr);

    /**
     * @brief (私有) 内部函数：根据 `column_names_` 填充 `column_map_`。
     */
    void build_column_map();

public:
    /**
     * @brief 构造函数 (V2)。
     * 立即 mmap 文件，但 *不* 索引。
     * @param filename 要打开的 LAMMPS dump 文件。
     * @throws std::runtime_error 如果 mmap 失败。
     */
    LAMMPSReader(const std::string& filename);

    /**
     * @brief 析构函数 (默认)。
     */
    ~LAMMPSReader() = default;

    /**
     * @brief (1c) 执行 mmap 和文件索引。
     * 清理任何现有的索引数据，并重新扫描 mmap 内存。
     */
    void index() override;

    /**
     * @brief (新增) 将计算出的索引序列化到缓存文件。
     * @param cache_filename 要写入的缓存文件路径。
     * @throws std::runtime_error 如果写入失败。
     */
    void save_index(const std::string& cache_filename);

    /**
     * @brief (新增) 从缓存文件反序列化索引。
     * @param cache_filename 要读取的缓存文件路径。
     * @throws std::runtime_error 如果读取失败或格式错误。
     */
    void load_index(const std::string& cache_filename);


    /**
     * @brief (1d) 并行读取多个帧。
     * @param frame_indices 要读取的帧的索引 (基于0)。
     * @return 包含所请求帧的数据列表。
     * @throws std::runtime_error 如果索引越界或解析失败。
     */
    std::vector<Frame> read_frames(
        const std::vector<size_t>& frame_indices
    ) override;

    // --- Getters ---
    
    /** @brief 获取索引到的总帧数 */
    size_t get_num_frames() const override { return frame_indices_.size(); }
    
    /** @brief 获取（第一帧的）原子数 */
    int get_num_atoms() const override { return n_atoms_; }
};