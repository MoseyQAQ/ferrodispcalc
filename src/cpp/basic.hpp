#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <charconv> // For std::from_chars, C++17 fast integer parsing
#include <cstring>  // For memchr
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

using Lattice = std::array<std::array<double, 3>, 3>;

struct FrameData {
    // 晶格 (Requirement 1b)
    Lattice lattice;

    // 原子信息 (Requirement 1c)
    std::vector<std::string> species; // 元素种类
    std::vector<double> positions;  // 展平的坐标 (size = N * 3)

    // 自由Key (用于'ase.Atoms.arrays')
    // e.g., "forces" -> [fx1, fy1, fz1, fx2, ...]
    // e.g., "dipole" -> [dx1, dy1, dz1, dx2, ...]
    std::unordered_map<std::string, std::vector<double>> atomic_properties;

    // 自由Key (用于'ase.Atoms.info')
    // e.g., "energy" -> 123.45
    // e.g., "step"   -> 10000
    std::unordered_map<std::string, double> frame_properties;
};

// --- 1. MappedFile RAII 封装类 (新) ---
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
    bool is_open() const { return file_ptr_ != nullptr && file_ptr_ != MAP_FAILED; }
};

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
    virtual FrameData read_frame(size_t frame_index) = 0;
};