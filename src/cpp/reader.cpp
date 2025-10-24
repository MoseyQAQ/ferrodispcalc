#include "reader.hpp"

MappedFile::MappedFile(const std::string& filename) {
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

MappedFile::~MappedFile() {
#ifdef _WIN32
        if (file_ptr_) UnmapViewOfFile(file_ptr_);
        if (hMapFile_) CloseHandle(hMapFile_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
        if (file_ptr_) {
            munmap((void*)file_ptr_, file_size_);
        }
        if (fd_ != -1) {
             // 理论上fd_在构造函数中已经关闭，但作为安全措施
            close(fd_);
        }
#endif
}