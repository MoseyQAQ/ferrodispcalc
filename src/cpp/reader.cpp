#include "reader.hpp"
#include <stdexcept>
#include <cstring>      // For memchr, strncmp, strstr
#include <algorithm>    // For std::min, std::max
#include <iterator>     // For std::istream_iterator
#include <map>
#include "basic.hpp"
#include <charconv>     // For std::from_chars (C++17)

#ifdef _OPENMP
#include <omp.h>        // For OpenMP parallel read
#endif

// ----------------------------------------------------------------------------
// 辅助函数 (Helpers)
// ----------------------------------------------------------------------------

/**
 * @brief (私有) 查找从 ptr 开始的下一个换行符 ('\n')，
 * 并返回 *下一行* 的起始指针。
 * @param ptr 当前位置。
 * @param end_ptr mmap 内存的结束位置。
 * @return 指向下一行开头的 const char*。如果
 * 找不到换行符，则返回 end_ptr。
 */
static const char* find_next_line(const char* ptr, const char* end_ptr) {
    if (ptr >= end_ptr) {
        return end_ptr;
    }
    const char* line_end = (const char*)memchr(ptr, '\n', end_ptr - ptr);
    return (line_end == nullptr) ? end_ptr : line_end + 1;
}

// ----------------------------------------------------------------------------
// 构造函数 (在 .hpp 中已内联定义)
// ----------------------------------------------------------------------------
// LAMMPSReader::LAMMPSReader(const std::string& filename)
// (此实现已在 reader.hpp 中作为 inline 构造函数提供)
// {
//     mapped_file_ = std::make_unique<MappedFile>(filename_);
//     file_ptr_ = mapped_file_->data();
//     file_size_ = mapped_file_->size();
//     if (!mapped_file_->is_open() && file_size_ > 0) {
//         throw std::runtime_error("mmap failed for: " + filename);
//     }
//     if(file_ptr_ == nullptr && file_size_ > 0) {
//          throw std::runtime_error("mmap returned null pointer for: " + filename);
//     }
// }

LAMMPSReader::LAMMPSReader(const std::string& filename)
    : filename_(filename) {
    // Initialize memory-mapped file handle
    mapped_file_ = std::make_unique<MappedFile>(filename_);
    file_ptr_ = mapped_file_->data();
    file_size_ = mapped_file_->size();
    if (!mapped_file_->is_open() && file_size_ > 0) {
        throw std::runtime_error("mmap failed for: " + filename_);
    }
    if (file_ptr_ == nullptr && file_size_ > 0) {
        throw std::runtime_error("mmap returned null pointer for: " + filename_);
    }
}

// ----------------------------------------------------------------------------
// 索引实现 (Indexing Implementation)
// ----------------------------------------------------------------------------

void LAMMPSReader::index() {
    // 1. 清理现有索引
    frame_indices_.clear();
    column_names_.clear();
    column_map_.clear();
    n_atoms_ = -1;

    if (file_ptr_ == nullptr || file_size_ == 0) {
        // 文件为空，无需索引
        return;
    }

    // 2. 查找第一个 "ITEM: TIMESTEP"
    const char* TIMESTEP_KEY = "ITEM: TIMESTEP";
    const char* p_first_frame = (const char*)strstr(file_ptr_, TIMESTEP_KEY);

    if (p_first_frame == nullptr) {
        // 文件中没有帧
        return;
    }

    // 3. (仅一次) 解析第一个帧的元数据 (n_atoms, columns)
    const char* NATOMS_KEY = "ITEM: NUMBER OF ATOMS";
    const char* ATOMS_KEY = "ITEM: ATOMS";
    const char* end_ptr = file_ptr_ + file_size_;

    // 3a. 查找 "ITEM: NUMBER OF ATOMS"
    const char* p_natoms = (const char*)strstr(p_first_frame, NATOMS_KEY);
    if (p_natoms == nullptr) {
        throw std::runtime_error("Index: Cannot find 'ITEM: NUMBER OF ATOMS' after first timestep.");
    }
    // 3b. 获取下一行 (包含原子数)
    const char* natoms_line_start = find_next_line(p_natoms, end_ptr);
    if (natoms_line_start == end_ptr) {
        throw std::runtime_error("Index: Truncated file after 'NUMBER OF ATOMS'.");
    }
    // 3c. 解析 n_atoms
    auto [ptr, ec] = std::from_chars(natoms_line_start, find_next_line(natoms_line_start, end_ptr), n_atoms_);
    if (ec != std::errc() || n_atoms_ <= 0) {
        throw std::runtime_error("Index: Failed to parse valid n_atoms.");
    }

    // 3d. 查找 "ITEM: ATOMS"
    const char* p_atoms = (const char*)strstr(natoms_line_start, ATOMS_KEY);
    if (p_atoms == nullptr) {
        throw std::runtime_error("Index: Cannot find 'ITEM: ATOMS' after n_atoms.");
    }
    // 3e. 解析列名 (从 "ITEM: ATOMS" 所在的行)
    const char* atoms_line_end = find_next_line(p_atoms, end_ptr);
    std::string atoms_line(p_atoms, atoms_line_end); // "ITEM: ATOMS id type x y z ..."
    std::istringstream iss(atoms_line);
    std::string token;
    iss >> token; // "ITEM:"
    iss >> token; // "ATOMS"
    while (iss >> token) {
        column_names_.push_back(token);
    }
    if (column_names_.empty()) {
        throw std::runtime_error("Index: Failed to parse column names from 'ITEM: ATOMS' line.");
    }

    // 4. 构建列 -> 索引的映射
    build_column_map();

    // 5. 调用私有扫描器，从第一个帧开始快速跳转
    scan_mmap_for_index(p_first_frame);
}


void LAMMPSReader::scan_mmap_for_index(const char* start_ptr) {
    // 9 行:
    // 1. ITEM: TIMESTEP
    // 2. <timestep>
    // 3. ITEM: NUMBER OF ATOMS
    // 4. <n_atoms>
    // 5. ITEM: BOX BOUNDS ...
    // 6. <x_bounds>
    // 7. <y_bounds>
    // 8. <z_bounds>
    // 9. ITEM: ATOMS ...
    const int HEADER_LINES = 9;
    const int lines_per_frame = n_atoms_ + HEADER_LINES;

    const char* TIMESTEP_KEY = "ITEM: TIMESTEP";
    const size_t TIMESTEP_LEN = 15;

    const char* current_ptr = start_ptr;
    const char* end_ptr = file_ptr_ + file_size_;

    while (current_ptr < end_ptr) {
        // 1. 存储当前帧的起始位置
        frame_indices_.push_back(current_ptr - file_ptr_);

        // 2. 跳过 (n_atoms + 9) 行
        const char* jump_ptr = current_ptr;
        for (int i = 0; i < lines_per_frame; ++i) {
            jump_ptr = find_next_line(jump_ptr, end_ptr);
            if (jump_ptr == end_ptr) break; // 到达文件末尾
        }
        
        // 3. 检查跳转后的位置是否是下一帧
        if (jump_ptr < end_ptr) {
             // 检查新位置是否确实是 "ITEM: TIMESTEP"
             if ((end_ptr - jump_ptr >= TIMESTEP_LEN) && 
                 strncmp(jump_ptr, TIMESTEP_KEY, TIMESTEP_LEN) == 0) {
                 // 成功，设置下一次循环的起始点
                 current_ptr = jump_ptr;
             } else {
                 // 发生错误或文件损坏 (帧不完整)
                 // 我们尝试通过搜索下一个 "ITEM: TIMESTEP" 来重新同步
                 current_ptr = (const char*)strstr(jump_ptr, TIMESTEP_KEY);
                 if (current_ptr == nullptr) {
                     break; // 找不到了，索引结束
                 }
             }
        } else {
            // 到达文件末尾
            current_ptr = jump_ptr;
        }
    }
}


void LAMMPSReader::build_column_map() {
    column_map_.clear();
    for (int i = 0; i < column_names_.size(); ++i) {
        column_map_[column_names_[i]] = i;
    }
}

// ----------------------------------------------------------------------------
// 缓存 I/O (Cache I/O)
// ----------------------------------------------------------------------------

void LAMMPSReader::save_index(const std::string& cache_filename) {
    std::ofstream out(cache_filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open cache file for writing: " + cache_filename);
    }

    // 1. 写入 n_atoms
    out.write(reinterpret_cast<const char*>(&n_atoms_), sizeof(n_atoms_));

    // 2. 写入列名
    size_t num_cols = column_names_.size();
    out.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));
    for (const auto& name : column_names_) {
        size_t len = name.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(name.c_str(), len);
    }

    // 3. 写入帧索引
    size_t num_frames = frame_indices_.size();
    out.write(reinterpret_cast<const char*>(&num_frames), sizeof(num_frames));
    out.write(reinterpret_cast<const char*>(frame_indices_.data()), num_frames * sizeof(long long));

    if (!out) {
         throw std::runtime_error("Error occurred while writing to cache file: " + cache_filename);
    }
}

void LAMMPSReader::load_index(const std::string& cache_filename) {
    std::ifstream in(cache_filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open cache file for reading: " + cache_filename);
    }

    // 清理
    frame_indices_.clear(); 
    column_names_.clear(); 
    column_map_.clear(); 
    n_atoms_ = -1;

    // 1. 读取 n_atoms
    in.read(reinterpret_cast<char*>(&n_atoms_), sizeof(n_atoms_));
    if (in.fail()) throw std::runtime_error("Cache read error: n_atoms");

    // 2. 读取列名
    size_t num_cols = 0;
    in.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    if (in.fail()) throw std::runtime_error("Cache read error: num_cols");
    
    column_names_.resize(num_cols);
    std::vector<char> name_buffer;
    for (size_t i = 0; i < num_cols; ++i) {
        size_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (in.fail()) throw std::runtime_error("Cache read error: col_len");

        if (len > 1024) throw std::runtime_error("Cache read error: unreasonable column length");
        name_buffer.resize(len);
        in.read(name_buffer.data(), len);
        if (in.fail()) throw std::runtime_error("Cache read error: col_data");
        
        column_names_[i] = std::string(name_buffer.data(), len);
    }
    
    // 3. 读取帧索引
    size_t num_frames = 0;
    in.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
    if (in.fail()) throw std::runtime_error("Cache read error: num_frames");

    frame_indices_.resize(num_frames);
    in.read(reinterpret_cast<char*>(frame_indices_.data()), num_frames * sizeof(long long));
    
    if (in.fail() && !in.eof()) {
        // eof 可能是 0 帧文件，但其他 fail 是错误的
         throw std::runtime_error("Cache read error: frame_indices data");
    }

    // 4. 重建辅助 map
    build_column_map();
}


// ----------------------------------------------------------------------------
// 帧读取 (Frame Reading)
// ----------------------------------------------------------------------------

std::vector<Frame> LAMMPSReader::read_frames(const std::vector<size_t>& frame_indices_idx) {
    std::vector<Frame> frames(frame_indices_idx.size());
    
    // 1. (串行) 边界检查
    for (size_t idx : frame_indices_idx) {
        if (idx >= frame_indices_.size()) {
            throw std::out_of_range(
                "Frame index " + std::to_string(idx) + 
                " is out of bounds (total frames: " + 
                std::to_string(frame_indices_.size()) + ")."
            );
        }
    }

    // 2. (并行) 解析
    // 启用 schedule(dynamic) 是因为某些帧（例如最后一帧）
    // 可能比其他帧稍大或稍小（如果文件截断），
    // 并且解析是主要瓶颈。
    // OpenMP on MSVC requires a signed integer loop variable. Convert the
    // unsigned size to a signed int for the loop index. We still use
    // frame_indices_idx[i] as size_t when indexing the vector to avoid
    // narrowing issues on platforms where size_t is larger than int.
    int n_frames_to_read = static_cast<int>(frame_indices_idx.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n_frames_to_read; ++i) {
        size_t frame_idx = frame_indices_idx[static_cast<size_t>(i)];
        
        // 获取帧的 [start, end) 字节范围
        long long start_pos = frame_indices_[frame_idx];
        long long end_pos = (frame_idx + 1 < frame_indices_.size()) 
                            ? frame_indices_[frame_idx + 1] 
                            : file_size_;

        // 调用解析器
        // (parse_frame_chunk 必须是线程安全的，
        //  因为它是无状态的并且只操作传入的指针，所以它是安全的)
        frames[i] = parse_frame_chunk(file_ptr_ + start_pos, file_ptr_ + end_pos);
    }

    return frames;
}


Frame LAMMPSReader::parse_frame_chunk(const char* frame_start, const char* frame_end) {
    // 这是一个线程安全的函数。
    // 它只读取 [frame_start, frame_end) 范围内的内存。
    // 并且只写入它在栈上创建的 'frame' 对象。
    
    Frame frame;
    frame.n_atoms = n_atoms_; // 从类成员获取
    
    // 1. 将 mmap 块复制到 stringstream 中。
    // 这是 (极快的) 内存到内存复制。
    std::string buffer(frame_start, frame_end);
    std::istringstream iss(buffer);
    std::string line;

    // --- 2. 解析 TIMESTEP ---
    if (!std::getline(iss, line) || line.find("ITEM: TIMESTEP") == std::string::npos) {
        throw std::runtime_error("Parser: 'ITEM: TIMESTEP' not found.");
    }
    if (!std::getline(iss, line)) throw std::runtime_error("Parser: truncated at timestep value.");
    frame.timestep = std::stoi(line);

    // --- 3. 解析 NUMBER OF ATOMS ---
    if (!std::getline(iss, line) || line.find("ITEM: NUMBER OF ATOMS") == std::string::npos) {
        throw std::runtime_error("Parser: 'ITEM: NUMBER OF ATOMS' not found.");
    }
    if (!std::getline(iss, line)) throw std::runtime_error("Parser: truncated at n_atoms value.");
    // 可选: assert(std::stoi(line) == n_atoms_);

    // --- 4. 解析 BOX BOUNDS (借鉴 basic.hpp::read_cell) ---
    if (!std::getline(iss, line) || line.find("ITEM: BOX BOUNDS") == std::string::npos) {
        throw std::runtime_error("Parser: 'ITEM: BOX BOUNDS' not found.");
    }
    
    std::vector<double> bounds;
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(iss, line)) throw std::runtime_error("Parser: truncated at box bounds.");
        std::istringstream line_ss(line);
        std::vector<double> lineData((std::istream_iterator<double>(line_ss)), std::istream_iterator<double>());
        if (lineData.size() == 2) {
            lineData.push_back(0.0); // 处理正交 (非三斜) 盒子
        }
        if (lineData.size() != 3) throw std::runtime_error("Parser: Invalid box bounds line.");
        bounds.insert(bounds.end(), lineData.begin(), lineData.end());
    }

    double xlo_bound = bounds[0], xhi_bound = bounds[1], xy = bounds[2];
    double ylo_bound = bounds[3], yhi_bound = bounds[4], xz = bounds[5];
    double zlo_bound = bounds[6], zhi_bound = bounds[7], yz = bounds[8];

    // (来自 basic.hpp) 确保三斜晶系边界的正确处理
    double xlo = xlo_bound - std::min({0.0, xy, xz, xy + xz});
    double xhi = xhi_bound - std::max({0.0, xy, xz, xy + xz});
    double ylo = ylo_bound - std::min(0.0, yz);
    double yhi = yhi_bound - std::max(0.0, yz);

    frame.cell[0][0] = xhi - xlo;
    frame.cell[0][1] = 0.0;
    frame.cell[0][2] = 0.0;
    frame.cell[1][0] = xy;
    frame.cell[1][1] = yhi - ylo;
    frame.cell[1][2] = 0.0;
    frame.cell[2][0] = xz;
    frame.cell[2][1] = yz;
    frame.cell[2][2] = zhi_bound - zlo_bound;

    // --- 5. 解析 ATOMS ---
    if (!std::getline(iss, line) || line.find("ITEM: ATOMS") == std::string::npos) {
        throw std::runtime_error("Parser: 'ITEM: ATOMS' not found.");
    }
    
    // --- 6. 准备向量 ---
    frame.positions.resize(n_atoms_*3);
    frame.types.resize(n_atoms_);
    
    // 查找关键列的索引 (一次)
    // 注意: column_map_ 是 const，因此在 OMP 中读取是安全的
    int type_col = column_map_.at("type");
    int x_col = column_map_.count("x") ? column_map_.at("x") : column_map_.at("xu");
    int y_col = column_map_.count("y") ? column_map_.at("y") : column_map_.at("yu");
    int z_col = column_map_.count("z") ? column_map_.at("z") : column_map_.at("zu");
    
    // 为其他属性分配空间
    std::map<std::string, int> other_prop_cols;
    for (const auto& pair : column_map_) {
        const std::string& name = pair.first;
        if (name != "id" && name != "type" && name != "x" && name != "y" && name != "z" &&
            name != "xu" && name != "yu" && name != "zu") {
            frame.arrays[name].resize(n_atoms_);
            other_prop_cols[name] = pair.second; // 存储列索引
        }
    }

    // --- 7. 循环读取 n_atoms 行 (性能关键点) ---
    std::vector<double> line_data;
    line_data.reserve(column_names_.size());

    for (int i = 0; i < n_atoms_; ++i) {
        if (!std::getline(iss, line)) {
            // 文件可能已损坏或截断
            throw std::runtime_error(
                "Parser: truncated at atom data. Timestep " + 
                std::to_string(frame.timestep) + ", expected " + 
                std::to_string(n_atoms_) + " atoms, got " + std::to_string(i));
        }
        std::istringstream atom_ss(line);
        
        // (借鉴 basic.hpp::read_coords) 将行读入 vector
        line_data.clear();
        double val;
        while(atom_ss >> val) {
            line_data.push_back(val);
        }
        
        if (line_data.size() < column_names_.size()) {
             throw std::runtime_error("Parser: Atom line has fewer columns (" + 
                std::to_string(line_data.size()) + ") than header (" +
                std::to_string(column_names_.size()) + ").");
        }
        
        // 存储数据 (按文件中的顺序，即索引 i)
        frame.types[i] = static_cast<int>(line_data[type_col]);
        frame.positions[i * 3 + 0] = line_data[x_col];
        frame.positions[i * 3 + 1] = line_data[y_col];
        frame.positions[i * 3 + 2] = line_data[z_col];

        // 存储其他属性
        for (const auto& prop : other_prop_cols) {
            frame.arrays[prop.first][i] = line_data[prop.second];
        }
    }
    
    return frame;
}