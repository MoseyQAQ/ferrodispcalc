#include <vector>
#include <string>
#include <array>
#include <unordered_map>

using Lattice = std::array<std::array<double, 3>, 3>;

struct Frame
{
    
    Lattice lattice;
    std::vector<std::string> species; // element symbols
    std::vector<double> positions; // coord in flat array [x1,y1,z1,x2,y2,z2,...]

    // Additional properties
    std::unordered_map<std::string, std::vector<double>> atomic_properties; 
    std::unordered_map<std::string, double> frame_properties;
};
