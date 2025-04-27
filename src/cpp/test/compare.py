import sys
from com_sys import compare_stru, compare_disp, compare_polar
from pathlib import Path

ref_file = sys.argv[1]
test_file = sys.argv[2]
compare_type = sys.argv[3]

if compare_type == 'stru':
    status, distance = compare_stru(ref_file, test_file)
elif compare_type == 'disp':
    status, distance = compare_disp(ref_file, test_file)
elif compare_type == 'polar':
    status, distance = compare_polar(ref_file, test_file)
else:
    raise ValueError('Unknown compare type: %s' % compare_type)
    
folder_name = str(Path(test_file).absolute().parent.name)
test_file_name = str(Path(test_file).absolute().name)
ref_file_name = str(Path(ref_file).absolute().name)
print(f"{folder_name:<30} {test_file_name:<20} {ref_file_name:<10} {'Pass' if status else 'Fail':5} {distance:8.4f}")