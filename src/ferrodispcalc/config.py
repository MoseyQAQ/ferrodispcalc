'''
Used for common configuration, including type map and born effective charges.

Author: Denan Li
Last modified: 2024-07-15
Email: lidenan@westlake.edu.cn

Todo:
use data classes to manage the configuration.
'''

# Type Map: map the element to the int.
UniPero: list[str] = ['Ba','Pb','Ca','Sr','Bi',
            'K','Na','Hf','Ti','Zr',
            'Nb','Mg','In','Zn','O']
PSTO: list[str]=['Sr','Pb','Ti','O']

TypeMap = {
    'UniPero': UniPero,
    'PSTO': PSTO
}

# Commonly used Born effective charges for perovskites.
PTO = {
    'Pb': 3.44,
    'Ti': 5.18,
    'O': -(3.44+5.18)/3
}

BEC = {
    'PTO': PTO
}

# commonly used structure: PbTiO3, BaTiO3, SrTiO3, etc