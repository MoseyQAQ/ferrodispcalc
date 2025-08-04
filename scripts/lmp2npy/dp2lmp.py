import dpdata

sys = dpdata.LabeledSystem("8/sc8/Pb512Ti512O1536", 
                           fmt='deepmd/npy', 
                           type_map=["Pb", "Ti", "O"])

for idx, i in enumerate(sys):
    i.to("lmp",f'8/conf/{idx+1:d}.conf')