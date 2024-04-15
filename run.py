import h5py

with h5py.File("data/features.h5", 'r') as f:
    for key in f.keys():
        print(key)    
        for subkey in f[key].keys():
            print(subkey)
            for subsubkey in f[key][subkey].keys():
                print(subsubkey)
            exit()
    f.close()