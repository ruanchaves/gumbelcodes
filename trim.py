import numpy as np
import os
import sys
import json 

def encode(x,y):
    return (x << 4) + y

def create_file(original_path, prefix=''):
    path = os.path.split(original_path)[0]
    fname = os.path.split(original_path)[1]
    fname = prefix + fname
    return os.path.join(path, fname)

def trim_codebook(codes_path):
    codes = np.loadtxt(codes_path)
    codes = codes.astype(dtype=np.int64)
    max_value = np.max(codes.flatten())

    assert(not [x for x in codes if len(x) != len(codes[0])])
    assert(max_value < len(codes[0]) )
    print(max_value)
    new_codes = np.array([ np.bincount(x, minlength=(max_value+1)) for x in codes ], dtype=np.uint8)
    max_rep = np.max(new_codes.flatten())
    print(max_rep)

    new_codes = [ [ x[idx:idx+2] for idx in range(0,len(x), 2) ] for x in new_codes ]
    new_codes = [ np.array([ encode(*y) for y in x ],dtype=np.uint8) for x in new_codes ]
    new_codes = np.array(new_codes, dtype=np.uint8)

    np.save(create_file(codes_path, 'trimmed_'), new_codes)

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        settings = json.load(f)
    for item in settings:
        codebook = item['codebook_prefix'] + '.codebook.npy'
        codes = item['codebook_prefix'] + '.codes'
        trim_codebook(codes)