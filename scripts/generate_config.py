import json
config = []
with open('filelist_example.txt','r') as f:
    for line in f:
        # Large embeddings don't work yet (disregard > 300)
        # All embeddings compress to the same size (disregard < 300)
        if not '300' in line:
            continue
        
        source_path = line.strip()
        dimension = int(line.strip().split('_')[-1].lstrip('s').rstrip('.txt'))
        
        if 'glove' in line:
            name = line.strip().split('/')[-1]
        else:
            name = line.strip().split('/')[-2] + '_' + line.strip().split('/')[-1]
        target_path = 'data/' + name.rstrip('.txt') + '.npy'

        dictionary_path = 'data/' + name.rstrip('.txt') + '_dictionary.json'
        trie_path = 'data/' + name.rstrip('.txt') + '_trie.marisa'

        codebook_prefix = 'data/' + name.rstrip('.txt') + '_'
        logging = True

        entry = {
            "source_path" : source_path,
            "dimension": dimension,
            "target_path": target_path,
            "dictionary_path": dictionary_path,
            "trie_path": trie_path,
            "codebook_prefix": codebook_prefix,
            "logging": logging
        }
        config.append(entry)

with open('../settings.json','w+') as f:
    json.dump(config, f)