from nncompress import EmbeddingCompressor
import numpy as np 

# Load my embedding matrix
matrix = np.load("data/skip_s100.npy")

# Initialize the compressor
compressor = EmbeddingCompressor(32, 16, "data/mymodel")

# Train the quantization model
compressor.train(matrix)

# Evaluate
distance = compressor.evaluate(matrix)
print("Mean euclidean distance:", distance)

# Export the codes and codebook
compressor.export(matrix, "data/mymodel")