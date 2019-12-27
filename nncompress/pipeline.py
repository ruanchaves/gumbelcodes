from __future__ import absolute_import, division, print_function

import json
import os
import sys
from argparse import ArgumentParser
import codecs
import numpy as np
from gumbelcodes.nncompress.embed_compress import EmbeddingCompressor
from loguru import logger


class PipelineTemplate(object):

    def __init__(self,
                 codebook_prefix=None,
                 codebook_suffix='.codebook.npy',
                 codes_suffix='.codes',
                 dimension=None,
                 source_path=None,
                 target_path=None,
                 words_suffix='.words',
                 logging=True):
        self.codebook_prefix = codebook_prefix
        self.codebook_suffix = codebook_suffix
        self.codes_suffix = codes_suffix
        self.dimension = dimension
        self.logging = logging
        self.source_path = source_path
        self.target_path = target_path
        self.words_suffix = words_suffix


class Pipeline(PipelineTemplate):

    def __init__(self, **kwargs):
        super(Pipeline, self).__init__(**kwargs)
        self.codebook_path = self.codebook_prefix + self.codebook_suffix
        self.codes_path = self.codebook_prefix + self.codes_suffix
        if self.logging:
            logger.debug("Source: {0} \n Target: {1}".format(
                self.source_path, self.target_path))

    def get_embeddings(self, get_words=True, ignore_errors=True):
        lines = list(open(self.source_path, encoding="utf-8"))
        embed_matrix = np.zeros((len(lines), self.dimension), dtype='float32')
        words = []
        if self.logging:
            logger.debug("Embedding shape: {0}".format(embed_matrix.shape))
        for i, line in enumerate(lines):
            parts = line.strip().split()
            try:
                w = parts[0]
                vec = np.array(parts[1:], dtype='float32')
                embed_matrix[i] = vec
                if get_words:
                    words.append(w)
            except Exception as e:
                if not ignore_errors:
                    raise Exception("Invalid embedding at line {0}".format(i))
                if self.logging:
                    logger.debug(e)
                    logger.debug(w)
                    logger.debug("Invalid embedding at line {0}".format(i))
        np.save(self.target_path, embed_matrix)
        if get_words:
            fname = self.codebook_prefix + self.words_suffix
            open(fname, 'w+').close()
            with codecs.open(fname, 'w', encoding='utf8') as f:
                for word in words:
                    print(word, file=f)

    def train(self):
        matrix = np.load(self.target_path)
        compressor = EmbeddingCompressor(32, 16, self.codebook_prefix)
        compressor.train(matrix)
        distance = compressor.evaluate(matrix)
        if self.logging:
            logger.debug("Mean euclidean distance:", distance)
        compressor.export(matrix, self.codebook_prefix)