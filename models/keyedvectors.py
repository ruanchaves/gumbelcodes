#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matutils
import marisa_trie
from six import string_types

class CompressedVectors(object):
    def __init__(self, vectors, codebook):
        self.codebook = np.load(codebook)
        self.codebook_vectors_odd = self.codebook[0] 
        self.codebook_vectors_even = self.codebook[1]
        self.vectors_odd = np.load(vectors) << 4
        self.vectors_even = np.load(vectors) & 0xF

    def __getitem__(self, index):
        odd = matutils.matrix_vector_multiply(self.codebook_vectors_odd, self.vectors_odd[index])
        even = matutils.matrix_vector_multiply(self.codebook_vectors_even, self.vectors_even[index])
        return np.sum(np.concatenate((odd, even), axis=0), axis=0)

class BaseKeyedVectors(object):
    """Abstract base class / interface for various types of word vectors."""
    def __init__(self, *args, **kwargs):
        self.vectors = None
        self.codebook = None
        self.trie = None

        self.__dict__.update(kwargs)
        self.vectors = CompressedVectors(self.vectors, self.codebook)
        self.vocab = marisa_trie.Trie().load(self.trie)

    @classmethod
    def load(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def get_vector(self, entity):
        """Get the entity's representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        entity : str
            Identifier of the entity to return the vector for.

        Returns
        -------
        numpy.ndarray
            Vector for the specified entity.

        Raises
        ------
        KeyError
            If the given entity identifier doesn't exist.

        """
        if entity in self.vocab:
            result = self.vectors[self.vocab[entity]]
            return result
        else:
            raise KeyError("'%s' not in vocabulary" % entity)

    def __getitem__(self, entities):
        """Get vector representation of `entities`.

        Parameters
        ----------
        entities : {str, list of str}
            Input entity/entities.

        Returns
        -------
        numpy.ndarray
            Vector representation for `entities` (1D if `entities` is string, otherwise - 2D).

        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return np.vstack([self.get_vector(entity) for entity in entities])

    def __contains__(self, entity):
        return entity in self.vocab


class WordEmbeddingsKeyedVectors(BaseKeyedVectors):
    """Class containing common methods for operations over word vectors."""
    def __init__(self, *args, **kwargs):
        super(WordEmbeddingsKeyedVectors, self).__init__(*args, **kwargs)

    def __contains__(self, word):
        return word in self.vocab

    def word_vec(self, word, use_norm=False):
        if word in self.vocab:
            result = self.vectors[self.vocab[word]]
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def get_vector(self, word):
        return self.word_vec(word)

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """Compute cosine similarities between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.ndarray
            Vector from which similarities are to be computed, expected shape (dim,).
        vectors_all : numpy.ndarray
            For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.ndarray
            Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = np.dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words=()):
        """Compute cosine distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : {str, numpy.ndarray}
            Word or vector from which distances are to be computed.
        other_words : iterable of str
            For each word in `other_words` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`.

        Raises
        -----
        KeyError
            If either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.vocab[word].index for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        """Compute cosine distance between two words.
        Calculate 1 - :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        float
            Distance between `w1` and `w2`.

        """
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        """Compute cosine similarity between two words.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        float
            Cosine similarity between `w1` and `w2`.

        """
        return np.dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """Compute cosine similarity between two sets of words.

        Parameters
        ----------
        ws1 : list of str
            Sequence of words.
        ws2: list of str
            Sequence of words.

        Returns
        -------
        numpy.ndarray
            Similarities between `ws1` and `ws2`.

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))

class Word2VecKeyedVectors(WordEmbeddingsKeyedVectors):

    @classmethod
    def load_word2vec_format(cls, *args, **kwargs):
        model = super(Word2VecKeyedVectors, cls).load(*args, **kwargs)
        return model

    @classmethod
    def load(cls, fname_or_handle, *args, **kwargs):
        model = super(Word2VecKeyedVectors, cls).load(*args, **kwargs)
        return model


KeyedVectors = Word2VecKeyedVectors  # alias for backward compatibility