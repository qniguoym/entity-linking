import argparse
import logging
import numpy
import os
import time
import torch
from utils.utils import *
import faiss
logger = logging.getLogger()

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file):
        logger.info("Serializing index to %s", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file):
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: np.array):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once,"
                "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [
                np.hstack((doc_vector, aux_dims[i].reshape(-1, 1)))
                for i, doc_vector in enumerate(vectors)
            ]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self.index.add(hnsw_vectors)
            cnt += self.buffer_size
            logger.info("Indexed data %d" % cnt)

        logger.info("Total data indexed %d" % n)

    def search_knn(self, query_vectors, top_k):
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_k)
        return scores, indexes

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1


def main(params):
    output_path = params.output_path
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = get_logger(output_dir)

    logger.info("Loading candidate encoding from path: %s" % params.candidate_encoding)
    candidate_encoding = torch.load(params.candidate_encoding)
    vector_size = candidate_encoding.size(1)
    index_buffer = params.index_buffer

    logger.info("Using HNSW index in FAISS")
    index = DenseHNSWFlatIndexer(vector_size, index_buffer)


    logger.info("Building index.")
    index.index_data(candidate_encoding.numpy())
    logger.info("Done indexing data.")

    if params.get("save_index", None):
        index.serialize(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",default='./data/index/index', type=str)
    parser.add_argument(
        "--candidate_encoding",
        default="models/all_entities_large.t7",
        type=str,
        help="file path for candidte encoding.",
    )
    parser.add_argument(
        "--hnsw", action='store_true',
        help='If enabled, use inference time efficient HNSW index',
    )
    parser.add_argument(
        "--save_index", action='store_true',
        help='If enabled, save index',
    )
    parser.add_argument(
        '--index_buffer', type=int, default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )

    params = parser.parse_args()

    main(params)