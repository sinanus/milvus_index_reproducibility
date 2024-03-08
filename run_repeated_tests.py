

import random
import numpy as np
import uuid
import pickle
import datetime
import itertools
from more_itertools import batched


from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Partition
)



def gen_vectors(dim, num):
    return [[random.randint(0, 255) for _ in range(dim)] for _ in range(num)]

# gen binary vectors
def gen_binary_vectors(dim, num):
    # in binary vectors, a bit represent a dimension.
    # a value of `uint8` describe 8 dimension
    vectors = gen_vectors(dim // 8, num)
    return vec_to_binary(vectors)

def vec_to_binary(vectors):
    data = np.array(vectors, dtype='uint8').astype("uint8")
    return [bytes(vector) for vector in data]



# Const names
_COLLECTION_NAME = 'Reproducibility_Test'
_EMBEDDING_ID_FIELD_NAME = 'embedding_id'
_ENTITY_ID_FIELD_NAME = 'entity_id'
_VECTOR_FIELD_NAME = 'binary_vector_field'

# Vector parameters
_DIM = 256

# Index parameters
_METRIC_TYPE = 'JACCARD'
_INDEX_TYPE = 'BIN_IVF_FLAT'
_NLIST = 1000
_NPROBE = 5
_TOPK = 3

def create_connection():
    connections.connect(
    alias="default",
    host='localhost',
    port='19530')


# Create a collection named 'demo'
def create_collection(name, embedding_id, id_field, vector_field):
    field0 = FieldSchema(name=embedding_id, dtype=DataType.VARCHAR, max_length=36, description="embedding_id", is_primary=True)
    field1 = FieldSchema(name=id_field, dtype=DataType.VARCHAR, max_length=36, description="entity_id", is_primary=False)
    field2 = FieldSchema(name=vector_field, dtype=DataType.BINARY_VECTOR, description="binary vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field0, field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema)
    partition = Partition(collection=collection, name='partition_'+name)
    print("\ncollection created:", name)
    print("\npartition created:", 'partition_'+name)
    return collection, partition

def create_collection_single_non_vector_field(name, embedding_id, vector_field):
    field0 = FieldSchema(name=embedding_id, dtype=DataType.VARCHAR, max_length=36, description="embedding_id", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.BINARY_VECTOR, description="binary vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field0, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema)
    partition = Partition(collection=collection, name='partition_'+name)
    print("collection created:", name)
    print("partition created:", 'partition_'+name)
    return collection, partition

def has_collection(name):
    return utility.has_collection(name)

# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("Drop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("list collections:")
    print(utility.list_collections())





def get_entity_num(collection):
    print("The number of entity:")
    print(collection.num_entities)


def create_index(collection, filed_name):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("Created index: {}".format(collection.index().params))


def drop_index(collection):
    collection.drop_index()
    print("Drop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()





def set_ttl_seconds(collection, val):
    collection.set_properties(properties={"collection.ttl.seconds": val})

def insert(collection, num, dim):
    data = [range(0,num),
            gen_binary_vectors(dim, num)]
    collection.insert(data)
    return data[1]

def insert_embeddings(collection, data):
    # data = [
    #     [i for i in range(num)],
    #     [[random.random() for _ in range(dim)] for _ in range(num)],
    # ]
    collection.insert(data)
    return data[1]

def search(collection, vector_field,  search_vectors, print_results=False):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "nprobe": _NPROBE},
        "limit": _TOPK}
    results = collection.search(**search_param)
    if print_results:
        for i, result in enumerate(results):
            print("\nSearch result for {}th vector: ".format(i))
            for j, res in enumerate(result):
                print("Top {}: {}".format(j, res))
    return results

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

if __name__ == "__main__":


    # with open('all_fingerprints_extract_and_store_test_bdeba22d-3b0a-4ffd-81f2-1cb6fabafc1e_descriptors.npy','rb') as f:
    #     vectors = np.load(f)
    vectors = gen_vectors(_DIM//8, int(2e6))
    data = np.array(vectors, dtype='uint8').astype("uint8")
    # data = data[:1200000]
    # convert to bytes
    data_bytes = [bytes(vector) for vector in data]
    N = int(len(data_bytes) * 0.9)
    vectors_to_index = data_bytes[:N]
    vectors_to_search = data_bytes[N:]

    uuids_for_embeddings = [str(uuid.uuid4()) for i in range(len(vectors_to_index))]

    embeddings_per_entity_vals = [1]
    N_insert_batch_size = 50000
    N_search_batch_size = 500
    single_non_vector_field_collection_vals = [False,True]
    N_index_rebuild = 3
    N_same_index_search_repeat = 2



    # all_test_results = []
    for single_non_vector_field in single_non_vector_field_collection_vals:
        for embeddings_per_entity in embeddings_per_entity_vals:
            vectors_to_index_batched = list(batched(vectors_to_index, embeddings_per_entity))
            entity_uuids_batched = [[str(uuid.uuid4())]*len(v) for v in vectors_to_index_batched]
            vectors_to_index_flat = flatten_extend(vectors_to_index_batched)
            entity_uuids_batched_flat =  flatten_extend(entity_uuids_batched)

            test_uuid = uuid.uuid4()
            # pickle.dump(uuids,open(f'uuids_{test_uuid}.pickle','wb'))
            
            test_metadata = {'uuid':test_uuid,
                            'single_non_vector_field':single_non_vector_field,
                            'embeddings_per_entity':embeddings_per_entity,
                            'number_vectors_per_insert_batch':N_insert_batch_size,
                            'number_vectors_per_search_batch':N_search_batch_size,
                            'number_index_rebuild':N_index_rebuild,
                            'number_same_index_search_repeat':N_same_index_search_repeat}
            print('\n\n')
            print('#'*10)
            print('Test conditions:')
            print(test_metadata)
            print('#'*10)

            results = []
            for i in range(N_index_rebuild):
                print('#'*5 + f' {i+1} of {N_index_rebuild} Index rebuilds' + '#'*5)

                create_connection()
                data_index = zip(list(batched(uuids_for_embeddings, N_insert_batch_size)),
                                list(batched(entity_uuids_batched_flat, N_insert_batch_size)),
                                 list(batched(vectors_to_index_flat, N_insert_batch_size)))

                # drop collection if the collection exists
                if has_collection(_COLLECTION_NAME):
                    drop_collection(_COLLECTION_NAME)

                # create collection
                if single_non_vector_field:
                    collection, partition = create_collection_single_non_vector_field(_COLLECTION_NAME, _EMBEDDING_ID_FIELD_NAME, _VECTOR_FIELD_NAME)
                else:
                    collection, partition = create_collection(_COLLECTION_NAME, _EMBEDDING_ID_FIELD_NAME, _ENTITY_ID_FIELD_NAME, _VECTOR_FIELD_NAME)

                # # alter ttl properties of collection level to something other than 0 for testing (0 means disabled)
                set_ttl_seconds(collection, 0 )

                # show collections
                list_collections()


                for d in data_index:

                    try:
                        if single_non_vector_field:
                            insert_embeddings(partition,[list(d[0]), list(d[2])])
                        else:
                            insert_embeddings(partition,[list(d[0]), list(d[1]), list(d[2])])
                    except:
                        import pdb
                        pdb.set_trace()

                collection.flush()

                get_entity_num(collection)

                # create index
                create_index(collection, _VECTOR_FIELD_NAME)

                load_collection(collection)
                res = []
                for ii in range(N_same_index_search_repeat):
                    r = [search(partition, _VECTOR_FIELD_NAME, list(vecs)) for vecs in batched(vectors_to_search, N_search_batch_size)]
                    res.append(r)
                res = list(zip(*res))
                results.append(res)

            all_results_same = np.asarray([str(r)==str(results[0]) for r in results[1:]]).all()
            test_metadata.update({'all_results_same':all_results_same})
            data_out = []
            for run_result in results:
                r = []
                for search_set_pair in run_result:
                    s = []
                    for pair_id, search_set in enumerate(search_set_pair):
                        sp = []
                        for hits in search_set:
                            sp.append({'identical_search_instance_id': pair_id,
                                    'res':{k:v for k,v in zip(hits.ids,hits.distances)}})
                        s.append(sp)
                    r.append(s)
                data_out.append(r)

            test_results = {'metadata':test_metadata,
                            'results':data_out}
            pickle.dump(test_results,open(f'non_reproducibile_index_test_results_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}.pickle','wb'))
            print('\n\n')
            print('OUTCOME:\n')
            if all_results_same:
                print('All results the same. Rebuilding the index with same parameters and input, and then querying with a query set always returns same values')
                print(test_metadata)
            else:
                print('Either index non-reproducible or search is non-reproducible')
                print(test_metadata)
            # all_test_results.append(test_results)

   