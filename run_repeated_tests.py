
import argparse
import random
import numpy as np
import uuid
import pickle
import datetime
import itertools
from more_itertools import batched
from tqdm import tqdm
import pandas as pd


from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Partition
)

# REPEATABILITY TEST PARAMS
embeddings_per_entity = 1
N_insert_batch_size = 50000
N_search_batch_size = 500
single_non_vector_field_collection_vals = [False,True]
N_index_rebuild = 2
N_same_index_search_repeat = 2
run_uuid = uuid.uuid4()

# most index creation code below is from https://github.com/milvus-io/milvus-lite/blob/2.3/examples/example.py

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
_TOPK = 3
_CONSISTENCY_LEVEL = "Strong"

index_params = {'metric_type':_METRIC_TYPE,
                'index_type':_INDEX_TYPE,
                'nlist':_NLIST,
                'topk':_TOPK,
                'consistency_level': _CONSISTENCY_LEVEL}

def create_connection():
    connections.connect(
    alias="default",
    host='localhost',
    port='19530')


# Create a collection
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

def insert_embeddings(collection, data):
    collection.insert(data)
    return data[1]

def search(collection, vector_field,  search_vectors, nprobe, print_results=False):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params":{"nprobe": nprobe}},
        "limit": _TOPK,
        "consistency_level": _CONSISTENCY_LEVEL}
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
    

def gen_vectors(dim, num):
    np.random.seed(42); vecs = np.random.randint(0,255,(num,dim))
    return vecs

# gen binary vectors
def gen_binary_vectors(dim, num):
    # in binary vectors, a bit represent a dimension.
    # a value of `uint8` describe 8 dimension
    vectors = gen_vectors(dim // 8, num)
    return vec_to_binary(vectors)

def vec_to_binary(vectors):
    data = np.array(vectors, dtype='uint8').astype("uint8")
    return [bytes(vector) for vector in data]

def all_list_elements_identical(l:list)->bool:
	l_str = [str(x) for x in l]
	if len(set(l_str))>1:
		return False
	return True
            
if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description='Run milvus reproducibility test')
    parser.add_argument('--n_vectors_index', dest='n_vectors_index', type=str, help='Number of Vectors to put in index')
    parser.add_argument('--n_vectors_search', dest='n_vectors_search', type=str, help='Number of Vectors to search')
    parser.add_argument('--n_vectors_min', dest='n_vectors_min', type=str, help='For testing multiple N_vector_index cases set min, max and count')
    parser.add_argument('--n_vectors_max', dest='n_vectors_max', type=str, help='For testing multiple N_vector_index cases set min, max and count')
    parser.add_argument('--n_vectors_count', dest='n_vectors_count', type=str, help='For testing multiple N_vector cases set min, max and count')
    parser.add_argument('--nprobe', dest='nprobe', type=str, default = '5', help='Search parameter nprobe for IVF indices')


    args = parser.parse_args()
    if args.n_vectors_index is not None:
        n_vectors = [int(float(args.n_vectors_index))]
    else:
        assert (args.n_vectors_min is not None) & (args.n_vectors_max is not None) & (args.n_vectors_count is not None)
        n_vectors = np.linspace(int(float(args.n_vectors_min)), 
                                int(float(args.n_vectors_max)),
                                int(float(args.n_vectors_count)),dtype=int)
    if args.n_vectors_search is not None:
        n_search = int(float(args.n_vectors_search))
    else:
        n_search = int(0.3 * n_vectors[0])

    nprobe = args.nprobe
    index_params.update({'nprobe':nprobe})
    # generate vectors
    vectors = gen_vectors(_DIM//8, n_vectors[-1]+n_search)
    data = np.array(vectors, dtype='uint8').astype("uint8")
    # convert to bytes
    data_bytes = [bytes(vector) for vector in data]   
    for n_vec in n_vectors:
        nvec_uuid = uuid.uuid4()
        data_bytes_subset = data_bytes[:n_vec]
        # N_index = int(len(data_bytes_subset) * 0.9)
        vectors_to_index = data_bytes_subset[:-n_search]
        vectors_to_search = data_bytes_subset[-n_search:]

        uuids_for_embeddings = [str(uuid.uuid4()) for i in range(len(vectors_to_index))]

        all_test_results = []
        for idx_single_non_vector_field, single_non_vector_field in enumerate(single_non_vector_field_collection_vals):
            vectors_to_index_batched = list(batched(vectors_to_index, embeddings_per_entity))
            entity_uuids_batched = [[str(uuid.uuid4())]*len(v) for v in vectors_to_index_batched]
            vectors_to_index_flat = flatten_extend(vectors_to_index_batched)
            entity_uuids_batched_flat =  flatten_extend(entity_uuids_batched)


            data_index = list(zip(list(batched(uuids_for_embeddings, N_insert_batch_size)),
                            list(batched(entity_uuids_batched_flat, N_insert_batch_size)),
                            list(batched(vectors_to_index_flat, N_insert_batch_size))))

            test_uuid = uuid.uuid4()
            # pickle.dump(uuids,open(f'uuids_{test_uuid}.pickle','wb'))
            
            test_metadata = {'test_uuid':str(test_uuid),
                            'run_uuid':str(run_uuid),
                            'nvec_uuid':str(nvec_uuid),
                            'n_vec_total':n_vec,
                            'n_vec_index':len(vectors_to_index),
                            'n_vec_search':len(vectors_to_search),
                            'n_insert_batch_size':N_insert_batch_size,
                            'n_search_batch_size':N_search_batch_size,
                            'index_params':index_params,
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
            for idx_index_rebuild in range(N_index_rebuild):
                print('#'*5 + f' {idx_index_rebuild+1} of {N_index_rebuild} Index rebuilds' + '#'*5)

                create_connection()
                # data_index = zip(list(batched(uuids_for_embeddings, N_insert_batch_size)),
                #                 list(batched(entity_uuids_batched_flat, N_insert_batch_size)),
                #                 list(batched(vectors_to_index_flat, N_insert_batch_size)))

                # drop collection if the collection exists
                if has_collection(_COLLECTION_NAME):
                    drop_collection(_COLLECTION_NAME)

                # create collection
                if single_non_vector_field:
                    collection, partition = create_collection_single_non_vector_field(_COLLECTION_NAME, _EMBEDDING_ID_FIELD_NAME, _VECTOR_FIELD_NAME)
                else:
                    collection, partition = create_collection(_COLLECTION_NAME, _EMBEDDING_ID_FIELD_NAME, _ENTITY_ID_FIELD_NAME, _VECTOR_FIELD_NAME)


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
                for idx_same_index_repeat in range(N_same_index_search_repeat):
                    for idx_search_vectors_batch_id, vecs in tqdm(enumerate(batched(vectors_to_search, N_search_batch_size)),total=np.ceil(len(vectors_to_search)/N_search_batch_size)):
                        
                        batch_search_res = search(partition, _VECTOR_FIELD_NAME, list(vecs), nprobe)
                        
                        for idx_search_vector_within_batch, hits in enumerate(batch_search_res):
                            results.append({
                                            'index_rebuild_id':idx_index_rebuild,
                                            'same_index_search_repeat_id': idx_same_index_repeat,
                                            'search_vectors_batch_id': idx_search_vectors_batch_id,
                                            'search_vector_within_batch_id':idx_search_vector_within_batch,
                                            'res':{k:v for k,v in zip(hits.ids,hits.distances)}
                                            })
                            
            results_df = pd.DataFrame(results)
            all_results_same = results_df.groupby(['search_vectors_batch_id','search_vector_within_batch_id'])['res']\
                        .apply(lambda x: all_list_elements_identical(x.to_list())).all()

            test_metadata.update({'results_reproducible':all_results_same})
            
            test_results = {'metadata':test_metadata,
                            'results':results}
            pickle.dump(test_results,open(f'index_and_search_test_results_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}.pickle','wb'))
            print('\n\n')
            print('OUTCOME:\n')
            if all_results_same:
                print('Results are reproducible. Rebuilding the index with same parameters and input, and then querying with a query set always returns same values')
                print(test_metadata)
            else:
                print('Either index or search is non-reproducible')
                print(test_metadata)
            all_test_results.append(test_results)
    

        pickle.dump(all_test_results,open(f'all_test_results_for_nvec_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}_uuid_{str(run_uuid)}.pickle','wb'))