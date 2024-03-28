# milvus_index_reproducibility
Scripts for testing Milvus Binary IVF index reproducibility by performing repeated indexing and searching using **identical index settings and vectors**.

Using these scripts on a 2019 MBP I start to get non-reproducible BIN_IVF indices at around 1.5 million vectors. Bug report will be submitted to milvus. This repo is mainly for other people to reproduce the bug.

## Setup
`sudo docker compose up -d`
`poetry install`

## Usage

In the script you can set:
* Index params (index_type, metric_type, nlist, nprobe, topk)
* `N_insert_batch_size`: Batch size for vector insertion in to index
* `N_search_batch_size`: Batch size for vector similarity search
* `N_index_rebuild`: Number of times you want to rebuild the index
* `N_same_index_search_repeat`: Number of times you want to repeat the search of all vectors for the same index (to confirm that search results are reproducible for a given index build)
* `single_non_vector_field_collection_vals`: You can give a list of booleans to have single or double non-vector field in the CollectionSchema. The non-reproducible index creation is somehow worsened by having multiple non-vector fields in the Collection Schema.
* `embeddings_per_entity`: At some point, I thought having multiple embeddings with the same identifier field (varchar type, as I use uuids) was to blame for the reproduciblity issue. Dont believe that is the case anymore so no need to modify this.


Can specify the number of vectors to insert in the index and the number of search vectors like so:

`poetry run python run_repeated_tests.py --n_vectors 5e6 --n_vectors_search 3e5 --nprobe 5`

Can also specify a range of index sizes and set NPROBE parameter for BIN_IVF_FLAT index:

`poetry run python run_repeated_tests.py --n_vectors_min 6e5 --n_vectors_max 6e6 --n_vectors_count 6 --n_vectors_search 3e5 --nprobe 5`

After the run is done, you can run the `explore_results.py` script to analyze the latests run.

`poetry run explore_results.py`

Below sample results ran on a Desktop with AMD Ryzen 9 7900X and Ubuntu. Results show that search on rebuilt indices (with same vectors) start to become unreproducible at ~2.5 million vectors (depending on number of Fields in Collection Schema), see cases where `fraction_reproducible_search_on_REBUILT_index`<1.0'. Also note that repeated searches on the same index can give non-reproducible results (see cases where `fraction_reproducible_search_on_SAME_index`<1.0')

`poetry run python run_repeated_tests.py --n_vectors_min 6e5 --n_vectors_max 6e6 --n_vectors_count 6 --n_vectors_search 2e5 --nprobe 5`
`poetry run explore_results.py`

|   ('', 'n_vec_index') |   ('', 'n_vec_search') | ('', 'single_non_vector_field')   | ('', 'results_reproducible')   |   ('', 'fraction_reproducible_search_on_SAME_index') |   ('', 'fraction_reproducible_search_on_REBUILT_index') |   ('n_vectors_returned_in_both_search', 0) |   ('n_vectors_returned_in_both_search', 1) |   ('n_vectors_returned_in_both_search', 2) |   ('n_vectors_returned_in_both_search', 3) |
|----------------------:|-----------------------:|:----------------------------------|:-------------------------------|-----------------------------------------------------:|--------------------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|
|                400000 |                 200000 | False                             | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|                400000 |                 200000 | True                              | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|               1480000 |                 200000 | False                             | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|               1480000 |                 200000 | True                              | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|               2560000 |                 200000 | False                             | False                          |                                             0.372535 |                                                0.205015 |                                      79550 |                                      11860 |                                        585 |                                     108005 |
|               2560000 |                 200000 | True                              | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|               3640000 |                 200000 | False                             | False                          |                                             0.45003  |                                                0.35     |                                      78395 |                                      11132 |                                        469 |                                     110004 |
|               3640000 |                 200000 | True                              | True                           |                                             1        |                                                1        |                                        nan |                                        nan |                                        nan |                                     200000 |
|               4720000 |                 200000 | False                             | False                          |                                             0.62503  |                                                3e-05    |                                     170948 |                                      27651 |                                       1385 |                                         16 |
|               4720000 |                 200000 | True                              | False                          |                                             0.385032 |                                                0.26001  |                                      84281 |                                      13111 |                                        602 |                                     102006 |
|               5800000 |                 200000 | False                             | False                          |                                             1        |                                                8e-05    |                                     171181 |                                      27479 |                                       1323 |                                         17 |
|               5800000 |                 200000 | True                              | False                          |                                             0.286322 |                                                0.12004  |                                     150843 |                                      23927 |                                       1213 |                                      24017 |


## Sample results with nlist = nprobe = 1000 
(Re-ran after fixing nprobe parameter setting bug)

`poetry run python run_repeated_tests.py --n_vectors_index 6e6 --n_vectors_search 2e5 --nprobe 1000`



|   ('', 'n_vec_index') |   ('', 'n_vec_search') | ('', 'single_non_vector_field')   | ('', 'results_reproducible')   |   ('', 'fraction_reproducible_search_on_SAME_index') |   ('', 'fraction_reproducible_search_on_REBUILT_index') |   ('n_vectors_returned_in_both_search', 0) |   ('n_vectors_returned_in_both_search', 1) |   ('n_vectors_returned_in_both_search', 2) |   ('n_vectors_returned_in_both_search', 3) |
|----------------------:|-----------------------:|:----------------------------------|:-------------------------------|-----------------------------------------------------:|--------------------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|
|               5800000 |                 200000 | False                             | False                          |                                             1        |                                                0.940175 |                                          1 |                                         54 |                                       6330 |                                     193615 |
|               5800000 |                 200000 | True                              | False                          |                                             0.987665 |                                                0.98376  |                                        nan |                                         12 |                                       1286 |                                     198702 |

## BIN_FLAT also can be non-reproducible
This is the weirdest as BIN_FLAT is suppsoed to be Frute Force search and 100% reproducible. Also very odd to have same index give different results now.

To run test with 6 million vectors on BIN_FLAT index type (takes ~30mins on my desktop)
`poetry run python bin_flat_reproduciblity_test_run.py --n_vectors_index 5e6 --n_vectors_search 2e5`

To get summary of results run
`poetry run python explore_bin_flat_reproducibility_test_results.py`

Sample results I got show that:
1. `fraction_reproducible_search_on_SAME_index` : Only ~98% of searches performed on the same index with the same vectors is reproducible (this was 100% on the previous BIN_IVF_FLAT index tests above)
2. `fraction_reproducible_search_on_REBUILT_index` : ~97.6% of searches performen on two indices built with same parameters and vectors is reproducible (this was ~0% for the BIN_IVF_FLAT index tests above)

|                                                             |   6fe3bb56-54d2-493f-b290-9fb0f4849197 |
|:------------------------------------------------------------|---------------------------------------:|
| ('', 'n_vec_index')                                         |                                4.8e+06 |
| ('', 'n_vec_search')                                        |                           200000       |
| ('', 'results_reproducible')                                |                                0       |
| ('', 'fraction_reproducible_search_on_SAME_index')          |                                0.99504 |
| ('', 'fraction_reproducible_search_on_REBUILT_index')       |                                0.93683 |
| ('n_vectors_returned_in_both_search', 1)                    |                               63       |
| ('n_vectors_returned_in_both_search', 2)                    |                             6610       |
| ('n_vectors_returned_in_both_search', 3)                    |                           193327       |
| ('distance_delta_for_vecs_returned_in_both_search', 'mean') |                                0       |
| ('distance_delta_for_vecs_returned_in_both_search', 'std')  |                                0       |
| ('distance_delta_for_vecs_returned_in_both_search', 'min')  |                                0       |
| ('distance_delta_for_vecs_returned_in_both_search', 'max')  |                                0       |