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
* `embeddings_per_entity_vals`: At some point, I thought having multiple embeddings with the same identifier field (varchar type, as I use uuids) was to blame for the reproduciblity issue. Dont believe that is the case anymore so no need to modify this.


Can specify the number of vectors to insert in the index and the number of search vectors like so:

`poetry run python run_repeated_tests.py --n_vectors 5e6 --n_vectors_search 3e5`

Can also specify a range of index sizes:

`poetry run python run_repeated_tests.py --n_vectors_min 1e6 --n_vectors_max 5e6 --n_vectors_count 5 --n_vectors_search 3e5`

After the run is done, you can run the `explore_results.py` script to analyze the latests run.

`poetry run explore_results.py`

Sample results showing that while repeated search on same index are reproducible regardless of index size (`n_vec_index`) search on rebuilt indices (with same vectors) becomes unreproducible at ~1.6 million vectors.

|   ('', 'n_vec_index') |   ('', 'n_vec_search') | ('', 'single_non_vector_field')   | ('', 'results_reproducible')   |   ('', 'fraction_reproducible_search_on_SAME_index') |   ('', 'fraction_reproducible_search_on_REBUILT_index') |   ('n_vectors_returned_in_both_search', 0) |   ('n_vectors_returned_in_both_search', 1) |   ('n_vectors_returned_in_both_search', 2) |   ('n_vectors_returned_in_both_search', 3) |
|----------------------:|-----------------------:|:----------------------------------|:-------------------------------|-----------------------------------------------------:|--------------------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|
|                600000 |                 300000 | False                             | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|                600000 |                 300000 | True                              | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|                944444 |                 300000 | False                             | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|                944444 |                 300000 | True                              | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|               1288888 |                 300000 | False                             | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|               1288888 |                 300000 | True                              | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|               1633333 |                 300000 | False                             | False                          |                                                    1 |                                             0.000186667 |                                     245109 |                                      51618 |                                       3215 |                                         58 |
|               1633333 |                 300000 | True                              | True                           |                                                    1 |                                             1           |                                        nan |                                        nan |                                        nan |                                     300000 |
|               1977777 |                 300000 | False                             | False                          |                                                    1 |                                             0.000183333 |                                     240587 |                                      55601 |                                       3755 |                                         57 |
|               1977777 |                 300000 | True                              | False                          |                                                    1 |                                             0.000156667 |                                     240376 |                                      55782 |                                       3792 |                                         50 |
|               2322222 |                 300000 | False                             | False                          |                                                    1 |                                             0.000146667 |                                     241095 |                                      55084 |                                       3776 |                                         45 |
|               2322222 |                 300000 | True                              | False                          |                                                    1 |                                             0.000196667 |                                     242241 |                                      54191 |                                       3506 |                                         62 |
|               2666666 |                 300000 | False                             | False                          |                                                    1 |                                             0.000176667 |                                     239356 |                                      56748 |                                       3839 |                                         57 |
|               2666666 |                 300000 | True                              | False                          |                                                    1 |                                             0.000183333 |                                     240905 |                                      55463 |                                       3576 |                                         56 |
|               3011111 |                 300000 | False                             | False                          |                                                    1 |                                             0.00021     |                                     240773 |                                      55547 |                                       3617 |                                         63 |
|               3011111 |                 300000 | True                              | False                          |                                                    1 |                                             0.000213333 |                                     238480 |                                      57410 |                                       4045 |                                         65 |
|               3355555 |                 300000 | False                             | False                          |                                                    1 |                                             0.00041     |                                     221374 |                                      72011 |                                       6487 |                                        128 |
|               3355555 |                 300000 | True                              | False                          |                                                    1 |                                             0.000203333 |                                     239318 |                                      56723 |                                       3894 |                                         65 |
|               3700000 |                 300000 | False                             | False                          |                                                    1 |                                             0.0002      |                                     236279 |                                      59469 |                                       4191 |                                         61 |
|               3700000 |                 300000 | True                              | False                          |                                                    1 |                                             0.00025     |                                     236061 |                                      59495 |                                       4367 |                                         77 |


# UPDATE 2024-03-20
## Sample results with nlist = nprobe = 1000

This is weird as in theory nlist = nprobe should be equivalent to Brute Force search

|   ('', 'n_vec_index') |   ('', 'n_vec_search') | ('', 'single_non_vector_field')   | ('', 'results_reproducible')   |   ('', 'fraction_reproducible_search_on_SAME_index') |   ('', 'fraction_reproducible_search_on_REBUILT_index') |   ('n_vectors_returned_in_both_search', 0) |   ('n_vectors_returned_in_both_search', 1) |   ('n_vectors_returned_in_both_search', 2) |   ('n_vectors_returned_in_both_search', 3) |
|----------------------:|-----------------------:|:----------------------------------|:-------------------------------|-----------------------------------------------------:|--------------------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|
|               5800000 |                 200000 | False                             | False                          |                                                    1 |                                                0.000225 |                                     158722 |                                      38457 |                                       2772 |                                         49 |
|               5800000 |                 200000 | True                              | False                          |                                                    1 |                                                0.00026  |                                     158747 |                                      38495 |                                       2704 |                                         54 |

## BIN_FLAT also can be non-reproducible
This is the weirdest as BIN_FLAT is suppsoed to be Frute Force search and 100% reproducible. Also very odd to have same index give different results now.

To run test with 6 million vectors on BIN_FLAT index type (takes ~30mins on my desktop)
`poetry run python bin_flat_reproduciblity_test_run.py --n_vectors_index 6e6 --n_vectors_search 2e5`

To get summary of results run
`poetry run python explore_bin_flat_reproducibility_test_results.py`

Sample results I got show that:
1. `fraction_reproducible_search_on_SAME_index` : Only ~98% of searches performed on the same index with the same vectors is reproducible (this was 100% on the previous BIN_IVF_FLAT index tests above)
2. `fraction_reproducible_search_on_REBUILT_index` : ~97.6% of searches performen on two indices built with same parameters and vectors is reproducible (this was ~0% for the BIN_IVF_FLAT index tests above)

|                                                             |   6f839d5f-7df5-44cc-a3f5-c1d7cf0d8cc3 |
|:------------------------------------------------------------|---------------------------------------:|
| ('', 'n_vec_index')                                         |                               5.8e+06  |
| ('', 'n_vec_search')                                        |                          200000        |
| ('', 'results_reproducible')                                |                               0        |
| ('', 'fraction_reproducible_search_on_SAME_index')          |                               0.981555 |
| ('', 'fraction_reproducible_search_on_REBUILT_index')       |                               0.97602  |
| ('n_vectors_returned_in_both_search', 1)                    |                              24        |
| ('n_vectors_returned_in_both_search', 2)                    |                            2280        |
| ('n_vectors_returned_in_both_search', 3)                    |                          197696        |
| ('distance_delta_for_vecs_returned_in_both_search', 'mean') |                               0        |
| ('distance_delta_for_vecs_returned_in_both_search', 'std')  |                               0        |
| ('distance_delta_for_vecs_returned_in_both_search', 'min')  |                               0        |
| ('distance_delta_for_vecs_returned_in_both_search', 'max')  |                               0        |