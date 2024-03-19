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

|   n_vec_index |   n_vec_search | single_non_vector_field   | results_reproducible   |   fraction_reproducible_search_on_SAME_index |   fraction_reproducible_search_on_REBUILT_index |
|--------------:|---------------:|:--------------------------|:-----------------------|---------------------------------------------:|------------------------------------------------:|
|        600000 |         300000 | False                     | True                   |                                            1 |                                     1           |
|        600000 |         300000 | True                      | True                   |                                            1 |                                     1           |
|        944444 |         300000 | False                     | True                   |                                            1 |                                     1           |
|        944444 |         300000 | True                      | True                   |                                            1 |                                     1           |
|       1288888 |         300000 | False                     | True                   |                                            1 |                                     1           |
|       1288888 |         300000 | True                      | True                   |                                            1 |                                     1           |
|       1633333 |         300000 | False                     | False                  |                                            1 |                                     0.000186667 |
|       1633333 |         300000 | True                      | True                   |                                            1 |                                     1           |
|       1977777 |         300000 | False                     | False                  |                                            1 |                                     0.000183333 |
|       1977777 |         300000 | True                      | False                  |                                            1 |                                     0.000156667 |
|       2322222 |         300000 | False                     | False                  |                                            1 |                                     0.000146667 |
|       2322222 |         300000 | True                      | False                  |                                            1 |                                     0.000196667 |
|       2666666 |         300000 | False                     | False                  |                                            1 |                                     0.000176667 |
|       2666666 |         300000 | True                      | False                  |                                            1 |                                     0.000183333 |
|       3011111 |         300000 | False                     | False                  |                                            1 |                                     0.00021     |
|       3011111 |         300000 | True                      | False                  |                                            1 |                                     0.000213333 |
|       3355555 |         300000 | False                     | False                  |                                            1 |                                     0.00041     |
|       3355555 |         300000 | True                      | False                  |                                            1 |                                     0.000203333 |
|       3700000 |         300000 | False                     | False                  |                                            1 |                                     0.0002      |
|       3700000 |         300000 | True                      | False                  |                                            1 |                                     0.00025     |
>>> 