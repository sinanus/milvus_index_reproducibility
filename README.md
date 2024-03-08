# milvus_index_reproducibility
Testing Milvus Index reproducibility


## Setup
`sudo docker compose up -d`
`poetry install`

## Run tests
Set the number of vectors to generate for the test. 90% will be indexed and 10% will be used for searching.

`time poetry run python run_repeated_tests.py --n_vectors 5e6`

## Explore results
`poetry run python explore_results.py`
