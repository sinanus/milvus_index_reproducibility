# milvus_index_reproducibility
Testing Milvus Index reproducibility


## Setup
`sudo docker compose up -d`
`poetry install`

## Run tests
`time poetry run python run_repeated_tests.py`

## Explore results
`poetry run python explore_results.py`
