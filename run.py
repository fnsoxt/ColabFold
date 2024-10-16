import sys

from colabfold.batch import get_queries, run
from colabfold.download import default_data_dir
from colabfold.utils import setup_logging
from pathlib import Path

result_dir = "test_all"
input_dir = "test-data/P54025.fasta"
num_models = 1

setup_logging(Path(result_dir).joinpath("log.txt"))

queries, is_complex = get_queries(input_dir)
run(
    queries=queries,
    result_dir=result_dir,
    num_models=num_models,
    is_complex=is_complex

