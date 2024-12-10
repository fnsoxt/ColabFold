import gzip, pickle
from colabfold.batch import get_queries, run
from colabfold.utils import setup_logging
from pathlib import Path

def feature_dict_callback(feature_dict):
    print("------------- start feature_dict_callback  ---------------")
    output_pklgz =  "features.pkl.gz"
    if feature_dict is not None and'aatype' in feature_dict:
        pickle.dump(feature_dict, gzip.open(output_pklgz, 'wb'), protocol=4)
        print(f"features  file saved into {output_pklgz}")
    else:
        print("failed in generating input features file.")

input_dir = "test-data/0127.fasta"
result_dir = "jobs_test"
setup_logging(Path(result_dir).joinpath("log.txt"))
num_models = 1
host_url = 'http://localhost:8888'
msa_mode = "mmseqs2_uniref"
use_templates = True

queries, is_complex = get_queries(input_dir)
print("queries: %s" % queries)
print("is_complex: %s" % is_complex)

run(
    queries=queries,
    result_dir=result_dir,
    num_models=num_models,
    is_complex=is_complex,
    use_templates=use_templates,
    msa_mode=msa_mode,
    host_url=host_url,
    feature_dict_callback=feature_dict_callback,
)


'''
def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
    result_dir: Union[str, Path],
    num_models: int,
    is_complex: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    model_order: List[int] = [1,2,3,4,5],
    num_ensemble: int = 1,
    model_type: str = "auto",
    msa_mode: str = "mmseqs2_uniref_env",
    use_templates: bool = False,
    custom_template_path: str = None,
    num_relax: int = 0,
    relax_max_iterations: int = 0,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired_paired",
    pairing_strategy: str = "greedy",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
    random_seed: int = 0,
    num_seeds: int = 1,
    recompile_padding: Union[int, float] = 10,
    zip_results: bool = False,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    jobname_prefix: Optional[str] = None,
    save_all: bool = False,
    save_recycles: bool = False,
    use_dropout: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score: float = 100,
    dpi: int = 200,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    pdb_hit_file: Optional[Path] = None,
    local_pdb_path: Optional[Path] = None,
    use_cluster_profile: bool = True,
    feature_dict_callback: Callable[[Any], Any] = None,
    **kwargs
)
'''
