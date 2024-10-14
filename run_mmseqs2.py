
'''
from colabfold.colabfold import run_mmseqs2

query_seqs_unique = 'MPKIIEAIYENGVFKPLQKVDLKEGE'
prefix = 'zhang'
use_env = False
# host_url = 'https://api.colabfold.com'
host_url = 'http://localhost:8888'
user_agent = ''
use_pairing = True # jobpair

result = run_mmseqs2(query_seqs_unique, prefix, use_env, use_templates=False, use_pairing=use_pairing, host_url=host_url, user_agent=user_agent)
print(result)
'''
from pathlib import Path
from colabfold.batch import get_msa_and_templates_sync

host_url = 'http://localhost:8888'
Q60262 = "MEIIALLIEEGIIIIKDKKVAERFLKDLESSQGMDWKEIRERAERAKKQLEEGIEWAKKTKL"
msa_mode = "mmseqs2_uniref"
(
    unpaired_msa,
    paired_msa,
    query_seqs_unique,
    query_seqs_cardinality,
    template_features,
) = get_msa_and_templates_sync(
    "/home/xukui/jobs/",
    Q60262,
    None,
    Path("."),
    msa_mode,
    False,
    None,
    "unpaired_paired",
    "greedy",
    host_url=host_url
)
print(unpaired_msa)
print(paired_msa)
print(query_seqs_unique)
print(query_seqs_cardinality)
print(template_features)


'''
def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, use_pairing=False, pairing_strategy="greedy",
                host_url="https://api.colabfold.com",
                user_agent: str = "") -> Tuple[List[str], List[str]]:

def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    a3m_lines: Optional[List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    pairing_strategy: str = "greedy",
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
'''
