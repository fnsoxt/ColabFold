from pathlib import Path
from colabfold.batch import get_msa_and_templates, get_msa_and_templates_sync

host_url = 'http://localhost:8888'
query_sequences = 'MPKIIEAIYENGVFKPLQKVDLKEGE'
# query_sequences = "MEIIALLIEEGIIIIKDKKVAERFLKDLESSQGMDWKEIRERAERAKKQLEEGIEWAKKTKL"
msa_mode = "mmseqs2_uniref"
use_templates = True

(
    unpaired_msa,
    paired_msa,
    query_seqs_unique,
    query_seqs_cardinality,
    template_features,
) = get_msa_and_templates_sync(
    jobname="",
    query_sequences=query_sequences,
    a3m_lines=None,
    result_dir=Path("/home/xukui/jobs/"),
    msa_mode=msa_mode,
    use_templates=use_templates,
    custom_template_path=None,
    pair_mode="unpaired_paired",
    pairing_strategy="greedy",
    host_url=host_url
)
print(unpaired_msa)
print(paired_msa)
print(query_seqs_unique)
print(query_seqs_cardinality)
print(template_features)
