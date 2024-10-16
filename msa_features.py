from __future__ import annotations

import os
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"

import json
import logging
import math
import random
import sys
import time
import zipfile
import tarfile

import shutil
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

import numpy as np
import pandas
import pickle
import gzip
try:
    import alphafold
except ModuleNotFoundError:
    raise RuntimeError(
        "\n\nalphafold is not installed. Please run `pip install colabfold[alphafold]`\n"
    )

from alphafold.common import protein, residue_constants

# delay imports of tensorflow, jax and numpy
# loading these for type checking only can take around 10 seconds just to show a CLI usage message
if TYPE_CHECKING:
    from numpy import ndarray

from alphafold.common import protein
from alphafold.data import (
    feature_processing,
    msa_pairing,
    pipeline,
    pipeline_multimer,
    templates,
)
from alphafold.data.tools import hhsearch

# DEFAULT_API_SERVER = "https://api.colabfold.com"
DEFAULT_API_SERVER = "http://127.0.0.1:8888"

# from mmseqsfold.utils import (
#     ACCEPT_DEFAULT_TERMS,
#     DEFAULT_API_SERVER,
#     NO_GPU_FOUND,
#     CIF_REVISION_DATE,
#     get_commit,
#     safe_filename,
#     setup_logging,
#     CFMMCIFIO,
# )
import requests

from clfold.utils import (
    # ACCEPT_DEFAULT_TERMS,
    # DEFAULT_API_SERVER,
    # NO_GPU_FOUND,
    # CIF_REVISION_DATE,
    # get_commit,
    # safe_filename,
    # setup_logging,
    CFMMCIFIO,
)

from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
from Bio.PDB.PDBIO import Select

logger = logging.getLogger(__name__)
# import jax
# import jax.numpy as jnp
# logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, use_pairing=False,
                host_url="https://api.colabfold.com") -> Tuple[List[str], List[str]]:
  submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

  def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
      query += f">{n}\n{seq}\n"
      n += 1

    while True:
      error_count = 0
      try:
        # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
        # "good practice to set connect timeouts to slightly larger than a multiple of 3"
        res = requests.post(f'{host_url}/{submission_endpoint}', data={'q':query,'mode': mode}, timeout=6.02)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while submitting to MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break

    try:
      out = res.json()
    except ValueError:
      logger.error(f"Server didn't reply with json: {res.text}")
      out = {"status":"ERROR"}
    return out

  def status(ID):
    while True:
      error_count = 0
      try:
        res = requests.get(f'{host_url}/ticket/{ID}', timeout=6.02)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while fetching status from MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
    try:
      out = res.json()
    except ValueError:
      logger.error(f"Server didn't reply with json: {res.text}")
      out = {"status":"ERROR"}
    return out

  def download(ID, path):
    error_count = 0
    while True:
      try:
        res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while fetching result from MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
    with open(path,"wb") as out: out.write(res.content)

  # process input x
  seqs = [x] if isinstance(x, str) else x

  # compatibility to old option
  if filter is not None:
    use_filter = filter
  # setup mode
  if use_filter:
    mode = "env" if use_env else "all"
  else:
    mode = "env-nofilter" if use_env else "nofilter"

  path = f"{prefix}"
  if use_pairing:
    mode = ""
    use_templates = False
    use_env = False
    path = f"{prefix}_pair"

  # define path
#   path = f"{prefix}_{mode}"
  if not os.path.isdir(path): os.mkdir(path)

  # call mmseqs2 api
  tar_gz_file = f'{path}/out.tar.gz'
  N,REDO = 101,True

  # deduplicate and keep track of order
  seqs_unique = []
  #TODO this might be slow for large sets
  [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
  Ms = [N + seqs_unique.index(seq) for seq in seqs]
  # lets do it!
  if not os.path.isfile(tar_gz_file):
    TIME_ESTIMATE = 150 * len(seqs_unique)
    with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
      while REDO:
        pbar.set_description("SUBMIT")

        # Resubmit job until it goes through
        out = submit(seqs_unique, mode, N)
        while out["status"] in ["UNKNOWN", "RATELIMIT"]:
          sleep_time = 5 + random.randint(0, 5)
          logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
          # resubmit
          time.sleep(sleep_time)
          out = submit(seqs_unique, mode, N)

        if out["status"] == "ERROR":
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

        if out["status"] == "MAINTENANCE":
          raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

        # wait for job to finish
        ID,TIME = out["id"],0
        pbar.set_description(out["status"])
        while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
          t = 5 + random.randint(0,5)
          logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
          time.sleep(t)
          out = status(ID)
          pbar.set_description(out["status"])
          if out["status"] == "RUNNING":
            TIME += t
            pbar.update(n=t)
          #if TIME > 900 and out["status"] != "COMPLETE":
          #  # something failed on the server side, need to resubmit
          #  N += 1
          #  break

        if out["status"] == "COMPLETE":
          if TIME < TIME_ESTIMATE:
            pbar.update(n=(TIME_ESTIMATE-TIME))
          REDO = False

        if out["status"] == "ERROR":
          REDO = False
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

      # Download results
      download(ID, tar_gz_file)

  # prep list of a3m files
  if use_pairing:
    a3m_files = [f"{path}/pair.a3m"]
  else:
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env: 
      env_raw_path = f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m"
      a3m_files.append(env_raw_path)

  # extract a3m files
  if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
    with tarfile.open(tar_gz_file) as tar_gz:
      tar_gz.extractall(path)
    
  if use_env: 
    env_new_path = f"{path}/env.a3m"
    os.system(f"mv {env_raw_path} {env_new_path}")
    a3m_files[1] = env_new_path

  # templates
  if use_templates:
    templates = {}
    #print("seq\tpdb\tcid\tevalue")
    for line in open(f"{path}/pdb70.m8","r"):
      p = line.rstrip().split()
      M,pdb,qid,e_value = p[0],p[1],p[2],p[10]
      M = int(M)
      if M not in templates: templates[M] = []
      templates[M].append(pdb)
      #if len(templates[M]) <= 20:
      #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

    template_paths = {}
    for k,TMPL in templates.items():
      # TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
      TMPL_PATH = f"{prefix}/templates_{k}"
      if not os.path.isdir(TMPL_PATH):
        os.mkdir(TMPL_PATH)
        TMPL_LINE = ",".join(TMPL[:20])
        response = None
        while True:
          error_count = 0
          try:
            # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
            # "good practice to set connect timeouts to slightly larger than a multiple of 3"
            response = requests.get(f"{host_url}/template/{TMPL_LINE}", stream=True, timeout=6.02)
          except requests.exceptions.Timeout:
            logger.warning("Timeout while submitting to template server. Retrying...")
            continue
          except Exception as e:
            error_count += 1
            logger.warning(f"Error while fetching result from template server. Retrying... ({error_count}/5)")
            logger.warning(f"Error: {e}")
            time.sleep(5)
            if error_count > 5:
              raise
            continue
          break
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
          tar.extractall(path=TMPL_PATH)
        os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
        with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
          f.write("")
      template_paths[k] = TMPL_PATH

  # gather a3m lines
  a3m_lines = {}
  for a3m_file in a3m_files:
    update_M,M = True,None
    for line in open(a3m_file,"r"):
      if len(line) > 0:
        if "\x00" in line:
          line = line.replace("\x00","")
          update_M = True
        if line.startswith(">") and update_M:
          M = int(line[1:].rstrip())
          update_M = False
          if M not in a3m_lines: a3m_lines[M] = []
        a3m_lines[M].append(line)

  # return results

  a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

  if use_templates:
    template_paths_ = []
    for n in Ms:
      if n not in template_paths:
        template_paths_.append(None)
        #print(f"{n-N}\tno_templates_found")
      else:
        template_paths_.append(template_paths[n])
    template_paths = template_paths_

  return (a3m_lines, template_paths) if use_templates else a3m_lines



def msa_to_str(
    unpaired_msa: List[str],
    paired_msa: List[str],
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
) -> str:
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    # build msa with cardinality of 1, it makes it easier to parse and manipulate
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa

def unserialize_msa(
    a3m_lines: List[str], query_sequence: Union[List[str], str]
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    List[str],
    List[int],
    List[Dict[str, Any]],
]:
    a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
    if not a3m_lines[0].startswith("#") or len(a3m_lines[0][1:].split("\t")) != 2:
        assert isinstance(query_sequence, str)
        return (
            ["\n".join(a3m_lines)],
            None,
            [query_sequence],
            [1],
            [mk_mock_template(query_sequence)],
        )

    if len(a3m_lines) < 3:
        raise ValueError(f"Unknown file format a3m")
    tab_sep_entries = a3m_lines[0][1:].split("\t")
    query_seq_len = tab_sep_entries[0].split(",")
    query_seq_len = list(map(int, query_seq_len))
    query_seqs_cardinality = tab_sep_entries[1].split(",")
    query_seqs_cardinality = list(map(int, query_seqs_cardinality))
    is_homooligomer = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] > 1 else False
    )
    is_single_protein = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1 else False
    )
    query_seqs_unique = []
    prev_query_start = 0
    # we store the a3m with cardinality of 1
    for n, query_len in enumerate(query_seq_len):
        query_seqs_unique.append(
            a3m_lines[2][prev_query_start : prev_query_start + query_len]
        )
        prev_query_start += query_len
    paired_msa = [""] * len(query_seq_len)
    unpaired_msa = [""] * len(query_seq_len)
    already_in = dict()
    for i in range(1, len(a3m_lines), 2):
        header = a3m_lines[i]
        seq = a3m_lines[i + 1]
        if (header, seq) in already_in:
            continue
        already_in[(header, seq)] = 1
        has_amino_acid = [False] * len(query_seq_len)
        seqs_line = []
        prev_pos = 0
        for n, query_len in enumerate(query_seq_len):
            paired_seq = ""
            curr_seq_len = 0
            for pos in range(prev_pos, len(seq)):
                if curr_seq_len == query_len:
                    prev_pos = pos
                    break
                paired_seq += seq[pos]
                if seq[pos].islower():
                    continue
                if seq[pos] != "-":
                    has_amino_acid[n] = True
                curr_seq_len += 1
            seqs_line.append(paired_seq)

        # is sequence is paired add them to output
        if (
            not is_single_protein
            and not is_homooligomer
            and sum(has_amino_acid) == len(query_seq_len)
        ):
            header_no_faster = header.replace(">", "")
            header_no_faster_split = header_no_faster.split("\t")
            for j in range(0, len(seqs_line)):
                paired_msa[j] += ">" + header_no_faster_split[j] + "\n"
                paired_msa[j] += seqs_line[j] + "\n"
        else:
            for j, seq in enumerate(seqs_line):
                if has_amino_acid[j]:
                    unpaired_msa[j] += header + "\n"
                    unpaired_msa[j] += seq + "\n"
    if is_homooligomer:
        # homooligomers
        num = 101
        paired_msa = [""] * query_seqs_cardinality[0]
        for i in range(0, query_seqs_cardinality[0]):
            paired_msa[i] = ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
    if is_single_protein:
        paired_msa = None
    template_features = []
    for query_seq in query_seqs_unique:
        template_feature = mk_mock_template(query_seq)
        template_features.append(template_feature)

    return (
        unpaired_msa,
        paired_msa,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )


def mk_mock_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def mk_template(
    a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


def validate_and_fix_mmcif(cif_file: Path):
    """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
    # check that required poly_seq and revision_date fields are present
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    required = [
        "_chem_comp.id",
        "_chem_comp.type",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "_entity_poly_seq.mon_id",
    ]
    for r in required:
        if r not in cif_dict:
            raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
    if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
        logger.info(
            f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
        )
        shutil.copy2(cif_file, str(cif_file) + ".bak")
        with open(cif_file, "a") as f:
            f.write(CIF_REVISION_DATE)

modified_mapping = {
  "MSE" : "MET", "MLY" : "LYS", "FME" : "MET", "HYP" : "PRO",
  "TPO" : "THR", "CSO" : "CYS", "SEP" : "SER", "M3L" : "LYS",
  "HSK" : "HIS", "SAC" : "SER", "PCA" : "GLU", "DAL" : "ALA",
  "CME" : "CYS", "CSD" : "CYS", "OCS" : "CYS", "DPR" : "PRO",
  "B3K" : "LYS", "ALY" : "LYS", "YCM" : "CYS", "MLZ" : "LYS",
  "4BF" : "TYR", "KCX" : "LYS", "B3E" : "GLU", "B3D" : "ASP",
  "HZP" : "PRO", "CSX" : "CYS", "BAL" : "ALA", "HIC" : "HIS",
  "DBZ" : "ALA", "DCY" : "CYS", "DVA" : "VAL", "NLE" : "LEU",
  "SMC" : "CYS", "AGM" : "ARG", "B3A" : "ALA", "DAS" : "ASP",
  "DLY" : "LYS", "DSN" : "SER", "DTH" : "THR", "GL3" : "GLY",
  "HY3" : "PRO", "LLP" : "LYS", "MGN" : "GLN", "MHS" : "HIS",
  "TRQ" : "TRP", "B3Y" : "TYR", "PHI" : "PHE", "PTR" : "TYR",
  "TYS" : "TYR", "IAS" : "ASP", "GPL" : "LYS", "KYN" : "TRP",
  "CSD" : "CYS", "SEC" : "CYS"
}

class ReplaceOrRemoveHetatmSelect(Select):
  def accept_residue(self, residue):
    hetfield, _, _ = residue.get_id()
    if hetfield != " ":
      if residue.resname in modified_mapping:
        # set unmodified resname
        residue.resname = modified_mapping[residue.resname]
        # clear hetatm flag
        residue._id = (" ", residue._id[1], " ")
        t = residue.full_id
        residue.full_id = (t[0], t[1], t[2], residue._id)
        return 1
      return 0
    else:
      return 1

def convert_pdb_to_mmcif(pdb_file: Path):
    """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
    i = pdb_file.stem
    cif_file = pdb_file.parent.joinpath(f"{i}.cif")
    if cif_file.is_file():
        return
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(i, pdb_file)
    cif_io = CFMMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_file), ReplaceOrRemoveHetatmSelect())


def mk_hhsearch_db(template_dir: str):
    template_path = Path(template_dir)

    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
        validate_and_fix_mmcif(cif_file)

    pdb_files = template_path.glob("*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_mmcif(pdb_file)

    pdb70_db_files = template_path.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    with open(template_path.joinpath("pdb70_a3m.ffdata"), "w") as a3m, open(
        template_path.joinpath("pdb70_cs219.ffindex"), "w"
    ) as cs219_index, open(
        template_path.joinpath("pdb70_a3m.ffindex"), "w"
    ) as a3m_index, open(
        template_path.joinpath("pdb70_cs219.ffdata"), "w"
    ) as cs219:
        n = 1000000
        index_offset = 0
        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            with open(cif_file) as f:
                cif_string = f.read()
            cif_fh = StringIO(cif_string)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("none", cif_fh)
            models = list(structure.get_models())
            if len(models) != 1:
                raise ValueError(
                    f"Only single model PDBs are supported. Found {len(models)} models."
                )
            model = models[0]
            for chain in model:
                amino_acid_res = []
                for res in chain:
                    if res.id[2] != " ":
                        raise ValueError(
                            f"PDB contains an insertion code at chain {chain.id} and residue "
                            f"index {res.id[1]}. These are not supported."
                        )
                    amino_acid_res.append(
                        residue_constants.restype_3to1.get(res.resname, "X")
                    )

                protein_str = "".join(amino_acid_res)
                a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                a3m_str_len = len(a3m_str)
                a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                index_offset += a3m_str_len
                a3m.write(a3m_str)
                cs219.write("\n\0")
                n += 1


def build_monomer_feature(
    sequence: str, unpaired_msa: str, template_features: Dict[str, Any]
):
    msa = pipeline.parsers.parse_a3m(unpaired_msa)
    # gather features
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)
        ),
        **pipeline.make_msa_features([msa]),
        **template_features,
    }

def build_multimer_feature(paired_msa: str) -> Dict[str, ndarray]:
    parsed_paired_msa = pipeline.parsers.parse_a3m(paired_msa)
    return {
        f"{k}_all_seq": v
        for k, v in pipeline.make_msa_features([parsed_paired_msa]).items()
    }

def process_multimer_features(
    features_for_chain: Dict[str, Dict[str, ndarray]]
) -> Dict[str, ndarray]:
    # import pdb;pdb.set_trace() 
    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    # np_example = feature_processing.pair_and_merge(
    #    all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    np_example = feature_processing.process_final(np_example)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=512)
    return np_example


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

def get_queries(
    input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix == ".csv" or input_path.suffix == ".tsv":
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pandas.read_csv(input_path, sep=sep)
            assert "id" in df.columns and "sequence" in df.columns
            queries = [
                (seq_id, sequence.upper().split(":"), None)
                for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
            ]
            for i in range(len(queries)):
                if len(queries[i][1]) == 1:
                    queries[i] = (queries[i][0], queries[i][1][0], None)
        elif input_path.suffix == ".a3m":
            (seqs, header) = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            query_sequence = seqs[0]
            # Use a list so we can easily extend this to multiple msas later
            a3m_lines = [input_path.read_text()]
            queries = [(input_path.stem, query_sequence, a3m_lines)]
        elif input_path.suffix in [".fasta", ".faa", ".fa"]:
            (sequences, headers) = parse_fasta(input_path.read_text())
            queries = []
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    # Single sequence
                    queries.append((header, sequence, None))
                else:
                    # Complex mode
                    queries.append((header, sequence.upper().split(":"), None))
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        assert input_path.is_dir(), "Expected either an input file or a input directory"
        queries = []
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in [".a3m", ".fasta", ".faa"]:
                logger.warning(f"non-fasta/a3m file in input directory: {file}")
                continue
            (seqs, header) = parse_fasta(file.read_text())
            if len(seqs) == 0:
                logger.error(f"{file} is empty")
                continue
            query_sequence = seqs[0]
            if len(seqs) > 1 and file.suffix in [".fasta", ".faa", ".fa"]:
                logger.warning(
                    f"More than one sequence in {file}, ignoring all but the first sequence"
                )

            if file.suffix.lower() == ".a3m":
                a3m_lines = [file.read_text()]
                queries.append((file.stem, query_sequence.upper(), a3m_lines))
            else:
                if query_sequence.count(":") == 0:
                    # Single sequence
                    queries.append((file.stem, query_sequence, None))
                else:
                    # Complex mode
                    queries.append((file.stem, query_sequence.upper().split(":"), None))

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len(t[1]))
    elif sort_queries_by == "random":
        random.shuffle(queries)
    is_complex = False
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = tab_sep_entries[0].split(",")
                query_seq_len = list(map(int, query_seq_len))
                query_seqs_cardinality = tab_sep_entries[1].split(",")
                query_seqs_cardinality = list(map(int, query_seqs_cardinality))
                is_single_protein = (
                    True
                    if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1
                    else False
                )
                if not is_single_protein:
                    is_complex = True
                    break
    return queries, is_complex

def pair_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
    return "\n".join(a3m_line_paired)

def pad_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)


def pair_msa(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    paired_msa: Optional[List[str]],
    unpaired_msa: Optional[List[str]],
) -> str:
    if paired_msa is None and unpaired_msa is not None:
        a3m_lines = pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    elif paired_msa is not None and unpaired_msa is not None:
        a3m_lines = (
            pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
            + "\n"
            + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    elif paired_msa is not None and unpaired_msa is None:
        a3m_lines = pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    else:
        raise ValueError(f"Invalid pairing")
    return a3m_lines



def generate_input_feature(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    unpaired_msa: List[str],
    paired_msa: List[str],
    template_features: List[Dict[str, Any]],
    is_complex: bool,
    model_type: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    input_feature = {}
    domain_names = {}
    if is_complex and "ptm" in model_type:

        full_sequence = ""
        Ls = []
        for sequence_index, sequence in enumerate(query_seqs_unique):
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                full_sequence += sequence
                Ls.append(len(sequence))

        # bugfix
        a3m_lines = f">0\n{full_sequence}\n"
        a3m_lines += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)        

        input_feature = build_monomer_feature(full_sequence, a3m_lines, mk_mock_template(full_sequence))
        input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
        input_feature["asym_id"] = np.concatenate([np.full(L,n) for n,L in enumerate(Ls)])
        if any(
            [
                template != b"none"
                for i in template_features
                for template in i["template_domain_names"]
            ]
        ):
            logger.warning(
                "alphafold2_ptm complex does not consider templates. Chose multimer model-type for template support."
            )

    else:
        features_for_chain = {}
        chain_cnt = 0
        # for each unique sequence
        for sequence_index, sequence in enumerate(query_seqs_unique):
            
            # get unpaired msa
            if unpaired_msa is None:
                input_msa = f">{101 + sequence_index}\n{sequence}"
            else:
                input_msa = unpaired_msa[sequence_index]

            feature_dict = build_monomer_feature(
                sequence, input_msa, template_features[sequence_index])

            if "multimer" in model_type:
                # get paired msa
                if paired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = paired_msa[sequence_index]
                feature_dict.update(build_multimer_feature(input_msa))

            # for each copy
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                features_for_chain[protein.encode_chain_id(chain_cnt)] = feature_dict
                chain_cnt += 1

        if "ptm" in model_type:
            input_feature = features_for_chain[protein.encode_chain_id(0)]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0],dtype=int)
            domain_names = {
                protein.encode_chain_id(0): [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
        else:
            # combine features across all chains
            input_feature = process_multimer_features(features_for_chain)
            domain_names = {
                chain: [
                    name.decode("UTF-8")
                    for name in feature["template_domain_names"]
                    if name != b"none"
                ]
                for (chain, feature) in features_for_chain.items()
            }
    return (input_feature, domain_names)


def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    # get template features
    template_features = []
    if use_templates:
        a3m_lines_mmseqs2, template_paths = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=True,
            host_url=host_url,
        )
        if custom_template_path is not None:
            # import pdb;pdb.set_trace()
            template_paths = {}
            for index in range(0, len(query_seqs_unique)):
                template_paths[index] = custom_template_path
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = mk_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = mk_mock_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = mk_mock_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"
    # import pdb;pdb.set_trace()
    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
            )
    else:
        a3m_lines = None
    # import pdb;pdb.set_trace()
    if msa_mode != "single_sequence" and (
        pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=True,
                host_url=host_url,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None
    # import pdb;pdb.set_trace()

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )


from colabfold.batch import get_msa_and_templates_sync

def run_mmf_MSAFt(query_sequence, result_dir, jobname="env", msa_mode="mmseqs2_uniref_env", use_templates=True, 
                custom_template_path=None, pair_mode="unpaired_paired"):
    ###########################################
    # generate MSA (a3m_lines) and templates
    ###########################################
    # queries, is_complex = get_queries(args.input, args.sort_queries_by)
    # raw_jobname, query_sequence, a3m_lines = queries
    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)
    host_url = DEFAULT_API_SERVER
    a3m_lines = None
    result_dir = Path(result_dir)
    # result_dir.mkdir(exist_ok=True)
    try:
        if use_templates or a3m_lines is None:
            (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
            = get_msa_and_templates_sync(jobname, query_sequence, a3m_lines, result_dir=result_dir, msa_mode=msa_mode, use_templates=use_templates, 
                custom_template_path=custom_template_path, pair_mode=pair_mode, host_url=host_url)
        if a3m_lines is not None:
            (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features_) \
            = unserialize_msa(a3m_lines, query_sequence)
            if not use_templates: template_features = template_features_

        # save a3m
        msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
        result_dir.joinpath(f"{jobname}_new.a3m").write_text(msa)
            
    except Exception as e:
        logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
        return None
    
    #######################
    # generate features
    #######################
    # import pdb;pdb.set_trace()
    is_complex = isinstance(query_sequence, list)  and len(query_sequence) > 1 
    print("is_complex: ", is_complex)

    model_type = "alphafold2_multimer_v2" if is_complex else "alphafold2_ptm"
    try:
        (feature_dict, domain_names) \
        = generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                    template_features, is_complex, model_type)
        pkl_path =  str(result_dir.joinpath("features.pkl.gz"))
        
        pickle.dump(feature_dict, gzip.open(pkl_path, 'wb'), protocol=4)
        print(f"features.pkl.gz file saved into {pkl_path}")
        return feature_dict
        # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
        # if feature_dict_callback is not None:
        #     feature_dict_callback(feature_dict)
        
    except Exception as e:
        logger.exception(f"Could not generate input features {jobname}: {e}")
        return None


if __name__ == "__main__":
    query_sequence = 'MPKIIEAIYENGVFKPLQKVDLKEGE'
    result_dir="/home/xukui/jobs/"
    jobname = ""
    run_mmf_MSAFt(query_sequence, result_dir, jobname)
