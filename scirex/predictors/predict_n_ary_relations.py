#! /usr/bin/env python

import json
import os
import warnings
from sys import argv
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import scipy.stats as st

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from scirex.predictors.utils import merge_method_subrelations

from scirex_utilities.json_utilities import load_jsonl, annotations_to_jsonl

import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)

def combine_span_and_cluster_file(span_file, cluster_file) :
    spans = load_jsonl(span_file)
    clusters = {item['doc_id'] :  item for item in load_jsonl(cluster_file)}

    for doc in spans :
        if 'clusters' in clusters[doc['doc_id']] :
            doc['coref'] = clusters[doc['doc_id']]['clusters']
        else :
            merge_method_subrelations(clusters[doc['doc_id']])
            doc['coref'] = {x: v for x, v in clusters[doc['doc_id']]['coref'].items() if len(v) > 0}

        if 'n_ary_relations' in doc:
            del doc['n_ary_relations']

        if 'method_subrelations' in doc :
            del doc['method_subrelations']

       
    annotations_to_jsonl(spans, 'tmp_relation_42424242.jsonl')


def predict(archive_folder, span_file, cluster_file, output_file, cuda_device):
    combine_span_and_cluster_file(span_file, cluster_file)

    test_file = 'tmp_relation_42424242.jsonl'
    relation_threshold = json.load(open(archive_folder + '/metrics.json'))['best_validation__n_ary_rel_global_threshold']
    print(relation_threshold)
    
    import_submodules("scirex")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    dataset_reader.prediction_mode = True
    instances = dataset_reader.read(test_file)

    for instance in instances :
        batch = Batch([instance])
        batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            with torch.no_grad() :
                batch = nn_util.move_to_device(batch, cuda_device)
                output_res = model.decode_relations(batch)

            n_ary_relations = output_res['n_ary_relation']
            predicted_relations, scores = n_ary_relations['candidates'], n_ary_relations['scores']
            try:
                metadata = output_res['n_ary_relation']['metadata'][0]
            except (KeyError, IndexError):
                continue
            doc_id = metadata['doc_id']
            coref_key_map = {k:i for i, k in metadata['document_metadata']['cluster_name_to_id'].items()}
        
            for i, rel in enumerate(predicted_relations) :
                predicted_relations[i] = tuple([coref_key_map[k] if k in coref_key_map else None for k in rel])

            if doc_id not in documents :
                documents[doc_id] = {'predicted_relations' : [], 'doc_id' : doc_id}
            scores_ = list(scores.ravel())
            if not scores_:
                warnings.warn(f"no relation scores defined for {doc_id}")
                continue
            label = [1 if x > relation_threshold else 0 for x in scores_]
            if all(l == 0 for l in label):
                decoding_mode = os.environ.get("SCIREX_RELATION_DECODING")
                if decoding_mode == "report_single_most_likely":
                    label[scores.argmax()] = 1
                elif decoding_mode == "report_probabilistically":
                    idxs_sorted_by_score = sorted(
                        range(len(label)),
                        key=lambda i: scores[i],
                        reverse=True  # highest score first
                    )
                    possible_decoding_idxs = \
                        [idxs_sorted_by_score[:i] for i in range(1, 11)]  # assuming that >10 relationships would never happen
                    def score_decoding(candidate_idxs):
                        """likelihood function for a geometric distribution fit
                        to the training distribution of number-of-relationships-per-document

                        :param candidate_idxs (List[int]): a list of idxs that represents a relationship distribution
                        :return: likelihood that distribution
                        """
                        score_from_n_relationships = st.geom.pmf(
                            len(candidate_idxs),
                            0.4046692607003891  # MLE from training distribution, i.e.: 1 / (1 + E[X])
                        )
                        score_from_indiv_relationships = scores[candidate_idxs]
                        return score_from_n_relationships * np.prod(score_from_indiv_relationships)
                    best_decoding_idxs = max(possible_decoding_idxs, key=score_decoding)
                    for idx in best_decoding_idxs:
                        label[idx] = 1

            scores = [round(float(x), 4) for x in list(scores.ravel())]
            documents[doc_id]['predicted_relations'] += list(zip(predicted_relations, scores, label))

        for d in documents.values() :
            predicted_relations = {}
            for r, s, l in d['predicted_relations'] :
                r = tuple(r)
                if r not in predicted_relations or predicted_relations[r][0] < s:
                    predicted_relations[r] = (s, l)

            d['predicted_relations'] = [(r, s, l) for r, (s, l) in predicted_relations.items()]

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


if __name__ == '__main__' :
    predict(argv[1], argv[2], argv[3], argv[4], int(argv[5]))