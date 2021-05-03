import os
from overrides import overrides
from allennlp.training.metrics.metric import Metric

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from scirex.metrics.f1 import compute_threshold


class NAryRelationMetrics(Metric):
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, candidate_relation_list, candidate_relation_labels, candidate_relation_scores, doc_id):
        try:
            candidate_relation_scores, = self.unwrap_to_tensors(candidate_relation_scores)
            candidate_relation_scores = list(candidate_relation_scores.numpy())
        except:
            breakpoint()

        assert len(candidate_relation_list) == len(candidate_relation_scores), breakpoint()
        assert len(candidate_relation_labels) == len(candidate_relation_scores), breakpoint()

        for relation, label, score in zip(
            candidate_relation_list, candidate_relation_labels, candidate_relation_scores
        ):
            relation = (doc_id, tuple(relation))
            if relation not in self._candidate_labels:
                self._candidate_labels[relation] = label

            assert self._candidate_labels[relation] == label

            if relation not in self._candidate_scores or self._candidate_scores[relation] < score:
                self._candidate_scores[relation] = score

    @overrides
    def get_metric(self, reset=False):
        if len(self._candidate_scores) == 0:
            return {}

        prediction_scores, gold = [], []
        for k in self._candidate_labels:
            prediction_scores.append(self._candidate_scores[k])
            gold.append(self._candidate_labels[k])

        try:
            threshold = compute_threshold(prediction_scores, gold)
        except:
            breakpoint()
        prediction = [1 if self._candidate_scores[k] > threshold else 0 for k in self._candidate_labels]
        if all(pred == 0 for pred in prediction) and os.environ.get("SCIREX_REPORT_MOST_LIKELY_WHEN_OTHERWISE_NONE_REPORTED", False):
            k_max, _ = max(
                [(k, self._candidate_scores[k]) for k in self._candidate_labels],
                key=lambda x: x[1]
            )
            prediction[k_max] = 1
        elif os.environ.get("SCIREX_PROBABILISTIC_DECODING", False):
            # probability mass for a poisson distribution fit to the ground truth
            # number-of-relationships-per-document distributions
            poisson = [0.0, 0.16041357336318754, 0.23835145115823195, 0.23610393675191074, 0.17540821131631165,
                       0.10425252884967452, 0.05163474432007915, 0.021920510983570014, 0.008142680032497306,
                       0.002688633111653137, 0.0007989842634372239, 0.0002158500846509767, 5.345368623379652e-05,
                       1.2219149843037534e-05, 2.5936993769525873e-06, 5.138484625738224e-07, 9.543809492090282e-08,
                       1.6683206409569134e-08, 2.7543155927959772e-09, 4.3079115978763763e-10, 6.400935776740507e-11,
                       9.057969580158813e-12, 1.223530668947186e-12, 1.5808610768642936e-13, 1.957443104981043e-14,
                       2.3267826773016636e-15, 2.659433632114712e-16, 2.927063036171183e-17, 3.1065670160121458e-18,
                       3.1833868043347594e-19, 3.153369333625557e-20, 3.022872493608182e-21, 2.807220542547777e-22,
                       2.5279546355357966e-23, 2.2095154463018706e-24, 1.876012295472675e-25, 1.5486021664077433e-26,
                       1.2437835860563036e-27, 9.726753444003379e-29, 7.411566048580087e-30, 5.506259495368715e-31,
                       3.9909795346840275e-32, 2.823819241521348e-33, 1.9515294991156167e-34, 1.3180416275372828e-35,
                       8.704088452919195e-37, 5.623052610787668e-38, 3.555338631212962e-39, 2.2011336705641793e-40,
                       1.334925471965462e-41, 7.934027445686726e-43, 4.623067188047487e-44, 2.6420044474625115e-45,
                       1.4813727706935699e-46, 8.152246078876013e-48, 4.404750097707562e-49, 2.3374370825327676e-50,
                       1.2186296955810874e-51, 6.243821028627064e-53, 3.1448875209837325e-54, 1.5576165379694948e-55,
                       7.588176036690336e-57, 3.6370761225733185e-58, 1.7156098226920493e-59, 7.966090444504941e-61,
                       3.6419884076415136e-62, 1.63983936034272e-63, 7.2733282246436055e-65, 3.178563972720736e-66,
                       1.3689530341240694e-67, 5.81161975635863e-69, 2.4324589817766242e-70, 1.0039676333592717e-71,
                       4.0869896192072334e-73, 1.641264204267607e-74, 6.503152181277399e-76, 2.54282813285395e-77,
                       9.813704239361523e-79, 3.738910293655332e-80, 1.4064510995777517e-81, 5.224459089388231e-83,
                       1.916739075552318e-84, 6.946336646767506e-86, 2.4870494334135323e-87, 8.798564341979278e-89,
                       3.076093778836878e-90, 1.062937680850308e-91, 3.6307406960889297e-93, 1.2260812297245853e-94,
                       4.093887649711318e-96, 1.3517615633271327e-97, 4.414335970905034e-99, 1.425884142857282e-100,
                       4.556254488972261e-102, 1.4404122386480614e-103, 4.505778934574749e-105, 1.394778775492688e-106,
                       4.273072668833961e-108, 1.2957490100371349e-109, 3.8894874446897226e-111]
            def score_decoding(decoding_idxs):
                """likelihood function for a poisson distribution fit
                to the training distribution of number-of-relationships-per-document

                :param decoding_idxs (List[int]): a list of idxs
                :return: likelihood of a single decoding
                """
                try:
                    score_from_n_relationships = poisson[len(decoding_idxs)]
                except IndexError:
                    score_from_n_relationships = 0
                score_from_indiv_relationships = [self._candidate_scores.get(k) for k in decoding_idxs]
                return score_from_n_relationships * np.prod(score_from_indiv_relationships)
            idxs_sorted_by_score = sorted(self._candidate_labels, key=self._candidate_scores.get)
            possible_decoding_idxs = [idxs_sorted_by_score[:i] for i in range(1, len(idxs_sorted_by_score))]
            best_decoding_idxs = max(possible_decoding_idxs, key=score_decoding)
            prediction = [1 if k in best_decoding_idxs else 0 for k in self._candidate_labels]
        try:
            metrics = pd.io.json.json_normalize(
                classification_report(gold, prediction, output_dict=True), sep="."
            ).to_dict(orient="records")[0]
        except:
            breakpoint()
        metrics = {k.replace(" ", "-"): v for k, v in metrics.items()}

        metrics["1.support_pred"] = sum(prediction)
        metrics["0.support_pred"] = len(prediction) - sum(prediction)

        metrics["threshold"] = float(threshold)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return metrics

    @overrides
    def reset(self):
        self._candidate_labels = {}
        self._candidate_scores = {}
