export test_file=scirex_dataset/release_data/test.jsonl
export test_output_folder=test_outputs/

echo "Predicting Relations End-to-End" \
&& python scirex/predictors/predict_n_ary_relations.py \
$scirex_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl \
$test_output_folder/relations_predictions.jsonl \
$cuda_device \
&& echo "Evaluating on all Predicted steps " \
&& python scirex/evaluation_scripts/scirex_relation_evaluate.py \
--gold-file $test_file \
--ner-file $test_output_folder/ner_predictions.jsonl \
--clusters-file $test_output_folder/salient_clusters_predictions.jsonl \
--relations-file $test_output_folder/relations_predictions.jsonl
