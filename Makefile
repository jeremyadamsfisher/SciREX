train_main:
	SCIREX_PROBABILISTIC_DECODING=1 \
	CUDA_DEVICE=0 \
	bash scirex/commands/train_scirex_model.sh main


train_multitask:
	CUDA_DEVICE=0 bash scirex/commands/train_pairwise_coreference.sh main


preds:
	scirex_archive=outputs_backup/pwc_outputs/experiment_scirex_full/main \
	scirex_coreference_archive=outputs_backup/pwc_outputs/experiment_coreference/main \
	cuda_device=0 \
    bash scirex/commands/predict_scirex_model.sh


gbi:
	CUDA_DEVICE=0 \
	bash scirex/commands/train_scirex_model_gbi.sh gbi


_decoding:
	SCIREX_RELATION_DECODING=$(DECODING_MODE) \
	DECODING_METRICS_OUTFP=relation_decoding_analysis/$(DECODING_MODE)_metrics.json \
	scirex_archive=outputs_backup/pwc_outputs/experiment_scirex_full/main \
	scirex_coreference_archive=outputs_backup/pwc_outputs/experiment_coreference/main \
	cuda_device=0 \
    bash scirex/commands/decoding_evaluation.sh


decoding:
	$(MAKE) _decoding DECODING_MODE="report_single_most_likely"
	$(MAKE) _decoding DECODING_MODE="baseline"
	$(MAKE) _decoding DECODING_MODE="report_probabilistically"


all: train_main train_multitask