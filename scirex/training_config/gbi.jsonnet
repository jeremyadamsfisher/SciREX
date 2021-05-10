// Import template file.

local template = import "template_full.libsonnet";

// Set options.

local params = {
  use_lstm: true,
  bert_fine_tune: std.extVar("bert_fine_tune"),
  loss_weights: {          // Loss weights for the modules.
    ner: std.extVar('nw'),
    saliency: std.extVar('lw'),
    n_ary_relation: std.extVar('rw')
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
  exact_match: std.extVar('em'),
  regularizer: [
    ["original_model", {
     "archive_file": "outputs/pwc_outputs/experiment_scirex_full/main/model.tar.gz",
     "path": "scirex.models.scirex_model.ScirexModel",
     "freeze": true
    }],
    ["^_cluster_n_ary_relation.*", {"type": "l2", "alpha": 0.01}]
  ],
  initializer: [
    [".*", {
            "type": "pretrained",
            "weights_file_path": "outputs/pwc_outputs/experiment_scirex_full/main/best.th",
            "parameter_name_overrides": {},
    }]
  ]
};

template(params)