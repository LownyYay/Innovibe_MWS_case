"""Portable inference for a train-only histogram-boosted date ranker (stdlib)."""
import math


def probability(features, model):
    value = model["bias"]
    for tree in model["trees"]:
        i = 0
        while not tree[i][0]:
            _, _, feature, threshold, left, right = tree[i]
            i = left if features[feature] <= threshold else right
        value += tree[i][1]
    return 1 / (1 + math.exp(-max(-60, min(60, value))))


def export_model(estimator):
    trees = []
    for stage in estimator._predictors:
        nodes = stage[0].nodes
        assert not nodes["is_categorical"].any()
        trees.append([[int(n["is_leaf"]), float(n["value"]), int(n["feature_idx"]),
                       float(n["num_threshold"]), int(n["left"]), int(n["right"])] for n in nodes])
    return {"bias": float(estimator._baseline_prediction.ravel()[0]), "trees": trees}


def choose(options, model):
    scored = sorted([(probability(x, model), i) for i, (_, x, _) in enumerate(options)], reverse=True)
    p, i = scored[0]
    return options[i][0], p, p - scored[1][0] if len(scored) > 1 else p, options[i][2]
