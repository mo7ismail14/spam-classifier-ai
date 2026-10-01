"""Trace a Decision Tree's decision path for one sample into plain-English rules."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from functions.decisiontree.data_preprocessing import FEATURE_DISPLAY_NAMES, BINARY_FEATURES


def _describe_condition(feature_name: str, raw_value, threshold: float, went_left: bool) -> str:
    """Turn one tree split into a plain-English clause using the applicant's own value."""
    display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)

    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        # The applicant left this field blank; the preprocessor imputed a value for the
        # tree split, but we should not claim to know the applicant's actual value.
        return f"{display_name} is not provided (the model filled in a typical value)"

    if feature_name in BINARY_FEATURES:
        # Credit_History: 0/1 stored as a number but really means "poor"/"good".
        return f"{display_name} is {'good' if float(raw_value) >= 0.5 else 'poor'}"

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        direction = 'below' if went_left else 'above'
        return f"{display_name} ({raw_value:g}) is {direction} the model's threshold of {threshold:.0f}"

    return f"{display_name} is '{raw_value}'"


def explain_prediction(
    tree: DecisionTreeClassifier,
    preprocessor: ColumnTransformer,
    feature_order: list[str],
    raw_input: dict,
    prediction_label: str
) -> str:
    """Walk the tree's decision path for raw_input and describe it in plain English."""
    input_df = pd.DataFrame([{col: raw_input.get(col) for col in feature_order}])
    encoded_input = preprocessor.transform(input_df)

    node_indicator = tree.decision_path(encoded_input)
    leaf_id = tree.apply(encoded_input)[0]
    sample_node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    feature_index_per_node = tree.tree_.feature
    threshold_per_node = tree.tree_.threshold

    seen_features = set()
    clauses = []
    for node_id in sample_node_ids:
        if node_id == leaf_id:
            continue  # leaf nodes carry no split condition

        feature_index = feature_index_per_node[node_id]
        feature_name = feature_order[feature_index]
        if feature_name in seen_features:
            continue
        seen_features.add(feature_name)

        threshold = threshold_per_node[node_id]
        sample_value = encoded_input[0, feature_index]
        went_left = sample_value <= threshold

        clauses.append(_describe_condition(feature_name, raw_input.get(feature_name), threshold, went_left))

    verdict = 'approved' if prediction_label == 'Y' else 'denied'
    if not clauses:
        return f"Loan {verdict} based on the applicant's overall profile."

    reasons = '; '.join(clauses)
    return f"Loan {verdict} because {reasons}."
