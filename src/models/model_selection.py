from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate


def find_optimal_tfidf_params(X, y,
                              param_grid: Dict[str, List],
                              models: Dict[str, object],
                              cv_kwargs: Optional[Dict]=None,
                              random_state: Optional[int]=7,
                              path: Optional[str]=None,
                              warm_start: Optional[List]=None,
                              verbose: Optional[bool]=False) -> pd.DataFrame:
    """
    Function performs grid search across Tfidf parameters from param_grid with input data X and
    output data y. Processed data is evaluated on models from `models` dict.

    Args:
        X: array like, input values for model. Should be column with plain text provided to Tfidf
        y: array like, output values for model. Encoded labels of classes from dataset
        param_grid: grids of parameters for tfidf. You can specify anything but it is dedicated
                    especially for ngram_range and max_features
        models: dictionary where key is name of model and value is specified sklearn model
        cv_kwargs: additional keyword arguments for cross_validate function. Passed as **cv_kwargs
        random_state: random seed for experiment
        path: path to save result. If specified results will be save for each param combitaion to dismiss
            loss of progress
        warm_start: used when your experiment stopped before end. It contains list of dictionary with
                    parameters computed in previous experiment. If you constantly saved previous results
                    you can check which combinations were calculated then exclude them from new compilation
        verbose: if true then progress raports will be printed

    Returns:
        results: data frame with name of model, parameter combination and list of score solutions from cross validation
    """

    # Setting up random state
    np.random.seed(random_state)
    # Defining metrics, Possible extension TODO : custom metrics
    scoring = ['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted']

    # If path to results actually exists then read it.
    # Possible extension TODO : add parameter which wpecify if you want to overwrite results
    try:
        results = _read_results(path)
    except:
        # Initializing empty dataframe with results
        results = _create_results()

    # If you actually have computed some results
    if warm_start:
        # Creating every combination of ngrams and max tfidf features which are not in warm start combinations
        params_combinations = [params
                               for ngram in param_grid['ngram_range']
                               for max_feature in param_grid['max_features']
                               # params is current params object we filter it with respect to values from warm_start
                               if (params := {'ngram_range': ngram, 'max_features': max_feature}) not in warm_start]
    else:
        # Creating every combination of ngrams and max tfidf features
        params_combinations = [{'ngram_range': ngram, 'max_features': max_feature}
                               for ngram in param_grid['ngram_range']
                               for max_feature in param_grid['max_features']]

    if verbose:
        # Initializing iteration parameters
        # Number of iteration is number of combinations times number of models trained
        max_iter = len(params_combinations) * len(models)
        iteration = 0
        print(f'Iteration {1}/{max_iter}')

    # We start with enumerate 1 for correct verbose iterator
    for i, params in enumerate(params_combinations, 1):
        # Defining tfidf instance with given params
        tfidf = TfidfVectorizer(**params)
        X_tfidf = tfidf.fit_transform(X).toarray()

        # For each model selected to experiment
        for name, model in models.items():
            # Cross validaiton scores
            cv = cross_validate(model, X_tfidf, y, scoring=scoring, **cv_kwargs)
            # Rounding long digits to .2 points
            round_cv_results(cv)
            # Temporary dataframe with given parameters
            temp = cv_to_dataframe(cv, name, **params)
            # Appending results
            results = results.append(temp, ignore_index=True)
        # Saving current result
        if path:
            results.to_csv(path, index=False)
        # Raporting progess
        if verbose and i % 3 == 0:
            iteration += 3 * len(models)
            print(f'Iteration {iteration}/{max_iter}')
    # Ending function
    if verbose:
        print(f'Iteration {max_iter}/{max_iter}')
    return results


def _create_results() -> pd.DataFrame:
    """
    helper function to create empty dataframe for results

    Returns:
        pd.DataFrame with column Name (of model), params and scores
    """
    return pd.DataFrame({'Name': [],
                         'max_features': [],
                         'ngram_range': [],
                         'Accuracy': [],
                         'Recall': [],
                         'Precision': [],
                         'F1': []})


def cv_to_dataframe(cv: Dict,
                    name: str,
                    max_features: Optional[int]=None,
                    ngram_range: Optional[Tuple[int, int]]=None) -> pd.DataFrame:
    """
    Function returns dataframe with one row of solution from cross validation

    Args:
        cv: result of `cross_validate` function
        name: name of model used for evaluate
        max_features: used max_features for preprocessing
        ngram_range:  used ngram_range for preprocessing

    Returns:
        pd.DataFrame with one row of solution
    """
    return pd.DataFrame({'Name': [name],
                         'max_features': [max_features],
                         'ngram_range': [ngram_range],
                         'Accuracy': [cv['test_accuracy']],
                         'Recall': [cv['test_recall_weighted']],
                         'Precision': [cv['test_precision_weighted']],
                         'F1': [cv['test_f1_weighted']]})


def round_cv_results(cv: Dict) -> None:
    """
    Function rounds values of scores for cross validation to 2nd digit

    Args:
        cv: result of `cross_validate` function

    Returns:
        None
    """
    # Iterating over test metrics from cross validation
    for key in ['test_accuracy', 'test_recall_weighted', 'test_precision_weighted', 'test_f1_weighted']:
        values = cv[key]
        # rounding values
        cv[key] = [np.round(i, 2) for i in values]


def _read_results(path: str) -> pd.DataFrame:
    """
    Helper function to read data frame with results and make columns "usable"

    Args:
        path: path to result data frame

    Returns:
        results: data frame with results
    """
    # Reading csv file
    results = pd.read_csv(path)
    # Converting max_features to int
    results['max_features'] = results['max_features'].astype(int)
    # Last columns should contains list of scores from cross validation
    # We need to apply eval to them to make them "usable"
    for column in results.columns[3:]:
        results[column] = results[column].apply(eval)
    return results