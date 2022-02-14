from typing import Optional, Dict, Tuple, List
from time import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

def find_optimal_tfidf_params(X, y,
                              param_grid: Dict[str, List],
                              models: Dict[str, object],
                              cv_kwargs: Optional[Dict] = None,
                              random_state: Optional[int] = 7,
                              path: Optional[str] = None,
                              warm_start: Optional[List] = None,
                              verbose: Optional[bool] = False) -> pd.DataFrame:
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
                    max_features: Optional[int] = None,
                    ngram_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
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


def search_preprocessors(model, df: pd.DataFrame, y,
                         text_columns: List[str],
                         tfidf_params: Dict[str, List],
                         bow_params: Dict[str, List],
                         scalers: Optional[List]=[None],
                         cv_kwargs: Optional[Dict]=None,
                         random_state: Optional[int]=None,
                         path: Optional[str]=None,
                         verbose: Optional[bool]=None) -> pd.DataFrame:
    """
    Function performs experiment to search over best preprocessing pipeline for given problem.
    The proposed text preprocessors are Tfidf and Bag of Words methods. We can inspect different types
    of scaling results.

    Args:
        model: Main model to evaluate results
        df: Input values. It should contain columns from `text_columns` list
        y: Output values. May be before encoding. Label encoding will be carry out regardless of y's dtype
        text_columns: list of columns with tokens or text send further to preprocessors
        tfidf_params: Param grid of TfIdf preprocessor
        bow_params: Param grid of Bag of Word preprocessor
        scalers: List of scalers used in experiment. If None then no scaling method will be used
        cv_kwargs: Additional arguments of GridSearchCV method
        random_state: initial random state
        path: path to saving results
        verbose: if true then short raport will be printed after each iteration of algorithm

    Returns:

    """
    # Setting the random state
    if random_state:
        np.random.seed(random_state)
    # Defining metrics to evaluate
    scoring = ['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted']

    # Altering the keys of tfidf and bow dictionaries
    # We define further sklearn Pipeline which needs a refer of parameters to specific objects
    tfidf_params = {'preprocessor__' + k: v for k, v in tfidf_params.items()}
    bow_params = {'preprocessor__' + k: v for k, v in bow_params.items()}

    # Encoding output
    le = LabelEncoder()
    y = le.fit_transform(y.copy())

    # If we want a raport after iterations
    if verbose:
        # Initializing iteration parameters
        # Number of iteration is number of scalres * number of columns on which we train
        # And times 2 because of Tfidf and Bag of words
        max_iter = len(scalers) * len(text_columns) * 2
        iteration = 0
        print(f'Iteration {0}/{max_iter}')

    # Iterating over usefull columns
    for column in text_columns:
        # Preprocessing input if column caontains lists instead of text
        if isinstance(df[column][0], list):
            # Joining list of words to a plain text
            X = df.copy()[column].apply(lambda x: ' '.join(x))

        # Avaible preprocessors
        preprocessors = [TfidfVectorizer(), CountVectorizer()]
        # Avaible preprocessors keyword args
        preprocessors_kwargs = [tfidf_params, bow_params]

        # Iterating over preprocessors and corresponding args
        for prep, prep_kwargs in zip(preprocessors, preprocessors_kwargs):
            # Saving name of preprocessor
            prep_name = type(prep).__name__
            # Iterating over avaible scalers
            for scaler in scalers:
                # Saving current scaler name
                scaler_name = type(scaler).__name__
                # If there is no scaler then we alter a bit pipeline
                if not scaler:
                    pipeline = Pipeline([('preprocessor', prep),
                                         ('to_dense', FunctionTransformer(lambda x: x.toarray())),
                                         ('model', model)])
                # If we scale the data then we need to add scaler step in pipeline
                else:
                    pipeline = Pipeline([('preprocessor', prep),
                                         ('to_dense', FunctionTransformer(lambda x: x.toarray())),
                                         ('scaler', scaler),
                                         ('model', model)])
                # Measuring time of grid search
                t = time()
                # Defining and compiling grid search on given set of parameters
                # We pass there additional parameters from cv_kwargs dict
                grid_search = GridSearchCV(pipeline, prep_kwargs, scoring=scoring, **cv_kwargs)
                grid_search.fit(X, y)
                # Preprocessing result to more usable form
                temp = _reading_grid_search_results(column, prep_name, scaler_name,
                                                    grid_search.cv_results_)
                # If the result data frame already exists then just append current results
                # If the error occurs then it means it is first iteration and we need to define results
                try:
                    results = results.append(temp, ignore_index=True)
                except:
                    results = temp
                # If we specified path to save results then we save the progress of experiment
                if path:
                    results.to_csv(path, index=False)
                # Raport after end of iteration
                if verbose:
                    iteration += 1
                    t2 = (time() - t)
                    print(f'Iteration {iteration}/{max_iter}. Time: {t2 / 60: .2f} min')
    # Saving results
    if path:
        results.to_csv(path, index=False)
    return results


def _reading_grid_search_results(col_name: str,
                                 prep_name: str,
                                 scaler_name: str,
                                 cv_results: Dict,
                                 scoring: Optional[List[str]]=['accuracy', 'recall_weighted',
                                                               'precision_weighted', 'f1_weighted']) -> pd.DataFrame:

    # Poping params of each grid search iteration
    params = cv_results['params']
    # Number of each combination
    n_combinations = len(params)
    # We create lists of column, preprocessors and scalers names used for given grid search
    col_names = [col_name for _ in range(n_combinations)]
    prep_names = [prep_name for _ in range(n_combinations)]
    scaler_names = [scaler_name for _ in range(n_combinations)]
    # List of mean scores of each metric in the order as `params` was evaluated
    scores = [list(np.round(cv_results[f'mean_test_{metric}'], 2))
              for metric in scoring]
    # Creating initial data frame
    dict_result = {'On Column': col_names,
                   'Preprocessor': prep_names,
                   'Scaler': scaler_names,
                   'Params': params}
    # Appending metric columns to dataframe
    for metric, result in zip(scoring, scores):
        # column names are capitalized and without anuthing after _ char
        dict_result[metric.split('_')[0].capitalize()] = result
    return pd.DataFrame(dict_result)
