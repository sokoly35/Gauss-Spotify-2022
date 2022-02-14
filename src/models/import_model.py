from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Filtering tpot warning about absence of torch library
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tpot.builtins import StackingEstimator


def define_tpot_model():
    """
    Function defines model obtained via tpot experiment

    Returns:
        model: tpot model object
    """
    # Creating pipeline
    model = make_pipeline(
        # Defining two stack estiamtors of Bernoulli
        StackingEstimator(estimator=BernoulliNB(alpha=0.1, fit_prior=False)),
        StackingEstimator(estimator=BernoulliNB(alpha=0.1, fit_prior=False)),
        # Defububg certain xgb model which enables label encoder outputs
        XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=3,
                      n_estimators=100, n_jobs=-1, subsample=0.5, verbosity=0,
                      random_state=7, use_label_encoder=False))
    return model
