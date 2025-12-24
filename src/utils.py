import pickle
from pathlib import Path

from sklearn.metrics import log_loss, recall_score, fbeta_score, roc_auc_score

from src.exception import CustomException

def save_object(obj, path: Path|str):
    try:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
            f.close()
            
    except Exception as e:
        raise CustomException(e)

def get_metric_scores(y_true, y_pred):
    logloss = log_loss(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2score = fbeta_score(y_true, y_pred, beta=2.0)
    auc = roc_auc_score(y_true, y_pred)

    return {'logloss': logloss, 'recall': recall, 'f2score': f2score, 'auc': auc}

def evaluate_models(X_train, y_train, X_test, y_test, models:dict):
    """Returns a dict of models and their scores on the test data across a set of metrics."""
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_metric_scores(y_test, y_pred)
        model_scores[name] = metrics
    return model_scores