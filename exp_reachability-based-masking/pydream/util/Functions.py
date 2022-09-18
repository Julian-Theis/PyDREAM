import datetime
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score


def time_delta_seconds(ts_start: datetime.datetime,
                       ts_end: datetime.datetime):
    if not isinstance(ts_start, datetime.datetime) or not isinstance(ts_end, datetime.datetime):
        raise ValueError('The timestamps are not of <class \'datetime.datetime\'>')
    return (ts_end - ts_start).total_seconds()


def multiclass_roc_auc_score(y_test: np.ndarray,
                             y_pred: np.ndarray,
                             average: str = "weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
