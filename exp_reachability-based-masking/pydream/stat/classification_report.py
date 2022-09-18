import pandas as pd
import numpy as np

from  sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from pydream.stat.util import print_cm
from pydream.stat.ci_auc import calculate_auc_ci

def class_report(y_true, y_pred, y_score, alpha, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    """
    matrix = confusion_matrix(y_true, y_pred)
    accs = matrix.diagonal() / matrix.sum(axis=1)
    print("accuracies")
    print(accs)
    """

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        auc_delong = dict()
        auc_ci = dict()
        auc_cov = dict()
        accs = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            y_true_imed = (y_true == label).astype(int)
            y_pred_imed = (y_pred == label).astype(int)
            y_score_imed = y_score[:, label_it]

            auc_dl, auc_co, ci = calculate_auc_ci(y_pred=y_pred_imed, y_true=y_true_imed, y_score=y_score_imed, alpha=alpha, print_results=False)
            auc_delong[label] = auc_dl
            auc_cov[label] = auc_co
            auc_ci[label] = ci

            accs[label] = accuracy_score(y_true=y_true_imed, y_pred=y_pred_imed)

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        accs["avg / total"] = np.mean(list(accs.values()))
        auc_delong["avg / total"] = ""#np.mean(list(auc_delong.values()))
        auc_cov["avg / total"] = "" #np.mean(list(auc_cov.values()))
        auc_ci["avg / total"] = ""


        class_report_df['accuracy'] = pd.Series(accs)
        class_report_df['AUC'] = pd.Series(roc_auc)

        class_report_df['AUC DeLong'] = pd.Series(auc_delong)
        class_report_df['AUC COV'] = pd.Series(auc_cov)
        class_report_df['AUC CI (' + str(alpha*100) + ' %)'] = pd.Series(auc_ci)


    return class_report_df


def run_classification_report(y_pred, y_true, y_score, alpha=0.95, print_results=True):
    class_report_df = class_report(y_true=y_true, y_pred=y_pred, y_score=y_score, alpha=float(alpha))

    if print_results == True:
        print("*** Confusion Matrix ***")
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        print_cm(cm, labels)
        print()

        print("*** Classification Report ***")
        print(class_report_df)


    return class_report_df