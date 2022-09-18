import json
import numpy as np
from scipy.stats.stats import pearsonr
import scipy
import scipy.stats as st

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if __name__ == "__main__":
    datasets = ["helpdesk_5",
                "helpdeskC_5",
                "bpic12_w_5",
                "bpic12_w_complete_5",
                "bpic13_closed_problems_5",
                "bpic12_o_5",
                "bpic12_a_5",
                "bpic13_i_5",
                "bpic12_5"
                ]
    exp = "ex2"

    nap_unweighted = []
    napm_unweighted = []

    nap_weighted = []
    napm_weighted = []

    nap_N_AUC = []
    napm_N_AUC = []

    for dataset in datasets:
        results = None
        with open("../performend_experiments/" + exp + "/results/" + dataset + ".json") as json_file:
            results = json.load(json_file)
            for k in results.keys():
                print("*** Fold", k, dataset, "***")
                print("Next Event in Mask (train): ", results[k]["next_event_in_mask_ratio_train"])
                print("Next Event in Mask (val): ", results[k]["next_event_in_mask_ratio_val"])
                print("Next Event in Mask (test): ", results[k]["next_event_in_mask_ratio_test"])
                print()

                nap_unweighted.append(results[k]["nap"]["unweighted mAUC"])
                napm_unweighted.append(results[k]["napm"]["unweighted mAUC"])

                nap_weighted.append(results[k]["nap"]["weighted mAUC"])
                napm_weighted.append(results[k]["napm"]["weighted mAUC"])

                for t in list(zip(results[k]["nap"]["N per AUC"], results[k]["nap"]["OVR AUCs"])):
                    nap_N_AUC.append(t)

                for t in list(zip(results[k]["napm"]["N per AUC"], results[k]["napm"]["OVR AUCs"])):
                    napm_N_AUC.append(t)

    print()
    print("*** Unweighted AUC Analysis ***")
    napm_wins, napm_loss, napm_diff = 0, 0, []
    for i, val in enumerate(napm_unweighted):
        diff = val - nap_unweighted[i]
        if diff > 0:
            napm_wins += 1
        else:
            napm_loss += 1
        napm_diff.append(diff)
    napm_diff = np.array(napm_diff)
    print("NAPM Wins/Loss:", napm_wins, "/", napm_loss)
    print("NAPM Win Ratio:", float(napm_wins/(napm_wins+napm_loss)))
    print("Mean NAPM Difference:", np.mean(napm_diff))
    print("Std. NAPM Difference:", np.std(napm_diff))
    print("NAPM Range:", np.mean(napm_diff) - np.std(napm_diff), np.mean(napm_diff) + np.std(napm_diff))
    print("NAPM 95% CI:", st.t.interval(0.95, len(napm_diff)-1, loc=np.mean(napm_diff), scale=st.sem(napm_diff)))
    print()

    print("*** Weighted AUC Analysis ***")
    napm_wins, napm_loss, napm_diff = 0, 0, []
    for i, val in enumerate(napm_weighted):
        diff = val - nap_weighted[i]
        if diff > 0:
            napm_wins += 1
        else:
            napm_loss += 1
        napm_diff.append(diff)
    napm_diff = np.array(napm_diff)
    print("NAPM Wins/Loss:", napm_wins, "/", napm_loss)
    print("NAPM Win Ratio:", float(napm_wins / (napm_wins + napm_loss)))
    print("Mean NAPM Difference:", np.mean(napm_diff))
    print("Std. NAPM Difference:", np.std(napm_diff))
    print("NAPM Range:", np.mean(napm_diff) - np.std(napm_diff), np.mean(napm_diff) + np.std(napm_diff))
    print("NAPM 95% CI:", st.t.interval(0.95, len(napm_diff) - 1, loc=np.mean(napm_diff), scale=st.sem(napm_diff)))
    print()

    print("*** Correlation Analysis between N and OVR AUCs  ***")
    napm_n, napm_auc = [], []
    nap_n, nap_auc = [], []
    for t in nap_N_AUC:
        nap_n.append(t[0])
        nap_auc.append(t[1])
    for t in napm_N_AUC:
        napm_n.append(t[0])
        napm_auc.append(t[1])

    print("Correlation N->OVRAUC NAP:", pearsonr(nap_n, nap_auc))
    print("Correlation N->OVRAUC NAPM:", pearsonr(napm_n, napm_auc))
    print()