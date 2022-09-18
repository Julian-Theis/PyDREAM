import json
import numpy as np
from scipy.stats.stats import pearsonr


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
    exp = "ex1"
    weighted = True
    precision = 3

    nap_unweighted = []
    napm_unweighted = []

    nap_weighted = []
    napm_weighted = []

    napm_weighted_wins = 0
    nap_weighted_wins = 0

    napm_unweighted_wins = 0
    nap_unweighted_wins = 0


    for dataset in datasets:
        results = None
        with open("../performend_experiments/" + exp + "/results/" + dataset + ".json") as json_file:
            results = json.load(json_file)
            for k in results.keys():
                nap_unweighted.append(results[k]["nap"]["unweighted mAUC"])
                napm_unweighted.append(results[k]["napm"]["unweighted mAUC"])
                if results[k]["nap"]["unweighted mAUC"] > results[k]["napm"]["unweighted mAUC"]:
                    nap_unweighted_wins += 1
                else:
                    napm_unweighted_wins += 1

                nap_weighted.append(results[k]["nap"]["weighted mAUC"])
                napm_weighted.append(results[k]["napm"]["weighted mAUC"])
                if results[k]["nap"]["weighted mAUC"] > results[k]["napm"]["weighted mAUC"]:
                    nap_weighted_wins += 1
                else:
                    napm_weighted_wins += 1



    if weighted:
        for k in results.keys():
            print("Fold ", k, " NAPM:", np.around(results[k]["napm"]["weighted mAUC"], decimals=precision))
            print("Fold ", k, " NAP:", np.around(results[k]["nap"]["weighted mAUC"], decimals=precision))
            print()

        print("Mean NAPM:", np.around(np.mean(napm_weighted), decimals=precision))
        print("SD NAPM:", np.around(np.std(napm_weighted), decimals=precision))
        print("Wins NAPM:", napm_weighted_wins)
        print()
        print("Mean NAP:", np.around(np.mean(nap_weighted), decimals=precision))
        print("SD NAP:", np.around(np.std(nap_weighted), decimals=precision))
        print("Wins NAP:", nap_weighted_wins)

    if not weighted:
        for k in results.keys():
            print("Fold ", k, " NAPM:", np.around(results[k]["napm"]["unweighted mAUC"], decimals=precision))
            print("Fold ", k, " NAP:", np.around(results[k]["nap"]["unweighted mAUC"], decimals=precision))
            print()

        print("Mean NAPM:", np.around(np.mean(napm_unweighted), decimals=precision))
        print("SD NAPM:", np.around(np.std(napm_unweighted), decimals=precision))
        print("Wins NAPM:", napm_unweighted_wins)
        print()
        print("Mean NAP:", np.around(np.mean(nap_unweighted), decimals=precision))
        print("SD NAP:", np.around(np.std(nap_unweighted), decimals=precision))
        print("Wins NAP:", nap_unweighted_wins)

