import json
import numpy as np
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri.importer import importer as pnml_importer

from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN
from pydream.predictive.nap.NAP import NAP
from pydream.predictive.nap.NAPM import NAPM

from sklearn.metrics import roc_auc_score

# 50% Dropout source: https://www.reddit.com/r/MachineLearning/comments/3oztvk/why_50_when_using_dropout/
if __name__ == "__main__":
    dataset = "sepsis"
    k = 5
    options = {"n_epochs": 150,
               "dropout_rate": 0.20}


    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
    log = pd.read_csv("../data/" + dataset + ".csv")
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    log = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    results = dict()
    for c in range(1, k+1):
        print("*** FOLD ", str(c), " ***")

        log_train = pd.read_csv("../data/" + dataset + "_" + str(c) + "_train.csv")
        log_train["time:timestamp"] = pd.to_datetime(log_train["time:timestamp"])
        log_train = log_converter.apply(log_train, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        log_wrapper_train = LogWrapper(log_train)

        log_val = pd.read_csv("../data/" + dataset + "_" + str(c) + "_validate.csv")
        log_val["time:timestamp"] = pd.to_datetime(log_val["time:timestamp"])
        log_val = log_converter.apply(log_val, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        log_wrapper_val = LogWrapper(log_val)

        log_test = pd.read_csv("../data/" + dataset + "_" + str(c) + "_test.csv")
        log_test["time:timestamp"] = pd.to_datetime(log_test["time:timestamp"])
        log_test = log_converter.apply(log_test, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        log_wrapper_test = LogWrapper(log_test)

        net, initial_marking, final_marking = pnml_importer.apply("../data/models/" + dataset + ".pnml")

        enhanced_pn = EnhancedPN(net, initial_marking)
        enhanced_pn.enhance(log_wrapper_train)
        enhanced_pn.explore_reachability_graph(log)
        enhanced_pn.saveToFile("../data/models/" + dataset + "_" + str(c) + "_enhanced.json")

        #enhanced_pn.decay_replay(log_wrapper=log_wrapper_train, update_rg=True)
        #enhanced_pn.decay_replay(log_wrapper=log_wrapper_val, update_rg=True)
        #enhanced_pn.decay_replay(log_wrapper=log_wrapper_test, update_rg=True)

        tss_json_train, tss_objs_train = enhanced_pn.decay_replay(log_wrapper=log_wrapper_train, update_rg=False, create_masks=True)
        tss_json_val, tss_objs_val = enhanced_pn.decay_replay(log_wrapper=log_wrapper_val, update_rg=False, create_masks=True)
        tss_json_test, tss_objs_test = enhanced_pn.decay_replay(log_wrapper=log_wrapper_test, update_rg=False, create_masks=True)

        next_event_in_mask_ratio_train = 0
        next_event_in_mask_ratio_val = 0
        next_event_in_mask_ratio_test = 0
        for tss in tss_objs_train:
            if tss.export()["nextEventInMask"]:
                next_event_in_mask_ratio_train += 1
        for tss in tss_objs_val:
            if tss.export()["nextEventInMask"]:
                next_event_in_mask_ratio_val += 1
        for tss in tss_objs_test:
            if tss.export()["nextEventInMask"]:
                next_event_in_mask_ratio_test += 1
        next_event_in_mask_ratio_train = next_event_in_mask_ratio_train / len(tss_objs_train)
        next_event_in_mask_ratio_val = next_event_in_mask_ratio_val / len(tss_objs_val)
        next_event_in_mask_ratio_test = next_event_in_mask_ratio_test / len(tss_objs_test)

        results_sample = dict()
        results_sample["next_event_in_mask_ratio_train"] = next_event_in_mask_ratio_train
        results_sample["next_event_in_mask_ratio_val"] = next_event_in_mask_ratio_val
        results_sample["next_event_in_mask_ratio_test"] = next_event_in_mask_ratio_test
        print("next_event_in_mask_ratio_train", next_event_in_mask_ratio_train)
        print("next_event_in_mask_ratio_val", next_event_in_mask_ratio_val)
        print("next_event_in_mask_ratio_test", next_event_in_mask_ratio_test)

        with open("../data/" + dataset + "_" + str(c) + "_tss_train.json", 'w') as fp:
            json.dump(tss_json_train, fp)
        with open("../data/" + dataset + "_" + str(c) + "_tss_val.json", 'w') as fp:
            json.dump(tss_json_val, fp)
        with open("../data/" + dataset + "_" + str(c) + "_tss_test.json", 'w') as fp:
            json.dump(tss_json_test, fp)

        """ NAPM EVALUATION """
        algo = NAPM(tss_train_file="../data/" + dataset + "_" + str(c) + "_tss_train.json",
                    tss_val_file="../data/" + dataset + "_" + str(c) + "_tss_val.json",
                    tss_test_file="../data/" + dataset + "_" + str(c) + "_tss_test.json",
                    options=options)
        algo.train(checkpoint_path="../data/models",
                   name=dataset + "_" + str(c),
                   save_results=True)
        nap_preds, next_events, scores = algo.predict(tss_objs_test)
        label_vec = []
        for i, tss in enumerate(tss_objs_test):
            try:
                label = algo.one_hot_dict[tss.data["nextEvent"]]
            except:
                continue
            label_vec.append(label)

        label_vec = np.array(label_vec)

        print("*** OUTCOMES FOR NAPM ***")
        try:
            wmulticlass_auc = roc_auc_score(label_vec, scores, multi_class = 'ovr', average = 'weighted')
            print("Weighted mAUC:", wmulticlass_auc)
        except Exception as err:
            print(err)

        try:
            umulticlass_auc = roc_auc_score(label_vec, scores, multi_class='ovr', average='macro')
            print("Unweighted Mean mAUC:", umulticlass_auc)
        except Exception as err:
            print(err)

        try:
            multiclass_aucs = roc_auc_score(label_vec, scores, multi_class='ovr', average=None)
            print("OVR AUCs:", multiclass_aucs)
        except Exception as err:
            print(err)

        nperauc = np.sum(label_vec, axis=0)
        print("N per AUC:", nperauc)


        results_sample["napm"] = {
            "unweighted mAUC" : umulticlass_auc,
            "weighted mAUC" : wmulticlass_auc,
            "OVR AUCs" : multiclass_aucs.tolist(),
            "N per AUC": nperauc.tolist()
        }
        print()

        """ NAP EVALUATION """
        options_nap = options
        options_nap["dropout_rate"] = 0.2

        algo = NAP(tss_train_file="../data/" + dataset + "_" + str(c) + "_tss_train.json",
                    tss_val_file="../data/" + dataset + "_" + str(c) + "_tss_val.json",
                    tss_test_file="../data/" + dataset + "_" + str(c) + "_tss_test.json",
                    options=options_nap)
        algo.train(checkpoint_path="../data/models",
                   name=dataset + "_" + str(c),
                   save_results=True)
        nap_preds, next_events, scores = algo.predict(tss_objs_test)
        label_vec = []
        for i, tss in enumerate(tss_objs_test):
            try:
                label = algo.one_hot_dict[tss.data["nextEvent"]]
            except:
                continue
            label_vec.append(label)

        label_vec = np.array(label_vec)

        print("*** OUTCOMES FOR NAP ***")
        try:
            wmulticlass_auc = roc_auc_score(label_vec, scores, multi_class='ovr', average='weighted')
            print("Weighted mAUC:", wmulticlass_auc)
        except Exception as err:
            print(err)

        try:
            umulticlass_auc = roc_auc_score(label_vec, scores, multi_class='ovr', average='macro')
            print("Unweighted Mean mAUC:", umulticlass_auc)
        except Exception as err:
            print(err)

        try:
            multiclass_aucs = roc_auc_score(label_vec, scores, multi_class='ovr', average=None)
            print("OVR AUCs:", multiclass_aucs)
        except Exception as err:
            print(err)

        nperauc = np.sum(label_vec, axis=0)
        print("N per AUC:", nperauc)

        results_sample["nap"] = {
            "unweighted mAUC": umulticlass_auc,
            "weighted mAUC": wmulticlass_auc,
            "OVR AUCs": multiclass_aucs.tolist(),
            "N per AUC": nperauc.tolist()
        }
        print()

        results[c] = results_sample

    for key in results:
        print("**** COMPARISON FOR " + dataset + " with c=", str(key), " *****")
        results_sample = results[key]
        if results_sample["nap"]["unweighted mAUC"] < results_sample["napm"]["unweighted mAUC"]:
            print("Unweighted: NAPM")
        else:
            print("Unweighted: NAP")
        if results_sample["nap"]["weighted mAUC"] < results_sample["napm"]["weighted mAUC"]:
            print("Weighted: NAPM")
        else:
            print("Weighted: NAP")

    print()
    with open("../data/results/" + dataset + "_" + str(k) + ".json", 'w') as fp:
        json.dump(results, fp)
    print("Results saved to file")


