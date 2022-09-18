import numpy as np
import math
from graphviz import Digraph
from pydream.reachability.ReachabilityGraph import ReachabilityGraph
from pydream.stat.ci_auc import calculate_auc_ci
from sklearn.metrics import roc_auc_score

class BinaryReachabilityInsights:
    """ These insights convert a multiclass prediction to binary (correct / incorrect) and provide an AUC
    per state with Confidence Interval based on a predefined alpha value

    """
    def __init__(self, rg:ReachabilityGraph, alpha=0.95, precision=3):
        self.rg = rg
        self.alpha = alpha
        self.precision = precision
        self.__state_classifications__ = dict()
        self.state_perf = dict()

        for state in self.rg.state_mem.getAllS():
            template = {"labels": [], "scores": []}
            self.__state_classifications__[state] = template

    def apply(self, bms, labels, scores):
        for i, bm in enumerate(bms):
            _, _, _, s = self.rg.state_mem.getStateFromBM(np.array(bm))
            self.__state_classifications__[s]["labels"].append(labels[i])
            self.__state_classifications__[s]["scores"].append(scores[i])

        for key in self.__state_classifications__.keys():
            obj = self.__state_classifications__[key]

            n_samples = len(obj["labels"])

            if n_samples < 4:
                auc = "N/A"
                ci = "N/A"
            else:
                auc, _, ci = calculate_auc_ci(y_true=np.array(obj["labels"]),
                                                    y_score=np.array(obj["scores"]),
                                                    y_pred=np.round(obj["scores"]),
                                                    alpha=self.alpha,
                                                    print_results=False)

                print(obj["labels"])
                print(obj["scores"])
                print(roc_auc_score(y_true=obj["labels"], y_score=obj["scores"]))
                print(auc)
                print()

                auc = np.around(auc, decimals=self.precision)
                if math.isnan(ci[0]):
                    ci = "N/A"
                else:
                    ci[0] = np.around(ci[0], decimals=self.precision)
                    ci[1] = np.around(ci[1], decimals=self.precision)

            perf = {"n_samples" : n_samples,
                    "auc": auc,
                    "ci": ci}
            self.state_perf[key] = perf

    def plot(self):
        gra = Digraph()
        gra.graph_attr["rankdir"] = "LR"
        gra.graph_attr["size"] = "16,9"

        _, _, _, init_state = self.rg.state_mem.getStateFromSM(str(self.rg.initial_marking))

        for state in self.rg.state_mem.getAllS():
            label = str(state) + "\n" + \
                    "AUC: " + str(self.state_perf[state]["auc"]) + "\n" + \
                    str(np.around(self.alpha * 100, 0)) + "% CI: " + str(self.state_perf[state]["ci"]) + "\n" + \
                    "N: " + str(self.state_perf[state]["n_samples"])

            if state == init_state:
                gra.node(str(state), label,
                         color="black",
                         style="filled",
                         fillcolor="green")
            else:
                gra.node(state, label)

        for stateA in self.rg.state_mem.getAllS():
            to_xs = self.rg.trans_mem.getOutgoing(stateA)
            for stateZ, t in to_xs:
                gra.edge(tail_name=stateA,
                         head_name=stateZ,
                         label=t.getTransition(),
                         color="black")

        gra.render("sample_data\\binary_insights_debug.pdf", view=True)




