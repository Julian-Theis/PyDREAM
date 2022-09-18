import pandas as pd
import json

from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

if __name__ == "__main__":
    dataset = "sepsis_small"

    log = pd.read_csv("../data/" + dataset + ".csv")
    print(log)
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
    log = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    net, initial_marking, final_marking = inductive_miner.apply(log)


    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)

    aligned_traces = token_replay.apply(log, net, initial_marking, final_marking)
    fit_traces = [x for x in aligned_traces if x['trace_is_fit']]
    perc_fitness = 0.00
    if len(aligned_traces) > 0:
        perc_fitness = len(fit_traces) / len(aligned_traces)
    print("perc_fitness=", perc_fitness)
    pnml_exporter.apply(net, initial_marking, "../data/models/" + dataset + ".pnml", final_marking)

    results = dict()
    results["perc_trace_fitness"] = perc_fitness
    with open("../data/models/" + dataset + ".json", 'w') as outfile:
        json.dump(results, outfile)



