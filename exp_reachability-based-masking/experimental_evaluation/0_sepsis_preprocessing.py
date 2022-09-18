import pandas as pd
import numpy as np
import json

from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from datetime import datetime, timedelta
from pm4py.objects.log.importer.xes import importer as xes_import_factory

if __name__ == "__main__":
    lines = list()
    lines.append('concept:name,time:timestamp,case:concept:name')
    log = xes_import_factory.apply('C:\DATASETS\sepsis\sepsis.xes')

    CRP_values = []
    Leuco_values = []
    Lactic_values = []
    ts_end = 19

    skipped = 0

    for c in range(len(log)):
        case = "c" + str(c) #log[c].attributes['concept:name']

        for e in range(len(log[c])):
            ev = log[c][e]
            ts = ev["time:timestamp"]

            if ev["concept:name"] == 'ER Registration':
                lines.append(ev["concept:name"] + "," + str(ts)[0:ts_end] + "," + case)
                """
                cnt = 1
                for key in ev.keys():
                    if ev[key] == 1:
                        ts = ts + timedelta(seconds=1)
                        lines.append(key + "," + str(ts)[0:ts_end] + "," + case)
                lines.append('ER Registration complete' + "," + str(ts)[0:ts_end] + "," + case)
                """
                continue
            elif ev["concept:name"] == 'CRP':
                if 'CRP' in ev.keys():
                    if ev['CRP'] >= 113.74575728466219:
                        new_event = "CRP_high"
                    else:
                        new_event = "CRP_low"
                else:
                    new_event = "CRP_none"

            elif ev["concept:name"] == 'Leucocytes':
                if 'Leucocytes' in ev.keys():
                    if ev['Leucocytes'] >= 12.949360309431718:
                        new_event = "Leucocytes_high"
                    else:
                        new_event = "Leucocytes_low"
                else:
                    new_event = "Leucocytes_none"

            elif ev["concept:name"] == 'LacticAcid':
                if 'LacticAcid' in ev.keys():
                    Lactic_values.append(ev['LacticAcid'])
                    if ev['LacticAcid'] >= 1.9491059147180192:
                        new_event = "LacticAcid_high"
                    else:
                        new_event = "LacticAcid_low"
                else:
                    new_event = "LacticAcid_none"

            elif ev["concept:name"] == 'Return ER':

                continue

            else:
                new_event = ev["concept:name"]

            lines.append(new_event + "," + str(ts)[0:ts_end] + "," + case)


    textfile = open('C:\DATASETS\sepsis\sepsis_small.csv', "w")
    for element in lines:
        textfile.write(element + "\n")
    textfile.close()
    #print(Lactic_values)
    #print("mean", np.array(Lactic_values).mean())

    """
    log = pd.read_csv("../data/" + dataset + ".csv")
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
    log = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    """



