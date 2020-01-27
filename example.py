import json
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.exporter import pnml as pnml_exporter
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.algo.discovery.heuristics import factory as heuristics_miner

from dream.LogWrapper import LogWrapper
from dream.EnhancedPN import EnhancedPN

if __name__== "__main__":
    log = xes_import_factory.apply('YOUR_EVENTLOG.xes')

    net, im, fm = heuristics_miner.apply(log, parameters={"dependency_thresh": 0.99})
    pnml_exporter.export_net(net, im, "discovered_pn.pnml")

    net, initial_marking, final_marking = pnml_importer.import_net("discovered_pn.pnml")

    log_wrapper = LogWrapper(log)
    enhanced_pn = EnhancedPN(net, initial_marking)
    enhanced_pn.enhance(log_wrapper)
    enhanced_pn.saveToFile("enhanced_discovered_pn.json")

    enhanced_pn = EnhancedPN(net, initial_marking, decay_function_file="enhanced_discovered_pn.json")
    timedstatesamples = enhanced_pn.decay_replay(log_wrapper=log_wrapper)

    with open("timedstatesamples.json", 'w') as fp:
        json.dump(timedstatesamples, fp)