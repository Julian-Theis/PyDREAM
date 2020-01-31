import json
import numpy as np

from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.exporter import pnml as pnml_exporter
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.algo.discovery.heuristics import factory as heuristics_miner

from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN
from pydream.predictive.nap.NAP import NAP

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

    algo = NAP(tss_train_file="timedstatesamples.json", tss_test_file="timedstatesamples.json", options={"n_epochs" : 100})
    algo.train(checkpoint_path="model-path", name="MODEL-NAME", save_results=True)

    algo = NAP(tss_train_file="timedstatesamples.json")
    algo.loadModel(path="model-path", name="MODEL-NAME")

    test_sample = np.array([
        [0.0, 0.0, 0.0, 10.0, 9.999999741502075, 10.0, 9.999999741502075, # Decay Vector
         0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, # Token Count
         0, 0, 0, 0, 0, 0, 1]]) # Marking
    print(algo.predict(test_sample))