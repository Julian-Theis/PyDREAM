import json

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN
from pydream.predictive.nap.NAP import NAP


if __name__ == "__main__":

    log = xes_importer.apply('sample_data\\toy.xes')

    net, im, fm = inductive_miner.apply(log)
    pnml_exporter.apply(net, im, "sample_data\\discovered_pn.pnml", fm)

    net, initial_marking, final_marking = pnml_importer.apply("sample_data\\discovered_pn.pnml")

    log_wrapper = LogWrapper(log)
    enhanced_pn = EnhancedPN(net, initial_marking)
    enhanced_pn.enhance(log_wrapper)
    enhanced_pn.saveToFile("sample_data\\enhanced_discovered_pn.json")

    enhanced_pn = EnhancedPN(net, initial_marking, decay_function_file="sample_data\\enhanced_discovered_pn.json")
    tss_json, tss_objs = enhanced_pn.decay_replay(log_wrapper=log_wrapper)

    with open("sample_data\\timedstatesamples.json", 'w') as fp:
        json.dump(tss_json, fp)

    algo = NAP(tss_train_file="sample_data\\timedstatesamples.json",
               tss_test_file="sample_data\\timedstatesamples.json",
               options={"n_epochs": 10})
    algo.train(checkpoint_path="sample_data\\model",
               name="sample_model",
               save_results=True)

    algo = NAP()
    algo.loadModel(path="sample_data\\model",
                   name="sample_model")
    nap_out, string_out = algo.predict(tss_objs)

