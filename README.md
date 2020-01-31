# PyDREAM
PyDREAM is a Python implementation of the Decay Replay Mining (DREAM) approach and the corresponding predictive algorithms (NAP and NAPr) similar to the ones described in the paper [Decay Replay Mining to Predict Next Process Events](https://ieeexplore.ieee.org/document/8811455) and is based on [PM4Py](http://pm4py.org/). The original Java implementation used for benchmarking can be found [here](https://github.com/Julian-Theis/DREAM-NAP). There exists also a ProM plugin [here](https://prominentlab.github.io/ProM-DREAM/). 

# How-To
Please see [example.py](example.py) for an end-to-end example.

## Prerequisites
PyDREAM requires an event log and a corresponding PNML Petri net file, both imported through PM4Py.
```python
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.importer import pnml as pnml_importer

log = xes_import_factory.apply('YOUR_EVENTLOG.xes')
net, initial_marking, _ = pnml_importer.import_net("YOURPETRINET.pnml")
```

## Event Logs
The event log must be wrapped into a PyDREAM LogWrapper instance.
```python
from pydream.LogWrapper import LogWrapper
from pm4py.objects.log.importer.xes import factory as xes_import_factory

log = xes_import_factory.apply('YOUR_EVENTLOG.xes')
log_wrapper = LogWrapper(log)
```

If you plan on using resources, for example to train a NAPr model, provide the relevant resource identifiers.
```python
log_wrapper = LogWrapper(log, resources=["IDENTIFIER"])
```

## Decay Function Enhancement
The loaded Petri net can be enhanced as described in the paper as described subsequently. An *EnhancedPN* instance will automatically detect if a given *LogWrapper* objects encompasses resources.
```python
from pydream.EnhancedPN import EnhancedPN

enhanced_pn = EnhancedPN(net, initial_marking)
enhanced_pn.enhance(log_wrapper)
enhanced_pn.saveToFile("YOURENHANCEDPN.json")
```

## Decay Replay
Timed State Samples through Decay Replay can be obtained by replaying an event log wrapped into an *LogWrapper* instance on the enhanced Petri net. The resulting Timed State Samples are of JSON format and can be stored for later usage. The Timed State Samples will contain resource counters if a given *LogWrapper* objects encompasses resources.
```python
import json
from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.objects.log.importer.xes import factory as xes_import_factory

log = xes_import_factory.apply('YOUR_EVENTLOG.xes')
log_wrapper = LogWrapper(log)

net, initial_marking, _ = pnml_importer.import_net("YOURPETRINET.pnml")
enhanced_pn = EnhancedPN(net, initial_marking, decay_function_file="YOURENHANCEDPN.json")

timedstatesamples = enhanced_pn.decay_replay(log_wrapper=log_wrapper)
with open("timedstatesamples.json", 'w') as f:
        json.dump(timedstatesamples, f)
```

## Next trAnsition Prediction
A *NAP* or *NAPr* predictor can be trained in the following way.
```python
from pydream.predictive.nap.NAP import NAP

algo = NAP(tss_train_file="timedstatesamples.json", tss_test_file="timedstatesamples.json", options={"n_epochs" : 100})
algo.train(checkpoint_path="model-path", name="MODEL-NAME", save_results=True)
```

The corresponding model will be stored automatically based on the provided *checkpoint_path* and *name* parameters. The implemented options include:
* "seed" : int
* "n_epochs" : str
* "n_batch_size" : int
* "dropout_rate" : float
* "eval_size" : float
* "activation_function" : str

A trained model can be loaded, but the training file must always be provided, otherwise the feature scaler will be initialized incorrectly. 

# Requirements
PyDREAM is developed for Python 3.6 and is based on PM4Py v1.2.9. NAP and NAPr require tensorflow and keras. The full list of requirements can be found in [requirements.txt](requirements.txt).

## Citation
```
@article{theis2019decay,
  title={Decay Replay Mining to Predict Next Process Events},
  author={Theis, Julian and Darabi, Houshang},
  journal={IEEE Access},
  volume={7},
  pages={119787--119803},
  year={2019},
  publisher={IEEE}
}
```

## Remarks
This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the corresponding license for more details.

[![HitCount](http://hits.dwyl.io/Julian-Theis/PyDREAM.svg)](http://hits.dwyl.io/Julian-Theis/PyDREAM)


