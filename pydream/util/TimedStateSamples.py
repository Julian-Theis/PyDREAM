import json
from datetime import datetime
from pm4py.objects.petri.petrinet import Marking


class TimedStateSample:

    def __init__(self,
                 current_time: datetime,
                 decay_values: dict,
                 token_counts: dict,
                 marking: Marking,
                 place_list: list,
                 resource_count: dict = None,
                 resource_indices: dict = None,
                 loadExisting: bool = False):
        self.__data__ = {"ct": current_time}
        #ct = current time

        if not loadExisting:
            decay_vector = []
            token_count_vector = []
            marking_vector = []

            for place in place_list:
                decay_vector.append(decay_values[str(place)])
                token_count_vector.append(token_counts[str(place)])

                if str(place) in [str(key) for key in marking.keys()]:
                    for key, val in marking.items():
                        if str(key) == str(place):
                            marking_vector.append(val)
                            break
                else:
                    marking_vector.append(0)

            if resource_count is None:
                self.__data__["tss"] = [decay_vector, token_count_vector, marking_vector]
            else:
                resource_vector = [0 for _ in range(len(resource_indices.keys()))]
                for key in resource_count.keys():
                    resource_vector[resource_indices[key]] = resource_count[key]
                self.__data__["tss"] = [decay_vector, token_count_vector, marking_vector, resource_vector]
        else:
            """ Load from File """
            self.__data__ = {"ct": current_time,
                         "tss": [decay_values, token_counts, marking]}

    def setResourceVector(self, resource_vector):
        if len(self.data["tss"]) < 4:
            self.__data__["tss"].append(resource_vector)
        else:
            self.__data__["tss"][3] = resource_vector

    def setLabel(self, label):
        self.__data__["label"] = label

    def setTraceIdentifier(self, trace_id):
        self.__data__["trace_id"] = trace_id

    def setEventId(self, event_id):
        self.__data__["event_id"] = event_id

    def export(self):
        return self.__data__


def loadTimedStateSamples(filename: str):
    """
    Load decay functions for a given petri net from file.

    :param filename: filename of Timed State Samples
    :return: list containing TimedStateSample objects
    """
    final = list()
    with open(filename) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            ts = TimedStateSample(sample["ct"],
                                  sample["tss"][0],
                                  sample["tss"][1],
                                  sample["tss"][2],
                                  None, loadExisting=True)
            """ Add resource count if exists """
            if len(sample["tss"]) > 3:
                ts.setResourceVector(sample["tss"][3])

            """ Add label if present """
            if "label" in sample.keys():
                ts.setNextEvent(sample["label"])

            final.append(ts)
    return final
