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
        self.data = {"current_time": current_time}

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
                self.data["TimedStateSample"] = [decay_vector, token_count_vector, marking_vector]
            else:
                resource_vector = [0 for _ in range(len(resource_indices.keys()))]
                for key in resource_count.keys():
                    resource_vector[resource_indices[key]] = resource_count[key]
                self.data["TimedStateSample"] = [decay_vector, token_count_vector, marking_vector, resource_vector]
        else:
            """ Load from File """
            self.data = {"current_time": current_time,
                         "TimedStateSample": [decay_values, token_counts, marking]}

    def setResourceVector(self, resource_vector):
        if len(self.data["TimedStateSample"]) < 4:
            self.data["TimedStateSample"].append(resource_vector)
        else:
            self.data["TimedStateSample"][3] = resource_vector

    def setRecentEvent(self, event):
        self.data["recentEvent"] = event

    def setNextEvent(self, event):
        self.data["nextEvent"] = event

    def export(self):
        return self.data


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
            ts = TimedStateSample(sample["current_time"],
                                  sample["TimedStateSample"][0],
                                  sample["TimedStateSample"][1],
                                  sample["TimedStateSample"][2],
                                  None, loadExisting=True)
            """ Add resource count if exists """
            if len(sample["TimedStateSample"]) > 3:
                ts.setResourceVector(sample["TimedStateSample"][3])

            """ Add next event if present """
            if "nextEvent" in sample.keys():
                ts.setNextEvent(sample["nextEvent"])

            final.append(ts)
    return final
