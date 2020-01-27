import numpy as np
import json

from pm4py.algo.conformance.tokenreplay.versions.token_replay import *
from pm4py.objects.petri import semantics

from dream.util.DecayFunctions import LinearDecay, REGISTER
from dream.util.TimedStateSamples import TimedStateSample
from dream.util.Functions import time_delta_seconds

class EnhancedPN:
    def __init__(self, net, initial_marking, decay_function_file=None):
        """
        Creates a new instance of an enhanced petri net
        :param net: petri net loaded from pm4py
        :param initial_marking: initial marking from pm4py
        :param decay_function_file: default=None, path to existing decay function file for the petri net
        """
        self.config = {}
        self.decay_functions = {}
        self.net = net
        self.initial_marking = initial_marking

        if decay_function_file is not None:
            self.loadFromFile(decay_function_file)

        self.activity_key = "concept:name"
        self.MAX_REC_DEPTH = 50

        self.place_list = list()
        for place in self.net.places:
            self.place_list.append(str(place))
        self.place_list = sorted(self.place_list)

        self.trans_map = {}
        for t in self.net.transitions:
            self.trans_map[t.label] = t

    def enhance(self, log_wrapper):
        """
        Enhance a given petri net based on an event log.
        :param log_wrapper: Event log under consideration as LogWrapper
        :return:
        """

        """ Standard Enhancement """
        beta = float(10)

        reactivation_deltas = {}
        for place in self.net.places:
            reactivation_deltas[str(place)] = list()

        log_wrapper.iterator_reset()
        last_activation = {}
        while(log_wrapper.iterator_hasNext()):
            trace = log_wrapper.iterator_next()

            for place in self.net.places:
                if place in self.initial_marking:
                    last_activation[str(place)] = trace[0]['time:timestamp']
                else:
                    last_activation[str(place)] = -1

            """ Replay and estimate parameters """
            places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
            marking = copy(self.initial_marking)

            for event in trace:
                if event[self.activity_key] in self.trans_map.keys():
                    activated_places = []
                    toi = self.trans_map[event[self.activity_key]]

                    """ If Transition of interest is not enabled yet, then go through hidden"""
                    if not semantics.is_enabled(toi, self.net, marking):

                        _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking),places_shortest_path_by_hidden, [], 0, set(),[copy(marking)])
                        for act_tran in act_trans:
                            for arc in act_tran.out_arcs:
                                activated_places.append(arc.target)
                            marking = semantics.execute(act_tran, self.net, marking)

                    """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        for arc in toi.in_arcs:
                            if arc.source not in marking:
                                marking[arc.source] += 1

                    """ Fire transition of interest """
                    for arc in toi.out_arcs:
                        activated_places.append(arc.target)
                    marking = semantics.execute(toi, self.net, marking)

                    """ Marking is gone - transition could not be fired ..."""
                    if marking is None:
                        raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")

                    """ Update Time Recordings """
                    for activated_place in activated_places:
                        if last_activation[str(activated_place)] != -1:
                            time_delta = time_delta_seconds(last_activation[str(activated_place)], event['time:timestamp'])
                            if time_delta > 0:
                                reactivation_deltas[str(place)].append(time_delta)
                        last_activation[str(activated_place)] = event['time:timestamp']

        """ Calculate decay function parameter """
        for place in self.net.places:
            if len(reactivation_deltas[str(place)]) > 1:
                self.decay_functions[str(place)] = LinearDecay(alpha=1/np.mean(reactivation_deltas[str(place)]), beta=beta)
            else:
                self.decay_functions[str(place)] = LinearDecay(alpha=1/log_wrapper.max_trace_duration, beta=beta)

    def decay_replay(self, log_wrapper):
        """
        Decay Replay on given event log.
        :param log_wrapper: Input event log as LogWrapper to be replayed.
        :return: list of timed state samples
        """
        tss = list()

        decay_values = {}
        token_counts = {}
        marks = {}

        last_activation = {}

        log_wrapper.iterator_reset()
        while log_wrapper.iterator_hasNext():
            trace = log_wrapper.iterator_next()

            """ Reset all counts for the next trace """
            for place in self.net.places:
                if place in self.initial_marking:
                    last_activation[str(place)] = trace[0]['time:timestamp']
                else:
                    last_activation[str(place)] = -1

            for place in self.net.places:
                decay_values[str(place)] = 0.0
                token_counts[str(place)] = 0.0
                marks[str(place)] = 0.0
            """ ----------------------------------> """

            places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
            marking = copy(self.initial_marking)

            """ Initialize counts based on initial marking """
            for place in marking:
                decay_values[str(place)] = self.decay_functions[str(place)].decay(t=0)
                token_counts[str(place)] += 1
                marks[str(place)] = 1
            """ ----------------------------------> """

            """ Replay """
            time_past = None
            time_recent = None
            init_time = None
            for event_id in range(len(trace)):
                event = trace[event_id]
                if event_id == 0:
                    init_time = event['time:timestamp']

                time_past = time_recent
                time_recent = event['time:timestamp']

                if event[self.activity_key] in self.trans_map.keys():
                    activated_places = list()

                    toi = self.trans_map[event[self.activity_key]]
                    """ If Transition of interest is not enabled yet, then go through hidden"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking), places_shortest_path_by_hidden,[], 0, set(), [copy(marking)])
                        for act_tran in act_trans:
                            for arc in act_tran.out_arcs:
                                activated_places.append(arc.target)
                            marking = semantics.execute(act_tran, self.net, marking)
                    """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        for arc in toi.in_arcs:
                            if arc.source not in marking:
                                marking[arc.source] += 1
                    """ Fire transition of interest """
                    for arc in toi.out_arcs:
                        activated_places.append(arc.target)
                    marking = semantics.execute(toi, self.net, marking)
                    """ Marking is gone - transition could not be fired ..."""
                    if marking is None:
                        raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")
                    """ ----->"""

                    """ Update Time Recordings """
                    for activated_place in activated_places:
                        last_activation[str(activated_place)] = event['time:timestamp']

                    """ Update Vectors and create TimedStateSamples """
                    if not time_past is None:
                        decay_values, token_counts = self.updateVectors(decay_values=decay_values, last_activation=last_activation, token_counts=token_counts, activated_places=activated_places, current_time=time_recent)

                        next_event_id = self.findNextEventId(event_id, trace)
                        if next_event_id is not None:
                            next_event = trace[next_event_id][self.activity_key]
                        else:
                            next_event = None

                        timedstatesample = TimedStateSample(time_delta_seconds(init_time, time_recent), copy(decay_values), copy(token_counts), copy(marking), copy(self.place_list))
                        timedstatesample.setNextEvent(next_event)
                        tss.append(timedstatesample.export())
        return tss

    def updateVectors(self, decay_values, last_activation, token_counts, activated_places, current_time):
        """ Update Decay Values """
        for place in self.net.places:
            if last_activation[str(place)] == -1:
                decay_values[str(place)] = 0.0
            else:
                delta = time_delta_seconds(last_activation[str(place)], current_time)
                decay_values[str(place)] = self.decay_functions[str(place)].decay(delta)

        """ Update Token Counts """
        for place in activated_places:
            token_counts[str(place)] += 1

        return decay_values, token_counts

    def findNextEventId(self, current_id, trace):
        next_event_id = current_id + 1

        found = False
        while(next_event_id < len(trace) and not found):
            event = trace[next_event_id]
            if event[self.activity_key] in self.trans_map.keys():
                found = True
            else:
                next_event_id += 1

        if found == False:
            return None
        else:
            return next_event_id

    
    def saveToFile(self, file):
        """
        Save the decay functions of the EnhancedPN to file.
        :param file: Output file
        :return:
        """

        dumping = copy(self.decay_functions)
        for key in dumping.keys():
            dumping[key] = dumping[key].toJSON()
        with open(file, 'w') as fp:
            json.dump(dumping, fp)

    def loadFromFile(self, file):
        """
        Load decay functions for a given petri net from file.
        :param file: Decay function file
        :return:
        """
        with open(file) as json_file:
            decay_data = json.load(json_file)

        for place in self.net.places:
            self.decay_functions[str(place)] = None

        if not set(decay_data.keys()) == set(self.decay_functions.keys()):
            self.decay_functions = {}
            raise ValueError("Set of decay functions is not equal to set of places of the petri net. Was the decay function file build on the same petri net? Loading from file cancelled.")

        for place in decay_data.keys():
                DecayFunctionClass = REGISTER[decay_data[place]['DecayFunction']]
                df = DecayFunctionClass()
                df.loadFromDict(decay_data[place])
                self.decay_functions[str(place)] = df
