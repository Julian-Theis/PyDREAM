from pm4py.objects.log.log import EventLog

from pydream.util.Functions import time_delta_seconds

class LogWrapper:
    def __init__(self, log, resources=None):
        """
        Creates a new instance of a LogWrapper. This class provides extra functionalities to instances of <class \'pm4py.objects.log.log.EventLog\'> for Decay Replay
        :param log: event log of class pm4py.objects.log.log.EventLog to be wrapped
        :param resources: resources to consider, pass as a list of keys
        """
        if not isinstance(log, EventLog):
            raise ValueError('The input event log is not an instance of <class \'pm4py.objects.log.log.EventLog\'>')

        self.log = log
        self.ignored_traces = set()
        self.max_trace_duration = self.getMaxTraceDuration()

        self.iter_index = -1
        self.iter_remaining = len(self.log) - len(self.ignored_traces)


        """ SETUP RESOURCE COUNTER """
        if resources is None:
            self.resource_keys = None
        else:
            resource_values = set()
            self.resource_keys = dict()
            while self.iterator_hasNext():
                trace = self.iterator_next()
                for event in trace:
                    for resource_key in resources:
                        if resource_key in event.keys():
                            resource_values.add(resource_key + "_:_" + str(event[resource_key]))
            self.iterator_reset()
            resource_values = list(resource_values)
            for i in range(len(resource_values)):
                self.resource_keys[resource_values[i]] = i

    def getResourceKeys(self):
        return self.resource_keys

    def getMaxTraceDuration(self):
        """ trace duration is measured in seconds """
        max_trace_duration = 0

        for trace in self.log:
            if len(trace) < 2:
                self.ignored_traces.add(trace.attributes['concept:name'])

            seconds = time_delta_seconds(trace[0]['time:timestamp'], trace[-1]['time:timestamp'])
            if seconds > max_trace_duration:
                max_trace_duration = seconds

        if max_trace_duration <= 0:
            raise ValueError('The maximum trace duration of the event log is smaller or equal to 0ms')
        return max_trace_duration

    def isTraceIgnored(self, trace):
        return trace.attributes['concept:name'] in self.ignored_traces

    def iterator_reset(self):
        self.iter_index = -1
        self.iter_remaining = len(self.log) - len(self.ignored_traces)

    def iterator_hasNext(self):
        return self.iter_remaining > 0

    def iterator_next(self):
        if self.iterator_hasNext():
            self.iter_index += 1
            trace = self.log[self.iter_index]

            while(self.isTraceIgnored(trace)):
                self.iter_index += 1
                trace = self.log[self.iter_index]

            self.iter_remaining -= 1
            return trace
        else:
            raise ValueError('No more traces in log iterator.')