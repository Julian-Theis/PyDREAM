from pm4py.objects.log.log import EventLog

from dream.util.Functions import time_delta_seconds

class LogWrapper:
    def __init__(self, log):
        """
        Creates a new instance of a LogWrapper. This class provides extra functionalities to instances of <class \'pm4py.objects.log.log.EventLog\'> for Decay Replay
        :param log: event log of class pm4py.objects.log.log.EventLog to be wrapped
        """
        if not isinstance(log, EventLog):
            raise ValueError('The input event log is not an instance of <class \'pm4py.objects.log.log.EventLog\'>')

        self.log = log
        self.ignored_traces = set()
        self.max_trace_duration = self.getMaxTraceDuration()

        self.iter_index = -1
        self.iter_remaining = len(self.log) - len(self.ignored_traces)

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






