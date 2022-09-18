import numpy as np
from graphviz import Digraph

from pm4py.objects.petri.petrinet import Marking


class ReachabilityGraph:

    def __init__(self, petrinet, initial_marking, place_list):
        self.pn = petrinet
        self.initial_marking = initial_marking

        self.state_mem = self.StateMemory(place_list)
        self.trans_mem = self.TransitionMemory()

        self.state_mem.addState(self.initial_marking)

    def state_to_state_move(self, marking_a, marking_z, transition):
        self.state_mem.addState(marking_a)
        self.state_mem.addState(marking_z)
        _, _, _, state_a = self.state_mem.getStateFromSM(str(marking_a))
        _, _, _, state_z = self.state_mem.getStateFromSM(str(marking_z))
        self.trans_mem.addTransitions(
            state_a=state_a,
            state_z=state_z,
            transition=transition
        )


    def getOutgoingTransitionNamesFromBM(self, bm):
        try:
            _, _, _, state = self.state_mem.getStateFromBM(bm)
            outgoing = self.trans_mem.getOutgoing(state)
            t = []
            for x in outgoing:
                t.append(x[1].getTransition())
            return t
        except:
            print("WARNING: BM not found, Returning empty list", bm)
            return list()

    def plot(self):
        gra = Digraph()
        gra.graph_attr["rankdir"] = "LR"
        gra.graph_attr["size"] = "16,9"

        _, _, _, init_state = self.state_mem.getStateFromSM(str(self.initial_marking))

        for state in self.state_mem.getAllS():
            if state == init_state:
                gra.node(str(state), str(state),
                         color="black",
                         style="filled",
                         fillcolor="green")
            else:
                gra.node(state, state)

        for stateA in self.state_mem.getAllS():
            to_xs = self.trans_mem.getOutgoing(stateA)
            for stateZ, t in to_xs:
                gra.edge(tail_name=stateA,
                         head_name=stateZ,
                         label=t.getTransition(),
                         color="black")

        gra.render("sample_data\\rg_debug.pdf", view=True)

    def debug(self):
        print("State Memory Debug:")
        print("m2bm:", len(self.state_mem.__m2bm__))
        print("bm2sm:", len(self.state_mem.__bm2sm__))
        print("sm2s:", len(self.state_mem.__sm2s__))
        print("s2m:", len(self.state_mem.__s2m__))
        print()
        print("Transition Memory Debug:")
        print("Transitions:", len(self.trans_mem.__transitions__))
        print("Incoming States:", len(self.trans_mem.__incoming_states__))
        print("Outgoing States:", len(self.trans_mem.__outgoing_states__))
        print()
        print("Plot Reachability Graph")
        self.plot()

    class TransitionMemory:
        """
        This is a memory consisting of transitions that can be queried by transition name, source state (S),
        and target state (S)
        """

        def __init__(self):
            self.__transitions__ = dict()
            self.__incoming_states__ = dict()
            self.__outgoing_states__ = dict()

        def addTransitions(self, state_a, state_z, transition):
            t = ReachabilityGraph.Transition(state_a, state_z, transition)
            if str(t) not in self.__transitions__.keys():
                self.__transitions__[str(t)] = t
                if t.getStateA() not in self.__outgoing_states__.keys():
                    self.__outgoing_states__[t.getStateA()] = list()
                if t.getStateZ() not in self.__incoming_states__.keys():
                    self.__incoming_states__[t.getStateZ()] = list()
                self.__outgoing_states__[t.getStateA()].append((t.getStateZ(), t))
                self.__incoming_states__[t.getStateZ()].append((t.getStateA(), t))

        def getOutgoing(self, state):
            try:
                return self.__outgoing_states__[state]
            except KeyError:
                return list()

        def getIncoming(self, state):
            try:
                return self.__incoming_states__[state]
            except KeyError:
                return list()

    class Transition:
        """ Transitions of the Reachability Graph """

        def __init__(self, state_a, state_z, transition):
            self.__state_a__ = state_a
            self.__state_z__ = state_z
            self.__transition__ = transition

        def getStateA(self):
            return self.__state_a__

        def getStateZ(self):
            return self.__state_z__

        def getTransition(self):
            return self.__transition__

        def __str__(self):
            return "[" + str(self.__state_a__) + "]->" + str(self.__transition__) + "->[" + str(self.__state_z__) + "]"

    class StateMemory:
        """
        This is a memory consisting of bidirectional circular hash maps to hold the reachability states in
        four different formats:
        m = marking object
        sm = marking represented as a string
        bm = marking represented as a binary numpy array
        s = state as a string + int (e.g. S100)
        """

        def __init__(self, place_list):
            self.place_list = place_list

            self.__m2bm__ = dict()
            self.__bm2sm__ = dict()
            self.__sm2s__ = dict()
            self.__s2m__ = dict()

        def memorySize(self):
            return len(self.__s2m__.keys())

        def getAllS(self):
            return self.__s2m__.keys()

        def getStateFromM(self, m):
            """ Get state representations from Marking object """
            try:
                bm = self.__m2bm__[m]
                sm = self.__bm2sm__[bm.tobytes()]
                s = self.__sm2s__[sm]
                return m, sm, bm, s
            except:
                raise ValueError("Could not load state representations from Marking:", m)

        def getStateFromSM(self, sm):
            """ Get state representations from string represented marking """
            try:
                s = self.__sm2s__[sm]
                m = self.__s2m__[s]
                bm = self.__m2bm__[m]
                return m, sm, bm, s
            except:
                raise ValueError("Could not load state representations from string marking:", sm)

        def getStateFromBM(self, bm):
            """ Get state representations from binary marking (numpy array) """
            try:
                sm = self.__bm2sm__[bm.tobytes()]
                s = self.__sm2s__[sm]
                m = self.__s2m__[s]
                return m, sm, bm, s
            except:
                raise ValueError("Could not load state representations from binary marking:", bm)

        def getStateFromS(self, s):
            """ Get state representations from string (+integer) """
            try:
                m = self.__s2m__[s]
                bm = self.__m2bm__[m]
                sm = self.__bm2sm__[bm.tobytes()]
                return m, sm, bm, s
            except:
                raise ValueError("Could not load state representations from string (+integer):", s)

        def addState(self, m):
            """ Adding a state to the memory using Marking """
            if isinstance(m, Marking):
                if str(m) not in self.__sm2s__.keys():
                    sm = str(m)
                    bm = self.__createBM__(m)
                    s = "S" + str(len(self.__s2m__))
                    self.__insert__(m, sm, bm, s)
            else:
                raise ValueError("StateMemoryError - addState: added state is not a Marking object.")

        def __insert__(self, m, sm, bm, s):
            self.__m2bm__[m] = bm
            self.__bm2sm__[bm.tobytes()] = sm
            self.__sm2s__[sm] = s
            self.__s2m__[s] = m

        def __createBM__(self, m):
            bm = np.zeros(len(self.place_list))
            for p in m:
                bm[self.place_list.index(str(p))] = 1
            return bm
