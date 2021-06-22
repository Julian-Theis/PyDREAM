from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.objects.petri.utils import add_place, add_transition, add_arc_from_to, remove_arc

class SESEProcessor:
    def __init__(self,
                 input_pnml: str,
                 output_pnml: str):

        self.input_pnml = input_pnml
        self.output_pnml = output_pnml

        self.net, initial_marking, final_marking = pnml_importer.apply(input_pnml)

        cnt = 0
        execute = True
        while execute:
            execute, cnt = self.__process_sese(cnt)

        pnml_exporter.apply(self.net, initial_marking, self.output_pnml)
        print("Processed PNML exported to", self.output_pnml)

    def __process_sese(self, cnt):
        execute = False
        for transition in self.net.transitions:
            if transition.label is None:
                continue
            else:
                out_arcs = transition.out_arcs
                for arc in out_arcs:
                    if len(arc.target.in_arcs) > 1:
                        execute = True
                        break
                if execute:
                    break
        if execute:
            print("Transition", transition, "goes to place ", arc.target, " which has multple incoming arcs. Converting to SESE.")
            new_p = add_place(self.net, "sese_" + str(cnt))
            new_t = add_transition(self.net, "sese_" + str(cnt), label=None)

            add_arc_from_to(transition, new_p, self.net, weight=1)
            add_arc_from_to(new_p, new_t, self.net, weight=1)
            add_arc_from_to(new_t, arc.target, self.net, weight=1)
            remove_arc(self.net, arc)
            cnt += 1

        return execute, cnt