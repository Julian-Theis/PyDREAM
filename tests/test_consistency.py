import os
import pytest
from pydream.EnhancedPN import EnhancedPN
from pm4py.objects.petri_net.importer import importer as pnml_importer

file_path = os.path.dirname(__file__)

@pytest.fixture
def enhancedpn1():
    """ Returns an EnhancedPN loaded from File """
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(file_path,"test_data", "petrinet1.pnml"))
    return EnhancedPN(net, initial_marking, decay_function_file=os.path.join(file_path,"test_data\\enhanced_petrinet1.json"))

@pytest.fixture
def enhancedpn2():
    """ Returns an EnhancedPN loaded from File """
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(file_path, "test_data\\petrinet1.pnml"))
    return EnhancedPN(net, initial_marking, decay_function_file=os.path.join(file_path, "test_data\\enhanced_petrinet1.json"))

def test_enhancedpn_load_consistency(enhancedpn1: EnhancedPN, enhancedpn2: EnhancedPN):
    errors = 0
    assert len(enhancedpn1.place_list) == len(enhancedpn2.place_list)
    for i in range(len(enhancedpn1.place_list)):
        if enhancedpn1.place_list[i] != enhancedpn2.place_list[i]:
            errors += 1
    assert errors == 0

