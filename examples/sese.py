import os
from pydream.preprocessing.SESE import SESEProcessor

if __name__ == "__main__":
    pnml_in = os.path.join("..", "sample_data", "discovered_pn.pnml")
    pnml_out = os.path.join("..", "sample_data", "sese_discovered_pn.pnml")

    _ = SESEProcessor(input_pnml=pnml_in, output_pnml=pnml_out)