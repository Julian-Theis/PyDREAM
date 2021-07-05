import os
from pydream.preprocessing.SESE import SESEProcessor

if __name__ == "__main__":
    fold = "9"

    pnml_in = os.path.join("..", "sample_data", "origpi_fold_" + fold + ".pnml")
    pnml_out = os.path.join("..", "sample_data", "sese_origpi_fold_" + fold + ".pnml")

    _ = SESEProcessor(input_pnml=pnml_in, output_pnml=pnml_out)