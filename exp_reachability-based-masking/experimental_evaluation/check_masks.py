import json
import numpy as np
import itertools

def checkMask(file):
    m_lengths = []
    with open(file) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            if sample["nextEvent"] is not None:
                m_lengths.append(len(sample["mask"]))

    return np.array(m_lengths)

if __name__ == "__main__":
    dataset = "bpic12_o"
    k = 5

    for c in range(1, k+1):
        print("*** FOLD ", str(c), " ***")

        tss_train_file = "../data/" + dataset + "_" + str(c) + "_tss_train.json"
        tss_val_file = "../data/" + dataset + "_" + str(c) + "_tss_val.json"
        tss_test_file = "../data/" + dataset + "_" + str(c) + "_tss_test.json"

        m_test = checkMask(tss_test_file)
        print("Min number:", np.min(m_test))
        print("max number:", np.max(m_test))
        print("mean number:", np.mean(m_test))
        print("median number:", np.median(m_test))
        print()


