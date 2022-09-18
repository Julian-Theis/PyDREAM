import pandas as pd
import numpy as np

if __name__ == "__main__":
    log = pd.read_csv("../data/bpic13_closed_problems.csv")
    cases = len(pd.DataFrame(log["case:concept:name"].unique()))
    unique_events = len(pd.DataFrame(log["concept:name"].unique()))
    average_case_len = len(log)/cases

    print("Cases:", cases)
    print("Unqiue Events:", unique_events)
    print("Total evets", len(log))
    print("Case Len", average_case_len)





