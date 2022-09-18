import pandas as pd
import numpy as np

if __name__ == "__main__":
    """ Splits into 60% train, 20% val, 20% test """
    k = 5
    log = pd.read_csv("../../data/helpdesk.csv")
    cases = pd.DataFrame(log["case:concept:name"].unique())

    c = 0
    while c < k:
        train, validate, test = np.split(cases.sample(frac=1),
                     [int(.6 * len(cases)), int(.8 * len(cases))])
        train = train[0].unique()
        validate = validate[0].unique()
        test = test[0].unique()

        train_log = log[log["case:concept:name"].isin(train)]
        validate_log = log[log["case:concept:name"].isin(validate)]
        test_log = log[log["case:concept:name"].isin(test)]

        reference_event_set = set(log["concept:name"].unique())
        train_event_set = set(train_log["concept:name"].unique())
        test_event_set = set(test_log["concept:name"].unique())

        train_event_check = (train_event_set == reference_event_set)
        test_event_check = (test_event_set == reference_event_set)

        if train_event_check and test_event_check:
            c += 1
            train_log.to_csv("../../data/helpdesk_" + str(c) + "_train.csv", index=None)
            validate_log.to_csv("../../data/helpdesk_" + str(c) + "_validate.csv", index=None)
            test_log.to_csv("../../data/helpdesk_" + str(c) + "_test.csv", index=None)
            print("Saved fold", c)