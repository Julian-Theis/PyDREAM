import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datasets = ["bpic13_i"]
    k = 5
    exp = "ex1"

    nap_val_losses = []
    napm_val_losses = []

    for dataset in datasets:
        results = None
        for c in range(1, k+1):
            with open("../performend_experiments/" + exp + "/models/" + dataset + "_" + str(c) + "_nap_results.json") as json_file:
                results = json.load(json_file)
                nap_val_losses.append(results["val_loss"])

            with open("../performend_experiments/" + exp + "/models/" + dataset + "_" + str(c) + "_napm_results.json") as json_file:
                results = json.load(json_file)
                napm_val_losses.append(results["val_loss"])

    for i in range(0, len(nap_val_losses)):
        plt.title("Sample " + str(i))
        plt.plot(nap_val_losses[i], color="blue", label="NAP")
        plt.plot(napm_val_losses[i], color="red", label="NAPM")
        plt.legend()
        plt.xlim(0, 150)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
