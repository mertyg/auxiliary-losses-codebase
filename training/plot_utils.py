import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_hist(args, train, test):
    t = train["t"]
    for key, value in train.items():
        if key == "t":
            continue
        path = args.plot_path.format(key)
        plt.figure()
        plt.plot(t, value, label="train")
        plt.plot(t, test[key], label="test")
        plt.legend()
        plt.savefig(path)
        plt.close()
