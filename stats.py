import pickle
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace
from scipy.interpolate import make_interp_spline, BSpline

STATS_PATH = "./stats"
FILE_NAME = "/12-17_21-59-29_E7A1ED.pickle"

def visualize_val_stats(stats):
    val_mean_aps = np.array(stats['val_mean_aps'])
    val_avg_pre = np.array(stats['val_avg_pre'])
    val_avg_rec = np.array(stats['val_avg_rec'])
    val_avg_hmean = np.array(stats['val_avg_hmean'])
    ticks = len(val_avg_hmean)

    plt.subplot(2, 1, 1)

    plt.plot(val_mean_aps)
    # plt.xticks(np.linspace(1, ticks, ticks, dtype="int8"))
    plt.title("TOP 50 MEAN AVERAGE PRECISION")
    plt.xlabel("EPOCHS")
    plt.ylabel("MAP")

    plt.subplot(2, 1, 2)
    plt.plot(val_avg_pre)
    plt.plot(val_avg_rec)
    plt.plot(val_avg_hmean)
    plt.title("STATS OVER TIME")
    plt.xlabel("EPOCHS")
    plt.ylabel("AVERAGE VALUE")
    plt.ylim([0.0, 0.7])
    plt.legend(
        ["Average Precision", "Average Recall", "Average Harmonic Mean"],
        loc="upper right")

    plt.subplots_adjust(hspace=1.0)

    plt.show()

def visualize_test_stats(stats):
    test_recall = np.array(stats['test_rec_curve'])
    test_precision = np.array(stats['test_pre_curve'])
    test_ap = np.array(stats['test_avg_pre'])
    test_map = np.array(stats['test_mean_ap'])

    plt.step(test_recall, test_precision)
    plt.fill_between(test_recall, test_precision, alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.xlim([0.0, 1.0])
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.title("Retrieval Performance: AP={:.4f}; TOP50 MAP={:.4f}."
                .format(test_ap, test_map))
    plt.show()

if __name__ == "__main__":
    # stats = None

    # with open(STATS_PATH + FILE_NAME, "rb") as file:
    #     stats = pickle.load(file)

    # visualize_val_stats(stats)
    # visualize_test_stats(stats)

    name = [
        "Low Res Baseline",
        "High Res Baseline",
        "L2 Constraint",
        "Image Alignment"
    ]

    stat_paths = [
        "/12-17_20-18-32_99AD3C.pickle",
        "/12-17_20-25-11_25F03D.pickle",
        "/12-17_21-59-29_E7A1ED.pickle",
        "/12-19_14-18-00_CE7872.pickle"
    ]

    colors = ['r', 'g', 'b', 'y']

    for idx, path in enumerate(stat_paths):
        with open(STATS_PATH + path, "rb") as file:
            stats = pickle.load(file)
        print(stats['test_mean_ap'])

        # val_mean_aps = np.array(stats['val_mean_aps'])
        # ticks = np.linspace(1, len(val_mean_aps),
        #                     len(val_mean_aps), dtype="uint8")

        # X = np.linspace(ticks.min(), ticks.max(), 100)
        # spline = make_interp_spline(ticks, val_mean_aps, k=3)
        # smooth = spline(X)
        # # plt.plot(val_mean_aps)
        # plt.plot(X, smooth, color=colors[idx])
        # # plt.xticks(np.linspace(1, ticks, ticks // 2, dtype="int8"))
        # plt.title("TOP 50 MEAN AVERAGE PRECISION")
        # plt.xlabel("EPOCHS")
        # plt.ylabel("MAP")

    # plt.legend(name, loc="lower right")
    # plt.show()
