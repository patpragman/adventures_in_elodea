import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from matplotlib import rcParams

def display_and_save_cm(y_actual=np.arange(5),
                        y_pred=np.arange(5),
                        name="Default Confusion Matrix Chart!",
                        labels=[f"Value = {i}" for i in range(0, 5)],
                        display_only=False):

    rcParams.update({'figure.autolayout': True})  # set rcParams to autoformat

    confusion_mtx = confusion_matrix(y_actual, y_pred)
    print("Raw confusion Matrix:\n", confusion_mtx)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx,
                                        display_labels=labels)
    cm_display.plot()
    ax = plt.gca()
    ax.set_xticklabels(labels, rotation=45)

    plt.title(name)
    if not display_only:
        plt.savefig(f"{name.replace(' ', '_')}.png")
    plt.show()

    rcParams.update({'figure.autolayout': False})  # undo the param set


if __name__ == "__main__":
    display_and_save_cm()
