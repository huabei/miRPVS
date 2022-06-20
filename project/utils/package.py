
import numpy as np
import matplotlib.pyplot as plt
import requests
from collections import defaultdict

def plot_fit_confidence_bond(x, y, r2, annot=True):
    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    # y_err = x.std() * np.sqrt(1 / len(x) +
    #                           (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot([-20, 0], [-20, 0], '-')
    ax.plot(x, y_est, '-')
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')
    if annot:
        num = 0
        for x_i, y_i in zip(x, y):
            ax.annotate(str(num), (x_i, y_i))
            # if y_i > -3:
            #     print(num)
            num += 1
    ax.set_xlabel('True Energy(Kcal/mol)')
    ax.set_ylabel('Predict Energy(Kcal/mol)')
    # ax.text(0.1, 0.5, 'r2:  ' + str(r2))
    ax.text(0.4, 0.9,
            'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=12)
    return fig


def send_to_wechat(message):
    key = 'SCT67936Tpp9RtEM5SnSNxczhMTKaMzW1'
    url = f'https://sctapi.ftqq.com/{key}.send'
    return requests.post(url=url, data=message)
