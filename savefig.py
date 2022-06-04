import json
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (13,8)


if __name__ == '__main__':
    with open('ruclip_final_scores.json', 'r') as f:
        all_scores = json.load(f)

    coco_scores = np.array([x['coco_translated_score'] for x in all_scores.values()])
    blip_scores = np.array([x['blip_score'] for x in all_scores.values()])
    ofa_scores = np.array([x['ofa_score'] for x in all_scores.values()])
    rudolph_scores = np.array([x['rudolph_score'] for x in all_scores.values()])

    bins=50
    alpha = 0.8
    plt.hist(rudolph_scores, bins, alpha=alpha, label=f'RuDOLPH ru (mean = {rudolph_scores.mean().round(2)})')
    plt.hist(blip_scores, bins, alpha=alpha, label=f'BLIP en-ru (mean = {blip_scores.mean().round(2)})')
    plt.hist(ofa_scores, bins, alpha=alpha, label=f'OFA en-ru (mean = {ofa_scores.mean().round(2)})')
    plt.hist(coco_scores, bins, alpha=alpha, label=f'Ground truth: COCO en-ru (mean = {coco_scores.mean().round(2)})')
    plt.legend(loc='upper right')
    plt.savefig('hist.png', bbox_inches='tight')
