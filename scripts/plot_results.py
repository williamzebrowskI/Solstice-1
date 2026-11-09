#!/usr/bin/env python3
"""Render the model-card figures from saved scores; matplotlib 3.10.8."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
COLORS = {'Solstice-1': '#176b65', 'OpenSML-150M': '#67768a',
          'SmolLM2-135M Instruct': '#c58a37', 'Checkpoint 167': '#9476a7'}


def read(path):
    return json.loads((ROOT / path).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.titleweight': 'bold'})
    choice = read('results/public/choices/results.json')['results']
    public = read('results/baselines/published_models.json')['models']
    old = public['OpenSML/OpenSML-150M']
    smol = public['HuggingFaceTB/SmolLM2-135M-Instruct']
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.subplots_adjust(left=.2, right=.97, top=.84, bottom=.16, wspace=.85, hspace=.55)
    fig.suptitle('Solstice-1 | Knowledge and commonsense', x=.08, ha='left', y=.97,
                 fontsize=19, fontweight='bold')
    fig.text(.08, .91, 'Zero-shot normalized accuracy · higher is better', fontsize=12)
    for ax, (key, label) in zip(axes.flat, [('arc_easy', 'ARC Easy'),
            ('arc_challenge', 'ARC Challenge'), ('piqa', 'PIQA'), ('hellaswag', 'HellaSwag')]):
        names = ['Solstice-1', 'OpenSML-150M']
        values = [choice[key]['acc_norm'], old['choices'][key]['acc_norm']]
        if key in smol['choices']:
            names.append('SmolLM2-135M Instruct')
            values.append(smol['choices'][key]['acc_norm'])
        ax.barh(range(len(names)), values, height=.55, color=[COLORS[n] for n in names])
        ax.set_yticks(range(len(names)), [n.replace(' Instruct', '\nInstruct') for n in names], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, .8); ax.set_xticks([0, .2, .4, .6, .8])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(label, loc='left', pad=10); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        for y, value in enumerate(values):
            ax.text(value+.015, y, f'{value:.2%}' if names[y] != 'SmolLM2-135M Instruct' else f'{value:.1%}',
                    va='center', fontsize=10)
    fig.text(.08, .06, 'Baseline values are publisher reports; evaluators and prompts differ.\n'
             'SmolLM2 publishes ARC only as an average (37.3%); its separate ARC scores are not inferred.',
             fontsize=10, color='#4b5563')
    save(fig, 'benchmark-comparison')

    own = read('results/public/ifeval/results.json')['metrics']
    prior = read('results/baselines/167/ifeval/results.json')['metrics']
    keys = [('strict', 'prompt'), ('strict', 'instruction'), ('loose', 'prompt'), ('loose', 'instruction')]
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    fig.subplots_adjust(left=.08, right=.98, top=.8, bottom=.24)
    fig.suptitle('Solstice-1 | Instruction following', x=.08, ha='left', y=.97,
                 fontsize=19, fontweight='bold')
    fig.text(.08, .9, 'IFEval accuracy · higher is better', fontsize=12)
    x = np.arange(4)
    for i, (name, data) in enumerate([('OpenSML-150M', old['ifeval']),
                                     ('Checkpoint 167', prior), ('Solstice-1', own)]):
        values = [data[k][u+'_accuracy'] for k, u in keys]
        bars = ax.bar(x+(i-1)*.25, values, width=.22, label=name, color=COLORS[name])
        ax.bar_label(bars, labels=[f'{v:.2%}' for v in values], padding=4, fontsize=9)
    ax.set_xticks(x, ['Strict\nprompts', 'Strict\ninstructions', 'Loose\nprompts', 'Loose\ninstructions'])
    ax.set_ylim(0, .4); ax.set_yticks([0, .1, .2, .3, .4]); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.18), ncols=3, frameon=False)
    fig.text(.08, .035, '150M used different formatting and a longer output cap. SmolLM2 reports an aggregate (29.9%),\n'
             'so its strict/loose sub-scores are not plotted. No normalized-accuracy metric applies to IFEval.',
             fontsize=10, color='#4b5563')
    save(fig, 'ifeval-comparison')

    chat = read('results/public/chat/results.json')
    categories = [('reading', 'Reading supplied text'), ('conversation', 'Conversation context'),
                  ('knowledge', 'Factual knowledge'), ('instructions', 'Instructions'),
                  ('answerability', 'Answerability'), ('reasoning', 'Short reasoning')]
    values = [chat['categories'][key]['answer_match']/chat['categories'][key]['count'] for key, _ in categories]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=.25, right=.96, top=.79, bottom=.2)
    fig.suptitle('Solstice-1 | Short-chat diagnostics', x=.08, ha='left', y=.97,
                 fontsize=19, fontweight='bold')
    fig.text(.08, .9, 'Conservative automatic answer matching · authored panel', fontsize=12)
    ax.barh(range(6), values, color=COLORS['Solstice-1'], height=.6)
    ax.set_yticks(range(6), [label for _, label in categories]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis='y', length=0); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for y, value in enumerate(values): ax.text(value+.015, y, f'{value:.1%}', va='center')
    fig.text(.08, .05, '30 checked responses per category. This is not a broad human assessment.\n'
             'Natural stopping is 99.48% across all 192 responses; stopping correctly does not imply a correct answer.',
             fontsize=10, color='#4b5563')
    save(fig, 'chat-diagnostics')


if __name__ == '__main__':
    main()
