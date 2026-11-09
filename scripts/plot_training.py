#!/usr/bin/env python3
"""Plot recorded logs; requires matplotlib 3.10.8 (not needed for inference)."""
import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    rows = list(csv.DictReader((ROOT/'results'/name).open()))
    return [dict(row, **{k:float(v) for k,v in row.items() if k!='phase'}) for row in rows]


def main():
    train = read('pretraining_steps.csv'); validation = read('pretraining_validation.csv')
    train = sorted({r['step']:r for r in train}.values(), key=lambda r:r['step'])
    validation = sorted({r['step']:r for r in validation}.values(), key=lambda r:r['step'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    output = ROOT/'docs/figures'; output.mkdir(parents=True, exist_ok=True)
    for name, lower, upper in [('pretraining-overview',0,42.1),('cooldown-detail',35,42.1)]:
        fig, (ax, lr) = plt.subplots(2,1,figsize=(10,6.8),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        ts=[r for r in train if r['tokens']/1e9>=lower]
        vs=[r for r in validation if r['tokens']/1e9>=lower]
        ax.plot([r['tokens']/1e9 for r in ts],[r['ema'] for r in ts],color='#8898a3',lw=.9,alpha=.8,label='Training loss EMA (changing mixture)')
        ax.plot([r['tokens']/1e9 for r in vs],[r['reported_val_loss'] for r in vs],color='#1557a0',lw=1.5,label='Held-out prose validation (logged BF16 path)')
        if lower==0: ax.set_ylim(2.2,5.6)
        ax.set_ylabel('Token cross-entropy loss'); ax.legend(loc='upper right',fontsize=9)
        lr.plot([r['tokens']/1e9 for r in ts],[r['lr'] for r in ts],color='#b25819',lw=1.5)
        lr.set_ylabel('Learning rate'); lr.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        lr.set_xlabel('Processed pretraining tokens (billions)')
        for panel in (ax,lr):
            panel.set_xlim(lower,upper); panel.grid(alpha=.16)
            panel.axvline(36.801216512,color='#5f6984',lw=.8,ls='--')
            panel.axvspan(40.001339392,42.001367040,color='#ead9ac',alpha=.4)
        fig.suptitle('Solstice-1 · 175M | '+('Pretraining history' if lower==0 else 'Late training and 2B-token cooldown'),fontsize=16,fontweight='bold')
        fig.supxlabel('Dashed line: Python/SQL introduced · Shaded region: final cooldown\nTraining and validation use different data; early training loss above 5.6 is outside the overview scale.' if lower==0 else 'Dashed line: Python/SQL introduced · Shaded region: final cooldown\nMetrics are transcribed from saved logs, with their original reported precision.',fontsize=9,color='#44505c')
        for ext in ('png','svg'): fig.savefig(output/f'{name}.{ext}',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
