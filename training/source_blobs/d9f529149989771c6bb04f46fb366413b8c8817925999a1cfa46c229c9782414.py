"""Explicit per-conversation SFT and per-token document replay objectives."""
from pathlib import Path
import numpy as np


def objective_weights(targets,replay_rows,sft_weight=.8,replay_weight=.2):
    targets=np.asarray(targets);flags=np.asarray(replay_rows,dtype=bool)
    counts=(targets!=-100).sum(axis=1);out=np.zeros(targets.shape,np.float32)
    for replay,weight in ((False,sft_weight),(True,replay_weight)):
        active=(flags==replay)&(counts>0);n=int(active.sum())
        if not n:raise ValueError('Mixed objective requires nonempty SFT and replay contributions')
        out[active]=(targets[active]!=-100)*(weight/n/counts[active,None])
    return out


class MixedBatcher:
    def __init__(self,corpus,data_dir,plan):
        self.corpus=corpus;self.plan=plan
        self.replay=np.load(Path(data_dir)/'replay.npy',mmap_mode='r',allow_pickle=False)
        if self.replay.shape[1]!=plan['replay_length']+1:raise ValueError('Replay sequence length changed')

    def batch(self,indices,step):
        plan=self.plan;sft_rows=plan['sft_rows_per_update'];replay_rows=plan['replay_rows_per_update']
        if sft_rows!=48 or replay_rows!=16:raise ValueError('Expected 48 SFT and 16 replay rows')
        a,b=self.corpus.batch(indices,size=sft_rows,max_length=plan['max_sequence_length'])
        length=max(a.shape[1],plan['replay_length']);x=np.zeros((64,length),np.int32);y=np.full((64,length),-100,np.int32)
        flags=np.arange(64)%4==3;positions=np.flatnonzero(~flags);x[positions,:a.shape[1]]=a;y[positions,:b.shape[1]]=b
        start=step*replay_rows;replay=np.asarray(self.replay[start:start+replay_rows],dtype=np.int32)
        if len(replay)!=replay_rows:raise ValueError('Replay pool exhausted; refusing silent recycling')
        x[flags,:plan['replay_length']]=replay[:,:-1];y[flags,:plan['replay_length']]=replay[:,1:]
        weights=objective_weights(y,flags,plan['sft_objective_weight'],plan['replay_objective_weight'])
        # One root writes each word. Float weights are transported losslessly as
        # int32 bits through the same checked host/RDMA transfer as token IDs.
        wire=np.stack([x,y,weights.view(np.int32)])
        info=dict(length=length,input_tokens=int(self.corpus.lengths[indices].sum())+replay_rows*plan['replay_length'],
            assistant_tokens=int((b!=-100).sum()),replay_tokens=replay_rows*plan['replay_length'])
        return wire,info
