"""Answer supervision plus optional forward KL to frozen parent on replay."""
import math
from collections import defaultdict
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .data import serialize,shifted,training_order

def encode(tokenizer,row,final_only):
    ids,mask=serialize(tokenizer,row['messages'])
    if final_only:
        prefix,_=serialize(tokenizer,row['messages'][:-1],generation=True)
        if not np.array_equal(ids[:len(prefix)],prefix):raise ValueError('Final answer prefix differs')
        mask=np.arange(len(ids))>=len(prefix)
    return shifted(ids,mask)

def batch(tokenizer,rows,final_only,max_length=2048):
    sequences=[encode(tokenizer,row,final_only) for row in rows]
    length=max(len(x) for x,_ in sequences)
    if length>max_length:raise ValueError('Never truncate a training conversation')
    length=min(max_length,max(16,2**math.ceil(math.log2(length))))
    x=np.zeros((len(rows),length),np.int32);y=np.full_like(x,-100)
    for i,(a,b) in enumerate(sequences):x[i,:len(a)]=a;y[i,:len(b)]=b
    return mx.array(x),mx.array(y),sum(len(a) for a,_ in sequences),sum(int((b!=-100).sum()) for _,b in sequences)

def log_probs(logits):
    values=logits.astype(mx.float32)
    return values-mx.logsumexp(values,axis=-1,keepdims=True)

def masked_kl(logits,teacher_logp,labels):
    teacher_logp=mx.stop_gradient(teacher_logp)
    per_token=(mx.exp(teacher_logp)*(teacher_logp-log_probs(logits))).sum(axis=-1)
    mask=labels!=-100
    return ((per_token*mask).sum(axis=-1)/mx.maximum(mask.sum(axis=-1),1)).mean()

def backward(model,kl_weight,*,compile_one=True):
    def objective(x,y,teacher_logp):
        result=model(x,targets=y);mask=y!=-100
        nll=((result['token_loss'].astype(mx.float32)*mask).sum(axis=-1)/mx.maximum(mask.sum(axis=-1),1)).mean()
        divergence=masked_kl(result['logits'],teacher_logp,y) if kl_weight else mx.array(0.,mx.float32)
        return nll+kl_weight*divergence,(nll,divergence)
    one=nn.value_and_grad(model,objective)
    return mx.compile(one,inputs=[model.state],outputs=[model.state]) if compile_one else one

def choose(rows,count,mixture,seed):
    rng=np.random.default_rng(seed);pools=defaultdict(list)
    for i,row in enumerate(rows):pools[row['component']].append(i)
    if mixture is None:
        if count>len(rows):raise ValueError('Insufficient distinct conversations')
        return list(map(int,rng.permutation(len(rows))[:count]))
    if abs(sum(mixture.values())-1)>1e-8:raise ValueError('Invalid mixture')
    counts={k:int(count*v) for k,v in mixture.items()}
    for k in sorted(mixture,key=lambda k:count*mixture[k]-counts[k],reverse=True)[:count-sum(counts.values())]:counts[k]+=1
    selected=[]
    for component,amount in counts.items():
        if amount>len(pools[component]):raise ValueError('Insufficient distinct '+component+' examples')
        indices=rng.permutation(pools[component])[:amount];selected.extend(map(int,indices))
    rng.shuffle(selected)
    if len(selected)!=count or len(set(selected))!=count:raise ValueError('Repeated or missing example')
    return selected

def orders(tokenizer,fresh,replay,plan):
    steps=plan['max_steps'];size=plan['rows_per_group']
    fresh_selected=choose(fresh,steps*size,plan['fresh_mixture'],plan['seed'])
    replay_selected=choose(replay,steps*size,plan['replay_mixture'],plan['seed']+1)
    result=[]
    for rows,selected,seed in [(fresh,fresh_selected,plan['seed']+2),(replay,replay_selected,plan['seed']+3)]:
        lengths=[len(encode(tokenizer,rows[i],False)[0]) for i in selected]
        order=training_order(lengths,seed,size,1)
        result.append([[selected[int(i)] for i in group] for group in order])
    if len(result[0])!=steps or len(result[1])!=steps:raise ValueError('Unexpected update budget')
    return result
