"""Tiny correctness fixture for assistant-token weighting and five-rank updates."""
import argparse
from pathlib import Path
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten
import numpy as np
from sml_v1.common import atomic_json
from sml_v1.jaccl_control import ControlChannel
from sml_v1.model import TransformerConfig, TransformerLM
from sml_v1.precision import MasterAdamW
from .train import mixed_backward, state_digest
from .mixed import objective_weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-dir',type=Path,required=True);a=p.parse_args()
    a.job_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(a.job_dir/f'worker_rank{os.getenv("MLX_RANK","0")}.json',dict(pid=os.getpid(),output=str(a.job_dir)))
    group=mx.distributed.init(backend='jaccl',strict=True);rank=group.rank()
    if group.size()!=5:raise ValueError('Expected all five Macs')
    control=ControlChannel(mx,group,ring_mode='native')
    cfg=TransformerConfig(vocab_size=32,d_model=16,n_heads=2,n_kv_heads=1,n_layers=1,
        max_seq_len=32,mlp_multiple_of=16,mlp_ratio=2,ce_impl='reference',ffn_impl='reference')
    mx.set_default_device(mx.cpu);mx.random.seed(20261102)
    model=TransformerLM(cfg)
    # Counter-constructed parameters avoid device-specific random initializer drift.
    for name,value in tree_flatten(model.parameters()):
        array=(np.arange(value.size,dtype=np.float32).reshape(value.shape)%17-8)*.01
        if value.ndim==1:array=np.ones(value.shape,np.float32)
        model.load_weights([(name,mx.array(array))],strict=False)
    initial=[(name,np.array(value)) for name,value in tree_flatten(model.parameters())]
    batches=[7,3,2,2,2];accum=4;length=16
    x=np.arange(64*length,dtype=np.int32).reshape(64,length)%32
    y=(x+1)%32
    for row in range(64):y[row,:(row%15)]=-100
    # Model the partial final batch, including a rank with zero supervised rows.
    y[43:]=-100
    count=int((y!=-100).sum())
    flags=np.arange(64)%4==3;weights=objective_weights(y,flags)
    # The weight plane survives the exact int32 wire representation.
    if not np.array_equal(weights,weights.view(np.int32).copy().view(np.float32)):
        raise ValueError('SFT objective weight transport changed')
    full_x,full_y=x.reshape(accum,16,length),y.reshape(accum,16,length)
    lo,hi=sum(batches[:rank]),sum(batches[:rank+1])
    loss,grads=mixed_backward(model,accum,dict(sft_objective_weight=.8,replay_objective_weight=.2),compile_one=False)(
        mx.array(full_x[:,lo:hi]),mx.array(full_y[:,lo:hi]),mx.array(weights.reshape(accum,16,length)[:,lo:hi]),mx.array(flags.reshape(accum,16)[:,lo:hi]))
    grads=nn.average_gradients(tree_map(lambda g:g*5,grads),group=group,communication_stream=mx.cpu)
    mx.eval(loss,grads)
    error=0.
    if rank==0:
        reference=TransformerLM(cfg);reference.load_weights([(n,mx.array(v)) for n,v in initial])
        objective=nn.value_and_grad(reference,lambda xx,yy:(reference(xx,targets=yy)['token_loss']*mx.array(weights)).sum())
        expected_loss,expected_grads=objective(mx.array(x),mx.array(y));mx.eval(expected_grads,expected_loss)
        expected=dict(tree_flatten(expected_grads))
        error=max(float(mx.max(mx.abs(g-expected[n])).item()) for n,g in tree_flatten(grads))
        if error>2e-5:raise ValueError(f'Weighted SFT gradient differs from full batch: {error}')
    losses=control.exchange(loss.tolist(),'loss')
    optimizer=MasterAdamW(learning_rate=3e-5,betas=(.9,.95),weight_decay=.01)
    optimizer.init(model.trainable_parameters());optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
    digest=state_digest(model,optimizer);digests=control.exchange(digest,'updated-replicas')
    if len(set(digests))!=1:raise ValueError('SFT update replicas differ')
    # Save/reload optimizer/master weights and check exact state identity.
    path=a.job_dir/f'probe_rank{rank}.safetensors';op=a.job_dir/f'probe_opt_rank{rank}.safetensors'
    mx.save_safetensors(str(path),dict(tree_flatten(model.parameters())))
    mx.save_safetensors(str(op),dict(tree_flatten(optimizer.state)))
    from mlx.utils import tree_unflatten
    model.load_weights(str(path));optimizer.state=tree_unflatten(list(mx.load(str(op)).items()))
    if state_digest(model,optimizer)!=digest:raise ValueError('SFT optimizer roundtrip differs')
    # Exercise the production BF16/Metal objective and FP32 optimizer on every GPU.
    mx.set_default_device(mx.gpu)
    gpu_cfg=TransformerConfig(vocab_size=128,d_model=128,n_heads=2,n_kv_heads=1,n_layers=1,
        max_seq_len=32,mlp_multiple_of=256,mlp_ratio=2,ce_impl='metal',ffn_impl='packed-metal')
    gpu_model=TransformerLM(gpu_cfg)
    for name,value in tree_flatten(gpu_model.parameters()):
        array=(np.arange(value.size,dtype=np.float32).reshape(value.shape)%17-8)*.001
        if value.ndim==1:array=np.ones(value.shape,np.float32)
        gpu_model.load_weights([(name,mx.array(array).astype(mx.bfloat16))],strict=False)
    gpu_loss,gpu_grads=mixed_backward(gpu_model,accum,dict(sft_objective_weight=.8,replay_objective_weight=.2))(
        mx.array(full_x[:,lo:hi]),mx.array(full_y[:,lo:hi]),mx.array(weights.reshape(accum,16,length)[:,lo:hi]),mx.array(flags.reshape(accum,16)[:,lo:hi]))
    gpu_grads=nn.average_gradients(tree_map(lambda g:g*5,gpu_grads),group=group,communication_stream=mx.cpu)
    gpu_optimizer=MasterAdamW(learning_rate=1e-5,betas=(.9,.95),weight_decay=.01)
    gpu_optimizer.init(gpu_model.trainable_parameters());gpu_optimizer.update(gpu_model,gpu_grads)
    mx.eval(gpu_model.parameters(),gpu_optimizer.state,gpu_loss)
    if not bool(mx.all(mx.isfinite(gpu_loss)).item()):raise ValueError('Nonfinite Metal SFT loss')
    gpu_digest=state_digest(gpu_model,gpu_optimizer)
    if len(set(control.exchange(gpu_digest,'gpu-updated-replicas')))!=1:raise ValueError('GPU SFT replicas differ')
    receipt=dict(rank=rank,status='passed',weighted_loss_contribution=losses[rank][0],state=digest,gpu_state=gpu_digest)
    receipts=control.exchange(receipt,'receipts')
    if rank==0:
        atomic_json(a.job_dir/'result.json',dict(status='passed',max_gradient_error=error,
            global_supervised_tokens=count,objective_loss=sum(v[0] for v in losses),sft_loss=sum(v[1] for v in losses),replay_loss=sum(v[2] for v in losses),receipts=receipts))
        print(f'[sft-correctness] five ranks passed; max_gradient_error={error:.3g}',flush=True)
    control.exchange(True,'done')


if __name__=='__main__':main()
