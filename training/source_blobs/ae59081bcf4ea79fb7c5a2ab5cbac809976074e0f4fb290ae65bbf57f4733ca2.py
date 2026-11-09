"""Read legacy and grounded SFT provenance without changing checkpoint bytes."""
SERIALIZATIONS=('explicit-ids-assistant-mask-v1','mixed-chat-and-plain-qa-masks-v1')


def sft_info(metadata):
    if metadata.get('training_kind')=='dpo':
        contract=metadata.get('dpo_contract',{})
        if contract.get('format')!='opensml-streaming-dpo-contract-v1' or not isinstance(contract.get('tokenizer'),str):
            raise ValueError('Unsupported DPO metadata')
        return dict(contract=contract,parent_step=metadata['pretraining_step'],parent_tokens=metadata['pretraining_tokens'],
            immediate_parent_step=metadata['source_step'],stage=metadata['sft_stage'],
            serialization='explicit-ids-assistant-mask-v1',posttraining='dpo')
    if 'sft' in metadata:
        info=metadata['sft']
        if info.get('format')!='opensml-sft-run-v1' or info.get('serialization','explicit-ids-assistant-mask-v1') not in SERIALIZATIONS:
            raise ValueError('Unsupported legacy SFT metadata')
        return dict(contract=info['contract'],parent_step=info.get('parent_step'),
            parent_tokens=info.get('parent_tokens'),serialization=info.get('serialization','explicit-ids-assistant-mask-v1'))
    if metadata.get('training_kind')=='sft':
        contract=metadata.get('sft_contract',{})
        if contract.get('format')!='opensml-grounded-sft-contract-v1' or not isinstance(contract.get('tokenizer'),str):
            raise ValueError('Unsupported grounded SFT metadata')
        return dict(contract=contract,parent_step=metadata.get('pretraining_step', metadata['source_step']),
            parent_tokens=metadata.get('pretraining_tokens', metadata['source_tokens']),
            immediate_parent_step=metadata['source_step'],stage=metadata.get('sft_stage',1),
            serialization='explicit-ids-assistant-mask-v1')
    return None
