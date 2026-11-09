"""Run the frozen historical test only for the already selected finalist."""
import argparse
from pathlib import Path
from sml_v1.common import read_json, file_sha256
from sml_v1.tokenization import Tokenizer
from .lora_repair_common import REPORT
from .lora_repair_eval import load
from .text_eval import evaluate, reference_inference

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    selected=read_json(REPORT/'selection_frozen.json')['winner']
    if str(a.bundle)!=selected['checkpoint'] or file_sha256(a.bundle/'manifest.json')!=selected['checkpoint_manifest']:raise ValueError('Only the frozen finalist may open this final test')
    model,state=load(a.bundle)
    with reference_inference(model):evaluate(model,Tokenizer(a.input/'tokenizer'),a.input/'corpus',read_json(a.input/'plan.json'),state['step'],a.output,a.input/'retention',split='test')
