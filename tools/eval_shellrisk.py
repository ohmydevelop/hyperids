import argparse, hashlib, json, math, re, time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer
from hyperids import schema

ROOT = Path('/home/ubuntu/Works/hyperids')
MODEL_DIR = ROOT / 'model/checkpoints_gpu/final_model_v9'
TEST_PARQUET = '/tmp/shellrisk/data/test-00000-of-00001.parquet'
HF_DATASET = 'kontext-security/ShellRisk-Bench'
KESTREL_URL = 'https://huggingface.co/kontext-security/Kestrel/resolve/b3a600213c0b30ce4d4e2c59fe85978e1b416a96/classifier.json'
KESTREL_SHA = '1df8b3e5f2bfc4e1fe95230ee9b3d37f63aaa461a8551982fcc0dd7103c8221b'

URL_RE = re.compile(r"(?i)https?://[^\t-\r\x1c-\x1f\x85\s'\"`;|)]+")
IP_RE = re.compile(r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
B64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")

class Kestrel:
    def __init__(self, model_path: Path):
        raw = model_path.read_text(encoding='utf-8')
        assert hashlib.sha256(raw.encode()).hexdigest() == KESTREL_SHA
        m = json.loads(raw)
        self.ngrams = m['ngrams']; self.ngram_min, self.ngram_max = m['ngram_range']; self.intercept = m['intercept']
    def normalize(self, text):
        if text == '': return text
        text = URL_RE.sub('http://example.com', text)
        text = IP_RE.sub('1.1.1.1', text)
        text = B64_RE.sub('BASE64', text)
        return text
    def score(self, command):
        normalized = self.normalize(command).lower()
        counts = {}
        for word in normalized.split():
            padded = ' ' + word + ' '
            L = len(padded)
            for n in range(self.ngram_min, self.ngram_max + 1):
                if L <= n:
                    if padded in self.ngrams: counts[padded] = counts.get(padded,0)+1
                    break
                for off in range(L-n+1):
                    ng = padded[off:off+n]
                    if ng in self.ngrams: counts[ng] = counts.get(ng,0)+1
        norm_sq = 0.0; dot = 0.0
        for ng, count in counts.items():
            coef_idf, idf = self.ngrams[ng]
            tfidf = count*idf
            norm_sq += tfidf*tfidf
            dot += count*coef_idf
        if norm_sq == 0.0: return self.intercept
        return dot/math.sqrt(norm_sq) + self.intercept

def segs(ids, w=120, s=90):
    if len(ids) <= w: return [ids]
    out=[]; st=0
    while True:
        out.append(ids[st:st+w])
        if st+w >= len(ids): break
        st += s
    if out[-1][-1] != ids[-1]: out.append(ids[-w:])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=0, help='limit samples for smoke')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--threads', type=int, default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)

    if not Path(TEST_PARQUET).exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_DATASET, filename='data/test-00000-of-00001.parquet', repo_type='dataset', local_dir='/tmp/shellrisk')
    df = pd.read_parquet(TEST_PARQUET)
    if args.max:
        df = df.head(args.max).reset_index(drop=True)
    commands = df['command'].astype(str).tolist()
    y = (df['label'] == 'risky').astype(int).tolist()

    # Kestrel
    kestrel_path = Path('/tmp/shellrisk/classifier.json')
    if not kestrel_path.exists():
        print('downloading Kestrel', flush=True)
        kestrel_path.write_bytes(urlopen(KESTREL_URL, timeout=60).read())
    kestrel = Kestrel(kestrel_path)
    kestrel_scores = []
    t0=time.time()
    for c in commands:
        kestrel_scores.append(kestrel.score(c))
    kestrel_time = time.time()-t0

    # HyperIDs
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, add_prefix_space=True)
    IDS = schema.all_label_ids(); OFF = schema.group_offsets(); VERDICT = IDS[OFF['verdict'][0]:OFF['verdict'][1]]
    model = GLiClassModel.from_pretrained(MODEL_DIR).eval()
    prefix_ids = tok(''.join(f'<<LABEL>>{l}' for l in IDS)+'<<SEP>>', add_special_tokens=False)['input_ids']
    sep_id = tok.sep_token_id; pad_id = tok.pad_token_id
    all_seg_rows = []  # (cmd_idx, segment_ids)
    for i,c in enumerate(commands):
        cids = tok(c, add_special_tokens=False)['input_ids']
        for sg in segs(cids):
            all_seg_rows.append((i, sg))
    merged = [[None, None] for _ in commands]  # best_verdict_logits, best_mal_prob
    t0=time.time()
    with torch.no_grad():
        for b in range(0, len(all_seg_rows), args.batch):
            chunk = all_seg_rows[b:b+args.batch]
            full = []
            for i,sg in chunk:
                ids = (prefix_ids + sg + [sep_id])[:512]
                full.append(ids)
            L = max(len(x) for x in full)
            inp = torch.tensor([x + [pad_id]*(L-len(x)) for x in full], dtype=torch.long)
            mask = (inp != pad_id).long()
            lg = model(input_ids=inp, attention_mask=mask, max_num_classes=len(IDS)).logits.float()
            if lg.ndim == 1: lg = lg.unsqueeze(0)
            lg = lg.numpy()
            for (i,sg), row in zip(chunk, lg):
                v = row[:3]
                mx = float(v.max())
                e = np.exp(v-mx); p = e/e.sum()
                mal = float(p[2])
                if merged[i][0] is None or mal > merged[i][1]:
                    merged[i] = [v, mal]
    hyper_time = time.time()-t0
    hyper_strict = [int(np.argmax(m[0]) == 2) for m in merged]
    hyper_loose = [int(np.argmax(m[0]) >= 1) for m in merged]

    def metrics(preds, name):
        tp = sum(1 for p,yt in zip(preds,y) if p==1 and yt==1)
        fp = sum(1 for p,yt in zip(preds,y) if p==1 and yt==0)
        fn = sum(1 for p,yt in zip(preds,y) if p==0 and yt==1)
        tn = sum(1 for p,yt in zip(preds,y) if p==0 and yt==0)
        precision = tp/(tp+fp) if tp+fp else float('nan')
        recall = tp/(tp+fn) if tp+fn else float('nan')
        f1 = 2*precision*recall/(precision+recall) if precision+recall else float('nan')
        acc = (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) else float('nan')
        fpr = fp/(fp+tn) if fp+tn else float('nan')
        return {'name':name,'n':len(preds),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':round(precision,4),'recall':round(recall,4),'f1':round(f1,4),'accuracy':round(acc,4),'fpr':round(fpr,6)}
    k0 = [int(s >= 0.0) for s in kestrel_scores]
    k40 = [int(s >= 0.40) for s in kestrel_scores]
    rows=[metrics(k0,'Kestrel thr=0.0'), metrics(k40,'Kestrel thr=0.40'), metrics(hyper_strict,'HyperIDs strict(malicious)'), metrics(hyper_loose,'HyperIDs loose(malicious+suspicious)')]
    report={'dataset':'ShellRisk-Bench test','n':len(commands),'positive':sum(y),'negative':len(y)-sum(y),'hyper_threads':args.threads,
            'elapsed_sec':{'kestrel_score_all':round(kestrel_time,3),'hyperids_score_all':round(hyper_time,3)},
            'metrics':rows}

    # latency microbench on first 200 commands
    nbench=min(200,len(commands))
    bench_cmds=commands[:nbench]
    # Kestrel warmup + runs
    for c in bench_cmds[:10]: kestrel.score(c)
    ktimes=[]
    t0=time.perf_counter()
    for c in bench_cmds: kestrel.score(c)
    ktimes.append((time.perf_counter()-t0)*1000/nbench)
    # HyperIDs: one-by-one with batched-like? measure per command using existing merged? Better full pipeline
    htimes=[]
    for c in bench_cmds[:10]:
        cids=tok(c, add_special_tokens=False)['input_ids']
        _=model(input_ids=torch.tensor([(prefix_ids+cids+[sep_id])[:512]], dtype=torch.long), attention_mask=torch.ones((1, min(512,len(prefix_ids+cids+[sep_id])))), max_num_classes=31)
    t0=time.perf_counter(); n=0
    for c in bench_cmds:
        cids=tok(c, add_special_tokens=False)['input_ids']
        ids=(prefix_ids+cids+[sep_id])[:512]
        tin=torch.tensor([ids], dtype=torch.long); msk=torch.ones_like(tin)
        with torch.no_grad():
            _=model(input_ids=tin, attention_mask=msk, max_num_classes=31)
        n+=1
    htimes.append((time.perf_counter()-t0)*1000/n)
    report['latency_ms_per_cmd']={'kestrel_python':round(ktimes[0],4),'hyperids_pytorch_cpu_1cmd':round(htimes[0],3)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out=Path('/tmp/shellrisk_compare.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('wrote',out)

if __name__=='__main__':
    main()
