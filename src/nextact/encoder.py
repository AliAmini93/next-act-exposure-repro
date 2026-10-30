"""Fine-tuned RoBERTa-base under the same randomised presence plan (D1, repeats 0-2) and the naive row split (D6).

Same targets, windows, labels, folds, groups and swaps as the lexical classifier (plans are rebuilt and their
fingerprints checked). Each fit starts from the pinned checkpoint (settings: FacebookAI/roberta-base revision
e2da8e2f811d1448a5b465c236feacd80ffbac7b). GPU. Row-level predictions go to <private>/lexical/<corpus>/;
analysis via `python -m nextact.lexical analyze --learner enc`.

Usage: python -m nextact.encoder --corpus sgd --designs D1,D6 [--dry-run]
"""
from nextact import paths
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from nextact import protocol as P  # noqa: E402
from nextact import lexical as R  # noqa: E402

ENC_SPEC = dict(model_id='FacebookAI/roberta-base', revision='e2da8e2f811d1448a5b465c236feacd80ffbac7b',
                max_len=160, batch=32, eval_batch=256, lr=3e-5, epochs=3, warmup=0.06, weight_decay=0.01,
                seed=1234, d1_repeats=(0, 1, 2), d6_repeats=(0,), d2_repeats=(0,))
ROLE_NAMES = {'sgd': {'usr': 'user', 'sys': 'system'}, 'star': {'usr': 'user', 'wiz': 'wizard'},
              'ami': {'self': 'same speaker', 'oth': 'other speaker'}}


def window_text(enc_hist, corpus):
    if not enc_hist:
        return '(start of dialogue)'
    names = ROLE_NAMES[corpus]
    return ' | '.join(f'{names.get(r, r)}: {t}' for r, t in enc_hist)


def model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download(ENC_SPEC['model_id'], revision=ENC_SPEC['revision'],
                             allow_patterns=['config.json', 'merges.txt', 'vocab.json', 'tokenizer.json',
                                             'tokenizer_config.json', 'model.safetensors'])


def encode_all(texts, path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.truncation_side = 'left'      # keep the most recent history when a window is too long
    enc = tok(list(texts), truncation=True, max_length=ENC_SPEC['max_len'])
    return enc['input_ids'], tok.pad_token_id, float(np.mean([len(x) >= ENC_SPEC['max_len'] for x in enc['input_ids']]))


def _batch(ids, idx, pad, device):
    import torch
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for k, i in enumerate(idx):
        s = ids[i]
        x[k, :len(s)] = torch.tensor(s)
        m[k, :len(s)] = 1
    return x.to(device, non_blocking=True), m.to(device, non_blocking=True)


def fit_predict_encoder(ids, pad, Y, multi, train_idx, eval_idx, seed, path):
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    dev = torch.device('cuda')
    n_cls = Y.shape[1]
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=n_cls,
        problem_type='multi_label_classification' if multi else 'single_label_classification').to(dev)
    no_decay = ('bias', 'LayerNorm.weight')
    groups = [{'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               'weight_decay': ENC_SPEC['weight_decay']},
              {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=ENC_SPEC['lr'], fused=True)
    bs = ENC_SPEC['batch']
    steps = ENC_SPEC['epochs'] * math.ceil(len(train_idx) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(ENC_SPEC['warmup'] * steps), steps)
    yt = torch.tensor(Y[train_idx].astype(np.float32) if multi else Y[train_idx].argmax(1), device=dev)
    pos = {int(j): k for k, j in enumerate(train_idx)}
    model.train()
    losses = []
    for ep in range(ENC_SPEC['epochs']):
        order = np_rng.permutation(len(train_idx))
        for b in range(0, len(order), bs):
            sel = train_idx[order[b:b + bs]]
            x, m = _batch(ids, sel, pad, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=m).logits.float()
            tgt = yt[[pos[int(j)] for j in sel]]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt) if multi
                    else torch.nn.functional.cross_entropy(logits, tgt))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
    model.eval()
    out = np.zeros((len(eval_idx), n_cls), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(eval_idx), ENC_SPEC['eval_batch']):
            sel = eval_idx[b:b + ENC_SPEC['eval_batch']]
            x, m = _batch(ids, sel, pad, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=m).logits.float()
            pr = torch.sigmoid(logits) if multi else torch.softmax(logits, -1)
            out[b:b + len(sel)] = pr.cpu().numpy()
    final_loss = float(np.mean(losses[-50:])) if losses else None
    del model, opt
    torch.cuda.empty_cache()
    return out, final_loss


def items_for(plans, design):
    reps = {'D1': ENC_SPEC['d1_repeats'], 'D6': ENC_SPEC['d6_repeats'], 'D2': ENC_SPEC['d2_repeats']}[design]
    return [it for it in plans[design] if it['repeat'] in reps]


def item_key(design, it):
    return f"{design}_{it['condition']}_r{it['repeat']}_f{it['fold']}_g{it['group'] if it['group'] is not None else 'x'}"


def run(corpus, designs, dry_run=False):
    import torch
    prep = R.prepare(corpus, R.frozen_n(corpus))
    T, Y = prep['T'], prep['Y']
    plans = R.build_plans(T, which=tuple(designs))
    R.verify_frozen(corpus, prep, plans)
    path = model_path()
    texts = [window_text(h, corpus) for h in T.enc_hist]
    ids, pad, trunc_share = encode_all(texts, path)
    multi = prep['kind'] == 'lrap'
    pdir = R.PRIV / corpus
    idir = pdir / 'enc_items'
    idir.mkdir(parents=True, exist_ok=True)
    log_path = R.paths.LOGS / 'lexical_encoder.jsonl'
    summary = []
    for d in designs:
        items = items_for(plans, d)
        for k, it in enumerate(items):
            f = idir / f'{item_key(d, it)}.npy'
            if f.is_file():
                continue
            assert not np.isin(it['eval_idx'], it['train_idx']).any(), 'evaluated target in its training set'
            if dry_run:
                print(d, k, len(it['train_idx']), len(it['eval_idx']))
                continue
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            seed = ENC_SPEC['seed'] + 100000 * ('D1', 'D6', 'D2').index(d) + k
            Pm, loss = fit_predict_encoder(ids, pad, Y, multi, it['train_idx'], it['eval_idx'], seed, path)
            np.save(f, Pm)
            rec = dict(corpus=corpus, design=d, item=k, n_items=len(items), seconds=round(time.time() - t0, 1),
                       peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), final_train_loss=loss,
                       n_train=int(len(it['train_idx'])), n_eval=int(len(it['eval_idx'])))
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        if dry_run:
            continue
        preds = {}
        for it in items:
            Pm = np.load(idir / f'{item_key(d, it)}.npy')
            key = (it['condition'], it['repeat'])
            preds.setdefault(key, np.full((len(T), Y.shape[1]), np.nan, dtype=np.float32))
            assert np.isnan(preds[key][it['eval_idx'], 0]).all()
            preds[key][it['eval_idx']] = Pm
        R.save_preds(pdir / f'{d}_enc.npz', preds, dict(design=d, n_items=len(items), spec=ENC_SPEC,
                                                         truncated_share=trunc_share))
        summary.append(dict(design=d, n_items=len(items), truncated_share=round(trunc_share, 4)))
    return summary


def decide_encoder(res, min_varying=P.MIN_VARYING_TARGETS):
    """Encoder decision without a purge arm: SIGNATURE_PRESENT = P1a, P1c and P1d; PARTIAL = P1a and P1c;
    FALSIFIED = P1a point <= 0 with enough varying targets; otherwise INCONCLUSIVE."""
    E = res['estimands']
    a = E.get('within_adj_continuation_lag1')
    if a is None or a['point'] is None or a.get('n_varying_targets', 0) < min_varying:
        return 'INCONCLUSIVE'
    if a['point'] <= 0:
        return 'FALSIFIED'

    def lo_pos(k):
        e = E.get(k)
        return e is not None and e['ci95'][0] is not None and e['ci95'][0] > 0
    p1a, p1c, p1d = (lo_pos('within_adj_continuation_lag1'), lo_pos('within_adj_interaction_lag1'),
                     lo_pos('within_adj_continuation_lag1_minus_placebo'))
    if p1a and p1c and p1d:
        return 'SIGNATURE_PRESENT'
    if p1a and p1c:
        return 'PARTIAL'
    return 'INCONCLUSIVE'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=R.CORPORA)
    ap.add_argument('--designs', default='D1,D6')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    print(json.dumps(run(a.corpus, a.designs.split(','), a.dry_run), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
