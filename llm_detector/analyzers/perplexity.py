"""Local perplexity scoring via distilgpt2.

AI text has low perplexity (< 20); human text typically > 35.
Ref: GLTR (Gehrmann et al. 2019), DetectGPT (Mitchell et al. 2023)

Surprisal diversity features based on:
- DivEye (variance of per-token surprisal)
- "When AI Settles Down" (volatility decay across text halves)
"""

from llm_detector.compat import HAS_PERPLEXITY, get_perplexity_model

if HAS_PERPLEXITY:
    import torch as _torch

_PPL_EMPTY = {
    'perplexity': 0.0, 'determination': None, 'confidence': 0.0,
    'surprisal_variance': 0.0, 'surprisal_first_half_var': 0.0,
    'surprisal_second_half_var': 0.0, 'volatility_decay_ratio': 1.0,
}


def run_perplexity(text):
    """Calculate token-level perplexity using distilgpt2.

    Returns dict with perplexity, determination, confidence, and
    surprisal diversity features (variance, half-variances, decay ratio).
    """
    if not HAS_PERPLEXITY:
        return {**_PPL_EMPTY, 'reason': 'Perplexity scoring unavailable (transformers/torch not installed)'}

    words = text.split()
    if len(words) < 50:
        return {**_PPL_EMPTY, 'reason': 'Perplexity: text too short'}

    model, tokenizer = get_perplexity_model()

    encodings = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=1024)
    input_ids = encodings.input_ids

    if input_ids.size(1) < 10:
        return {**_PPL_EMPTY, 'reason': 'Perplexity: too few tokens after encoding'}

    with _torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = _torch.exp(loss).item()

    # ── Surprisal diversity & volatility decay ──
    surprisal_variance = 0.0
    volatility_decay_ratio = 1.0
    first_half_var = 0.0
    second_half_var = 0.0
    n_tokens = 0
    try:
        with _torch.no_grad():
            logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = _torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        losses = per_token_loss.float().cpu().numpy()
        n_tokens = len(losses)

        if n_tokens >= 10:
            surprisal_variance = float(losses.var())
            mid = n_tokens // 2
            first_half_var = float(losses[:mid].var()) if mid > 1 else surprisal_variance
            second_half_var = float(losses[mid:].var()) if (n_tokens - mid) > 1 else surprisal_variance
            volatility_decay_ratio = (first_half_var / second_half_var) if second_half_var > 1e-6 else 1.0
    except Exception:
        pass  # Variance features are supplementary

    if ppl <= 15.0:
        det = 'AMBER'
        conf = min(0.65, (20.0 - ppl) / 20.0)
        reason = f"Low perplexity ({ppl:.1f}): highly predictable text"
    elif ppl <= 25.0:
        det = 'YELLOW'
        conf = min(0.35, (30.0 - ppl) / 30.0)
        reason = f"Moderate perplexity ({ppl:.1f}): somewhat predictable"
    else:
        det = None
        conf = 0.0
        reason = f"Normal perplexity ({ppl:.1f}): consistent with human text"

    # Layer 2: DivEye + Volatility compound upgrade (Basani & Chen; Sun et al.)
    diveye_signal = surprisal_variance < 2.0 and n_tokens >= 30
    volatility_signal = volatility_decay_ratio > 1.5 and n_tokens >= 40

    if diveye_signal and volatility_signal:
        if det is None:
            det = 'YELLOW'
            conf = min(0.40, 0.20 + (2.0 - surprisal_variance) * 0.05
                       + (volatility_decay_ratio - 1.0) * 0.05)
            reason = (f"Surprisal uniformity (var={surprisal_variance:.2f}, "
                      f"decay={volatility_decay_ratio:.2f}): machine rhythm detected")
        elif det == 'YELLOW':
            det = 'AMBER'
            conf = min(0.65, conf + 0.15)
            reason += (f" + DivEye(var={surprisal_variance:.2f}, "
                       f"decay={volatility_decay_ratio:.2f})")
        elif det == 'AMBER':
            conf = min(0.80, conf + 0.10)
            reason += (f" + DivEye(var={surprisal_variance:.2f}, "
                       f"decay={volatility_decay_ratio:.2f})")
    elif diveye_signal or volatility_signal:
        if det is not None:
            conf = min(conf + 0.05, 0.70)
            if diveye_signal:
                reason += f" + low_variance({surprisal_variance:.2f})"
            else:
                reason += f" + volatility_decay({volatility_decay_ratio:.2f})"

    return {
        'perplexity': round(ppl, 2),
        'determination': det,
        'confidence': conf,
        'reason': reason,
        'surprisal_variance': round(surprisal_variance, 4),
        'surprisal_first_half_var': round(first_half_var, 4),
        'surprisal_second_half_var': round(second_half_var, 4),
        'volatility_decay_ratio': round(volatility_decay_ratio, 4),
    }
