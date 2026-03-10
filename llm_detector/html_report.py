"""HTML report generator with span-level highlighting."""

import html


_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px;
       margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
.header { border-bottom: 3px solid #1a1a1a; padding-bottom: 16px; margin-bottom: 24px; }
.det { font-size: 28px; font-weight: 700; }
.det-RED { color: #d32f2f; }
.det-AMBER { color: #f57c00; }
.det-YELLOW { color: #fbc02d; }
.det-GREEN { color: #388e3c; }
.det-MIXED { color: #1976d2; }
.meta { color: #666; font-size: 14px; margin-top: 8px; }
.text-container { background: white; border: 1px solid #e0e0e0; border-radius: 8px;
                  padding: 24px; line-height: 1.8; font-size: 15px; white-space: pre-wrap;
                  word-wrap: break-word; }
.signal { padding: 2px 0; border-bottom: 3px solid; cursor: help; }
.signal-CRITICAL { border-color: #ff1744; background: #ffebee; }
.signal-HIGH { border-color: #ff5722; background: #fbe9e7; }
.signal-MEDIUM { border-color: #ff9800; background: #fff3e0; }
.signal-pattern { border-color: #ff9800; background: #fff8e1; }
.signal-keyword { border-color: #42a5f5; background: #e3f2fd; }
.signal-uppercase { border-color: #e53935; background: #ffcdd2; }
.signal-fingerprint { border-color: #ab47bc; background: #f3e5f5; }
.signal-hot_window { border-color: #ef5350; background: #ffcdd2; }
.legend { margin-top: 24px; padding: 16px; background: #f5f5f5; border-radius: 8px;
          font-size: 13px; }
.legend span { display: inline-block; margin-right: 16px; padding: 2px 6px; }
.channels { margin-top: 24px; }
.ch-row { display: flex; align-items: center; padding: 8px 0;
          border-bottom: 1px solid #eee; font-size: 14px; }
.ch-name { width: 160px; font-weight: 600; }
.ch-sev { width: 80px; font-weight: 600; }
"""


def generate_html_report(text, result, output_path=None):
    """Generate an HTML report with highlighted detection spans.

    Args:
        text: Original input text.
        result: Pipeline result dict (must include 'detection_spans').
        output_path: Where to write the HTML file. If None, returns string.

    Returns HTML string, or writes to file and returns path.
    """
    spans = result.get('detection_spans', [])
    det = result.get('determination', 'GREEN')
    reason = result.get('reason', '')
    confidence = result.get('confidence', 0)
    task_id = result.get('task_id', '')
    word_count = result.get('word_count', 0)

    # Filter to character-level spans and sort
    char_spans = sorted(
        [s for s in spans if 'start' in s and 'end' in s],
        key=lambda s: s['start'],
    )

    highlighted = _apply_highlights(text, char_spans)

    # Channel summary
    cd = result.get('channel_details', {}).get('channels', {})
    channel_rows = []
    for ch_name in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
        info = cd.get(ch_name, {})
        sev = info.get('severity', 'GREEN')
        expl = info.get('explanation', '')[:80]
        channel_rows.append(f"""
            <div class="ch-row">
                <div class="ch-name">{html.escape(ch_name)}</div>
                <div class="ch-sev det-{sev}">{sev}</div>
                <div style="flex:1">{html.escape(expl)}</div>
            </div>""")

    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BEET Detection Report</title>
<style>{_CSS}</style></head><body>
<div class="header">
    <div class="det det-{det}">{det}</div>
    <div class="meta">
        Task: {html.escape(task_id)} | Words: {word_count} |
        Confidence: {confidence:.1%} |
        Mode: {result.get('mode', '?')}
    </div>
    <div class="meta" style="margin-top:4px">{html.escape(reason[:200])}</div>
</div>
<div class="text-container">{highlighted}</div>
<div class="legend">
    <strong>Legend:</strong>
    <span class="signal signal-CRITICAL">CRITICAL preamble</span>
    <span class="signal signal-HIGH">HIGH preamble</span>
    <span class="signal signal-pattern">Lexicon pack hit</span>
    <span class="signal signal-keyword">Keyword match</span>
    <span class="signal signal-uppercase">Uppercase normative</span>
    <span class="signal signal-fingerprint">Fingerprint word</span>
    <span class="signal signal-hot_window">Hot window</span>
</div>
<div class="channels">
    <h3>Channel Scores</h3>
    {''.join(channel_rows)}
</div>
</body></html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return output_path

    return report


_BATCH_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px;
       margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
.batch-header { border-bottom: 3px solid #1a1a1a; padding-bottom: 16px; margin-bottom: 24px; }
.batch-header h1 { margin: 0 0 8px 0; }
.summary-table { width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 14px; }
.summary-table th { text-align: left; padding: 8px 12px; background: #e0e0e0; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.submission { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 32px;
              background: white; overflow: hidden; }
.sub-header { padding: 16px 24px; border-bottom: 1px solid #e0e0e0; }
.det { font-weight: 700; }
.det-RED { color: #d32f2f; }
.det-AMBER { color: #f57c00; }
.det-YELLOW { color: #fbc02d; }
.det-GREEN { color: #388e3c; }
.det-MIXED { color: #1976d2; }
.meta { color: #666; font-size: 14px; margin-top: 4px; }
.text-container { padding: 24px; line-height: 1.8; font-size: 15px; white-space: pre-wrap;
                  word-wrap: break-word; }
.signal { padding: 2px 0; border-bottom: 3px solid; cursor: help; }
.signal-CRITICAL { border-color: #ff1744; background: #ffebee; }
.signal-HIGH { border-color: #ff5722; background: #fbe9e7; }
.signal-MEDIUM { border-color: #ff9800; background: #fff3e0; }
.signal-pattern { border-color: #ff9800; background: #fff8e1; }
.signal-keyword { border-color: #42a5f5; background: #e3f2fd; }
.signal-uppercase { border-color: #e53935; background: #ffcdd2; }
.signal-fingerprint { border-color: #ab47bc; background: #f3e5f5; }
.signal-hot_window { border-color: #ef5350; background: #ffcdd2; }
.channels { padding: 0 24px 16px; }
.ch-row { display: flex; align-items: center; padding: 6px 0;
          border-bottom: 1px solid #eee; font-size: 13px; }
.ch-name { width: 140px; font-weight: 600; }
.ch-sev { width: 70px; font-weight: 600; }
.legend { margin: 24px 0; padding: 16px; background: #f5f5f5; border-radius: 8px;
          font-size: 13px; }
.legend span { display: inline-block; margin-right: 16px; padding: 2px 6px; }
.toc { margin-bottom: 32px; }
.toc a { color: #1976d2; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""


def generate_batch_html_report(results, text_map, output_path=None):
    """Generate a single consolidated HTML report for multiple submissions.

    Args:
        results: List of pipeline result dicts (typically flagged ones).
        text_map: Dict mapping task_id -> original text.
        output_path: Where to write the HTML file. If None, returns string.

    Returns HTML string, or writes to file and returns path.
    """
    from datetime import datetime

    n_red = sum(1 for r in results if r.get('determination') == 'RED')
    n_amber = sum(1 for r in results if r.get('determination') == 'AMBER')
    n_mixed = sum(1 for r in results if r.get('determination') == 'MIXED')
    n_yellow = sum(1 for r in results if r.get('determination') == 'YELLOW')

    # Table of contents
    toc_rows = []
    for idx, r in enumerate(results):
        tid = r.get('task_id', f'submission_{idx}')
        det = r.get('determination', 'GREEN')
        conf = r.get('confidence', 0)
        att = r.get('attempter', '')
        label = html.escape(att or tid)
        toc_rows.append(
            f'<tr><td><a href="#sub-{idx}">{label}</a></td>'
            f'<td class="det det-{det}">{det}</td>'
            f'<td>{conf:.1%}</td>'
            f'<td>{html.escape(tid)}</td></tr>'
        )

    # Submission sections
    sections = []
    for idx, r in enumerate(results):
        tid = r.get('task_id', f'submission_{idx}')
        det = r.get('determination', 'GREEN')
        reason = r.get('reason', '')
        confidence = r.get('confidence', 0)
        word_count = r.get('word_count', 0)
        att = r.get('attempter', '')
        text = text_map.get(tid, '')

        spans = r.get('detection_spans', [])
        char_spans = sorted(
            [s for s in spans if 'start' in s and 'end' in s],
            key=lambda s: s['start'],
        )
        highlighted = _apply_highlights(text, char_spans)

        cd = r.get('channel_details', {}).get('channels', {})
        channel_rows = []
        for ch_name in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
            info = cd.get(ch_name, {})
            sev = info.get('severity', 'GREEN')
            expl = info.get('explanation', '')[:80]
            channel_rows.append(
                f'<div class="ch-row">'
                f'<div class="ch-name">{html.escape(ch_name)}</div>'
                f'<div class="ch-sev det-{sev}">{sev}</div>'
                f'<div style="flex:1">{html.escape(expl)}</div></div>'
            )

        att_label = f' | Fellow: {html.escape(att)}' if att else ''
        sections.append(f"""
<div class="submission" id="sub-{idx}">
    <div class="sub-header">
        <span class="det det-{det}" style="font-size:22px">{det}</span>
        <span class="meta" style="margin-left:12px">
            Task: {html.escape(tid)}{att_label} | Words: {word_count} |
            Confidence: {confidence:.1%} | Mode: {r.get('mode', '?')}
        </span>
        <div class="meta" style="margin-top:4px">{html.escape(reason[:200])}</div>
    </div>
    <div class="text-container">{highlighted}</div>
    <div class="channels">{''.join(channel_rows)}</div>
</div>""")

    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BEET Batch Detection Report</title>
<style>{_BATCH_CSS}</style></head><body>
<div class="batch-header">
    <h1>BEET Batch Detection Report</h1>
    <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
        Submissions: {len(results)} |
        <span class="det-RED">RED: {n_red}</span> |
        <span class="det-AMBER">AMBER: {n_amber}</span> |
        <span class="det-MIXED">MIXED: {n_mixed}</span> |
        <span class="det-YELLOW">YELLOW: {n_yellow}</span>
    </div>
</div>
<div class="toc">
    <h3>Flagged Submissions</h3>
    <table class="summary-table">
        <tr><th>Fellow / ID</th><th>Determination</th><th>Confidence</th><th>Task ID</th></tr>
        {''.join(toc_rows)}
    </table>
</div>
<div class="legend">
    <strong>Legend:</strong>
    <span class="signal signal-CRITICAL">CRITICAL preamble</span>
    <span class="signal signal-HIGH">HIGH preamble</span>
    <span class="signal signal-pattern">Lexicon pack hit</span>
    <span class="signal signal-keyword">Keyword match</span>
    <span class="signal signal-uppercase">Uppercase normative</span>
    <span class="signal signal-fingerprint">Fingerprint word</span>
    <span class="signal signal-hot_window">Hot window</span>
</div>
{''.join(sections)}
</body></html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return output_path

    return report


def _get_span_class(span):
    """Determine CSS class for a span based on its type/severity."""
    sev = span.get('severity')
    if sev in ('CRITICAL', 'HIGH', 'MEDIUM'):
        return sev
    span_type = span.get('type', 'pattern')
    if span_type in ('fingerprint', 'hot_window', 'keyword', 'uppercase'):
        return span_type
    return 'pattern'


_SEVERITY_ORDER = {
    'CRITICAL': 6, 'HIGH': 5, 'MEDIUM': 4,
    'uppercase': 3, 'fingerprint': 2, 'pattern': 1, 'keyword': 0,
    'hot_window': 1,
}


def _apply_highlights(text, spans):
    """Apply highlight markup to text at span positions.

    Handles overlapping spans by using the highest-severity span at each position.
    """
    if not spans:
        return html.escape(text)

    # Build a priority map: each character position gets the highest-severity annotation
    char_map = [None] * len(text)
    for span in spans:
        start = span.get('start', 0)
        end = span.get('end', start)
        css_class = _get_span_class(span)
        tooltip = span.get('pack', span.get('pattern', span.get('source', '')))
        priority = _SEVERITY_ORDER.get(css_class, 0)
        for i in range(max(0, start), min(end, len(text))):
            if char_map[i] is None or priority > char_map[i][2]:
                char_map[i] = (css_class, tooltip, priority)

    # Build output with runs of same annotation
    out = []
    i = 0
    while i < len(text):
        if char_map[i] is None:
            j = i
            while j < len(text) and char_map[j] is None:
                j += 1
            out.append(html.escape(text[i:j]))
            i = j
        else:
            css_class, tooltip, _ = char_map[i]
            j = i
            while j < len(text) and char_map[j] is not None and char_map[j][0] == css_class and char_map[j][1] == tooltip:
                j += 1
            out.append(
                f'<span class="signal signal-{css_class}" title="{html.escape(tooltip)}">'
                f'{html.escape(text[i:j])}</span>'
            )
            i = j

    return ''.join(out)
