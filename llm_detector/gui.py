"""Desktop GUI for the LLM Detection Pipeline."""

import os
import threading
from collections import Counter

from llm_detector.compat import HAS_TK
from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv

if HAS_TK:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

_DET_COLORS = {
    'RED': '#d32f2f',
    'AMBER': '#f57c00',
    'MIXED': '#7b1fa2',
    'YELLOW': '#fbc02d',
    'REVIEW': '#0288d1',
    'GREEN': '#388e3c',
}

_CHANNELS = ['prompt_structure', 'stylometry', 'continuation', 'windowing']


class DetectorGUI:
    """Desktop GUI with channel ablation, mode selection, and detailed output."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Detector Pipeline v0.66")
        self.root.geometry("1120x860")

        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')
        self.mode_var = tk.StringVar(value='auto')
        self.show_details_var = tk.BooleanVar(value=True)

        self.ablation_vars = {}
        for ch in _CHANNELS:
            self.ablation_vars[ch] = tk.BooleanVar(value=False)

        self._build_layout()

    def _build_layout(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # File input row
        file_row = ttk.Frame(frame)
        file_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(file_row, text='Input file (CSV/XLSX):').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(side=tk.LEFT)

        # File options row
        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(opts, text='Prompt column').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=18).grid(row=0, column=1, sticky='w', padx=6)
        ttk.Label(opts, text='Sheet (xlsx)').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=16).grid(row=0, column=3, sticky='w', padx=6)
        ttk.Label(opts, text='Attempter filter').grid(row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=18).grid(row=0, column=5, sticky='w', padx=6)

        # Mode & options row
        mode_row = ttk.Frame(frame)
        mode_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(mode_row, text='Detection mode:').pack(side=tk.LEFT)
        ttk.Combobox(mode_row, textvariable=self.mode_var,
                     values=['auto', 'task_prompt', 'generic_aigt'],
                     width=14, state='readonly').pack(side=tk.LEFT, padx=(6, 16))
        ttk.Checkbutton(mode_row, text='Show details', variable=self.show_details_var).pack(side=tk.LEFT, padx=(0, 16))

        # Channel ablation
        ablation_frame = ttk.LabelFrame(frame, text='Channel Ablation (disable for experiments)')
        ablation_frame.pack(fill=tk.X, pady=(0, 8))
        for ch in _CHANNELS:
            ttk.Checkbutton(ablation_frame, text=ch, variable=self.ablation_vars[ch]).pack(side=tk.LEFT, padx=8, pady=4)

        # Continuation analysis
        l3 = ttk.LabelFrame(frame, text='Continuation Analysis (DNA-GPT)')
        l3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(l3, text='Provider').grid(row=0, column=0, sticky='w', padx=6, pady=6)
        ttk.Combobox(l3, textvariable=self.provider_var, values=['anthropic', 'openai'], width=12, state='readonly').grid(row=0, column=1, sticky='w', pady=6)
        ttk.Label(l3, text='API Key (optional)').grid(row=0, column=2, sticky='w', padx=(16, 6), pady=6)
        ttk.Entry(l3, textvariable=self.api_key_var, show='*').grid(row=0, column=3, sticky='ew', padx=(0, 6), pady=6)
        l3.columnconfigure(3, weight=1)

        # Text input
        ttk.Label(frame, text='Single text input (optional):').pack(anchor='w')
        self.text_input = tk.Text(frame, height=8, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(4, 8))

        # Action buttons
        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Analyze Text', command=lambda: self._run_async(self._analyze_text)).pack(side=tk.LEFT)
        ttk.Button(actions, text='Analyze File', command=lambda: self._run_async(self._analyze_file)).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text='Clear Output', command=self._clear_output).pack(side=tk.LEFT)

        # Results output with scrollbar
        ttk.Label(frame, text='Results:').pack(anchor='w')
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output = tk.Text(output_frame, height=20, wrap=tk.WORD,
                              font=('Consolas', 10), yscrollcommand=scrollbar.set)
        self.output.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.output.yview)

        # Color tags for determinations
        for det, color in _DET_COLORS.items():
            self.output.tag_configure(det, foreground=color)
        self.output.tag_configure('HEADER', foreground='#1565c0', font=('Consolas', 10, 'bold'))
        self.output.tag_configure('DIM', foreground='#757575')
        self.output.tag_configure('ALERT', foreground='#d32f2f', font=('Consolas', 10, 'bold'))

        ttk.Label(frame, textvariable=self.status_var).pack(anchor='w', pady=(8, 0))

    def _get_disabled_channels(self):
        disabled = [ch for ch, var in self.ablation_vars.items() if var.get()]
        return disabled or None

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[('Data files', '*.csv *.xlsx *.xlsm'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _clear_output(self):
        self.output.delete('1.0', tk.END)
        self.status_var.set('Ready')

    def _run_async(self, fn):
        self.status_var.set('Running...')

        def runner():
            try:
                fn()
                self.root.after(0, lambda: self.status_var.set('Done'))
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set('Error'))
                self.root.after(0, lambda: messagebox.showerror('Analysis Error', str(exc)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text, tag=None):
        def do_append():
            if tag:
                self.output.insert(tk.END, text, tag)
            else:
                self.output.insert(tk.END, text)
            self.output.see(tk.END)
        self.root.after(0, do_append)

    def _display_result(self, result):
        det = result.get('determination', 'GREEN')
        conf = result.get('confidence', 0)
        wc = result.get('word_count', 0)
        reason = result.get('reason', '')

        # Main determination line
        self._append(f"  {det}", det)
        self._append(f" | conf={conf:.2f} | words={wc}\n")
        self._append(f"  {reason}\n", 'DIM')

        if not self.show_details_var.get():
            self._append('\n')
            return

        # Calibrated confidence
        cal_conf = result.get('calibrated_confidence')
        if cal_conf is not None:
            self._append(f"  Calibrated: {cal_conf:.2f}", 'HEADER')
            stratum = result.get('calibration_stratum', '')
            conformity = result.get('conformity_level', '')
            if stratum or conformity:
                self._append(f" ({stratum}/{conformity})", 'DIM')
            self._append('\n')

        # Channel details
        cd = result.get('channel_details', {})
        channels = cd.get('channels', {})
        mode = cd.get('mode', '?')
        self._append(f"  Mode: {mode}\n", 'HEADER')

        if channels:
            self._append("  Channels:\n", 'HEADER')
            for ch_name, info in channels.items():
                sev = info.get('severity', 'GREEN')
                score = info.get('score', 0)
                eligible = info.get('mode_eligible', True)
                sufficient = info.get('data_sufficient', True)
                disabled = info.get('disabled', False)

                status_parts = []
                if disabled:
                    status_parts.append('DISABLED')
                if not sufficient:
                    status_parts.append('no-data')
                if not eligible:
                    status_parts.append('mode-ineligible')
                status = f" [{', '.join(status_parts)}]" if status_parts else ''

                self._append(f"    {ch_name:20s} ", None)
                self._append(f"{sev:6s}", sev)
                self._append(f" score={score:.2f}{status}\n")

        # Attack types
        attack_types = result.get('norm_attack_types', [])
        if attack_types:
            self._append(f"  Attacks neutralized: {', '.join(attack_types)}\n", 'ALERT')

        # Binoculars score
        bino = result.get('binoculars_score', 0)
        bino_det = result.get('binoculars_determination')
        if bino and bino > 0:
            self._append(f"  Binoculars: {bino:.4f}", None)
            if bino_det:
                self._append(f" ({bino_det})", bino_det)
            self._append('\n')

        # Shadow model disagreement
        shadow = result.get('shadow_determination')
        if shadow and shadow != det:
            self._append(f"  Shadow model: {shadow} (disagrees)\n", 'ALERT')

        # Detection spans summary
        spans = result.get('detection_spans', [])
        if spans:
            span_sources = Counter(s.get('source', '?') for s in spans)
            span_str = ', '.join(f"{src}={ct}" for src, ct in span_sources.items())
            self._append(f"  Detection spans: {len(spans)} ({span_str})\n", 'DIM')

        self._append('\n')

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Enter text to analyze.'))
            return

        mode = self.mode_var.get()
        result = analyze_prompt(
            text,
            run_l3=True,
            api_key=self.api_key_var.get().strip() or None,
            dna_provider=self.provider_var.get(),
            mode=mode if mode != 'auto' else 'auto',
            disabled_channels=self._get_disabled_channels(),
        )
        self._display_result(result)

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Choose a CSV/XLSX file to analyze.'))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            tasks = load_xlsx(path, sheet=self.sheet_var.get().strip() or None, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.csv':
            tasks = load_csv(path, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        else:
            self.root.after(0, lambda: messagebox.showerror('Unsupported file', f'Unsupported extension: {ext}'))
            return
        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo('No tasks', 'No qualifying prompts found.'))
            return

        api_key = self.api_key_var.get().strip() or None
        mode = self.mode_var.get()
        disabled = self._get_disabled_channels()
        counts = Counter()

        for i, task in enumerate(tasks, 1):
            r = analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                run_l3=True,
                api_key=api_key,
                dna_provider=self.provider_var.get(),
                mode=mode if mode != 'auto' else 'auto',
                disabled_channels=disabled,
            )
            counts[r['determination']] += 1
            self._append(f"[{i}/{len(tasks)}] ")
            self._display_result(r)

        parts = []
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0 or det in ('RED', 'AMBER', 'YELLOW', 'GREEN'):
                parts.append(f"{det}={ct}")
        self._append(f"Summary: {' | '.join(parts)}\n", 'HEADER')

    @staticmethod
    def _format_result(result):
        return (
            f"{result.get('determination')} | conf={result.get('confidence', 0):.2f} | "
            f"words={result.get('word_count', 0)} | reason={result.get('reason', '')}"
        )


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()
