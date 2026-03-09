"""Desktop GUI for the LLM Detection Pipeline."""

import os
import json
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
    """Full-featured desktop GUI exposing all pipeline capabilities."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Detector Pipeline v0.66")
        self.root.geometry("1180x920")

        self._memory_store = None
        self._cal_table = None
        self._last_results = []
        self._last_text_map = {}

        self._init_vars()
        self._build_layout()

    def _init_vars(self):
        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')
        self.mode_var = tk.StringVar(value='auto')
        self.show_details_var = tk.BooleanVar(value=True)
        self.dna_model_var = tk.StringVar()
        self.dna_samples_var = tk.StringVar(value='3')
        self.no_layer3_var = tk.BooleanVar(value=False)
        self.verbose_var = tk.BooleanVar(value=False)
        self.output_csv_var = tk.StringVar()
        self.html_report_var = tk.StringVar()
        self.cost_var = tk.StringVar(value='400.0')
        self.no_similarity_var = tk.BooleanVar(value=False)
        self.sim_threshold_var = tk.StringVar(value='0.40')
        self.sim_store_var = tk.StringVar()
        self.instructions_var = tk.StringVar()
        self.memory_var = tk.StringVar()
        self.collect_var = tk.StringVar()
        self.cal_table_var = tk.StringVar()
        self.calibrate_var = tk.StringVar()
        self.baselines_jsonl_var = tk.StringVar()
        self.baselines_csv_var = tk.StringVar()
        self.labeled_corpus_var = tk.StringVar()
        self.confirm_task_var = tk.StringVar()
        self.confirm_label_var = tk.StringVar(value='ai')
        self.confirm_reviewer_var = tk.StringVar()
        self.attempter_history_var = tk.StringVar()

        self.ablation_vars = {}
        for ch in _CHANNELS:
            self.ablation_vars[ch] = tk.BooleanVar(value=False)

    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1: Analysis
        self._build_analysis_tab(notebook)
        # Tab 2: Configuration
        self._build_config_tab(notebook)
        # Tab 3: Memory & Learning
        self._build_memory_tab(notebook)
        # Tab 4: Calibration & Baselines
        self._build_calibration_tab(notebook)
        # Tab 5: Reports
        self._build_reports_tab(notebook)

        # Status bar
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor='w', padx=10, pady=(0, 6))

    # ── Tab 1: Analysis ──────────────────────────────────────────────

    def _build_analysis_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Analysis  ')

        # File input
        file_row = ttk.Frame(tab)
        file_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(file_row, text='Input file:').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(side=tk.LEFT)

        # File options
        opts = ttk.Frame(tab)
        opts.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(opts, text='Prompt col').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=14).grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(opts, text='Sheet').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=14).grid(row=0, column=3, sticky='w', padx=4)
        ttk.Label(opts, text='Attempter').grid(row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=14).grid(row=0, column=5, sticky='w', padx=4)

        # Mode & detection options
        mode_row = ttk.Frame(tab)
        mode_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(mode_row, text='Mode:').pack(side=tk.LEFT)
        ttk.Combobox(mode_row, textvariable=self.mode_var,
                     values=['auto', 'task_prompt', 'generic_aigt'],
                     width=14, state='readonly').pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(mode_row, text='Show details', variable=self.show_details_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(mode_row, text='Verbose', variable=self.verbose_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(mode_row, text='Skip Layer 3', variable=self.no_layer3_var).pack(side=tk.LEFT)

        # Channel ablation
        abl = ttk.LabelFrame(tab, text='Channel Ablation')
        abl.pack(fill=tk.X, pady=(0, 6))
        for ch in _CHANNELS:
            ttk.Checkbutton(abl, text=ch, variable=self.ablation_vars[ch]).pack(side=tk.LEFT, padx=6, pady=3)

        # Text input
        ttk.Label(tab, text='Single text input:').pack(anchor='w')
        self.text_input = tk.Text(tab, height=7, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(2, 6))

        # Action buttons
        actions = ttk.Frame(tab)
        actions.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(actions, text='Analyze Text', command=lambda: self._run_async(self._analyze_text)).pack(side=tk.LEFT)
        ttk.Button(actions, text='Analyze File', command=lambda: self._run_async(self._analyze_file)).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text='Clear', command=self._clear_output).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(actions, text='Save CSV', command=self._save_results_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text='Save HTML Reports', command=self._save_html_reports).pack(side=tk.LEFT)

        # Results output
        output_frame = ttk.Frame(tab)
        output_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output = tk.Text(output_frame, height=16, wrap=tk.WORD,
                              font=('Consolas', 10), yscrollcommand=scrollbar.set)
        self.output.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.output.yview)

        for det, color in _DET_COLORS.items():
            self.output.tag_configure(det, foreground=color)
        self.output.tag_configure('HEADER', foreground='#1565c0', font=('Consolas', 10, 'bold'))
        self.output.tag_configure('DIM', foreground='#757575')
        self.output.tag_configure('ALERT', foreground='#d32f2f', font=('Consolas', 10, 'bold'))

    # ── Tab 2: Configuration ─────────────────────────────────────────

    def _build_config_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Configuration  ')

        # DNA-GPT / Continuation
        dna = ttk.LabelFrame(tab, text='Continuation Analysis (DNA-GPT)')
        dna.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dna, text='Provider').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Combobox(dna, textvariable=self.provider_var, values=['anthropic', 'openai'],
                     width=12, state='readonly').grid(row=0, column=1, sticky='w', pady=4)
        ttk.Label(dna, text='API Key').grid(row=0, column=2, sticky='w', padx=(12, 6), pady=4)
        ttk.Entry(dna, textvariable=self.api_key_var, show='*').grid(row=0, column=3, sticky='ew', padx=(0, 6), pady=4)
        dna.columnconfigure(3, weight=1)
        ttk.Label(dna, text='Model (optional)').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(dna, textvariable=self.dna_model_var, width=24).grid(row=1, column=1, columnspan=2, sticky='w', pady=4)
        ttk.Label(dna, text='Samples').grid(row=1, column=2, sticky='e', padx=(12, 6), pady=4)
        ttk.Spinbox(dna, textvariable=self.dna_samples_var, from_=1, to=10, width=4).grid(row=1, column=3, sticky='w', pady=4)

        # Similarity
        sim = ttk.LabelFrame(tab, text='Similarity Analysis')
        sim.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(sim, text='Disable similarity', variable=self.no_similarity_var).grid(
            row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Label(sim, text='Threshold').grid(row=0, column=1, sticky='w', padx=(12, 6), pady=4)
        ttk.Entry(sim, textvariable=self.sim_threshold_var, width=6).grid(row=0, column=2, sticky='w', pady=4)
        ttk.Label(sim, text='Sim store (JSONL)').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(sim, textvariable=self.sim_store_var).grid(row=1, column=1, columnspan=2, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(sim, text='...', width=3, command=lambda: self._browse_save(self.sim_store_var, [('JSONL', '*.jsonl')])).grid(
            row=1, column=3, sticky='w', padx=2, pady=4)
        ttk.Label(sim, text='Instructions file').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(sim, textvariable=self.instructions_var).grid(row=2, column=1, columnspan=2, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(sim, text='...', width=3, command=lambda: self._browse_open(self.instructions_var)).grid(
            row=2, column=3, sticky='w', padx=2, pady=4)
        sim.columnconfigure(2, weight=1)

        # Output
        out = ttk.LabelFrame(tab, text='Output Options')
        out.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(out, text='Output CSV').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.output_csv_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(out, text='...', width=3, command=lambda: self._browse_save(self.output_csv_var, [('CSV', '*.csv')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(out, text='HTML report dir').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.html_report_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(out, text='...', width=3, command=lambda: self._browse_dir(self.html_report_var)).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(out, text='Cost per prompt ($)').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.cost_var, width=8).grid(row=2, column=1, sticky='w', pady=4)
        out.columnconfigure(1, weight=1)

        # Baseline collection
        bl = ttk.LabelFrame(tab, text='Baseline Collection')
        bl.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(bl, text='Collect to JSONL').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.collect_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_save(self.collect_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        bl.columnconfigure(1, weight=1)

    # ── Tab 3: Memory & Learning ─────────────────────────────────────

    def _build_memory_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Memory & Learning  ')

        # Memory store directory
        mem = ttk.LabelFrame(tab, text='BEET Memory Store')
        mem.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(mem, text='Store directory').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(mem, textvariable=self.memory_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(mem, text='...', width=3, command=lambda: self._browse_dir(self.memory_var)).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(mem, text='Load', command=self._load_memory).grid(row=0, column=3, sticky='w', padx=6, pady=4)
        mem.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(mem)
        btn_row.grid(row=1, column=0, columnspan=4, sticky='w', padx=6, pady=4)
        ttk.Button(btn_row, text='Print Summary', command=lambda: self._run_async(self._memory_summary)).pack(side=tk.LEFT, padx=(0, 6))

        # Confirmations
        conf = ttk.LabelFrame(tab, text='Record Ground Truth Confirmation')
        conf.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(conf, text='Task ID').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(conf, textvariable=self.confirm_task_var, width=24).grid(row=0, column=1, sticky='w', padx=(0, 6), pady=4)
        ttk.Label(conf, text='Label').grid(row=0, column=2, sticky='w', padx=(12, 6), pady=4)
        ttk.Combobox(conf, textvariable=self.confirm_label_var, values=['ai', 'human'],
                     width=8, state='readonly').grid(row=0, column=3, sticky='w', pady=4)
        ttk.Label(conf, text='Reviewer').grid(row=0, column=4, sticky='w', padx=(12, 6), pady=4)
        ttk.Entry(conf, textvariable=self.confirm_reviewer_var, width=16).grid(row=0, column=5, sticky='w', padx=(0, 6), pady=4)
        ttk.Button(conf, text='Confirm', command=self._record_confirmation).grid(row=0, column=6, sticky='w', padx=6, pady=4)

        # Attempter history
        hist = ttk.LabelFrame(tab, text='Attempter History')
        hist.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(hist, text='Attempter name').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(hist, textvariable=self.attempter_history_var, width=24).grid(row=0, column=1, sticky='w', padx=(0, 6), pady=4)
        ttk.Button(hist, text='Show History', command=lambda: self._run_async(self._show_attempter_history)).grid(
            row=0, column=2, sticky='w', padx=6, pady=4)

        # Learning tools
        learn = ttk.LabelFrame(tab, text='Learning Tools (require memory store)')
        learn.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(learn, text='Labeled corpus (JSONL)').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(learn, textvariable=self.labeled_corpus_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(learn, text='...', width=3, command=lambda: self._browse_open(self.labeled_corpus_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        learn.columnconfigure(1, weight=1)

        btn_row2 = ttk.Frame(learn)
        btn_row2.grid(row=1, column=0, columnspan=3, sticky='w', padx=6, pady=4)
        ttk.Button(btn_row2, text='Rebuild Shadow Model', command=lambda: self._run_async(self._rebuild_shadow)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Rebuild Centroids', command=lambda: self._run_async(self._rebuild_centroids)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Discover Lexicon', command=lambda: self._run_async(self._discover_lexicon)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Rebuild All', command=lambda: self._run_async(self._rebuild_all)).pack(side=tk.LEFT)

    # ── Tab 4: Calibration & Baselines ───────────────────────────────

    def _build_calibration_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Calibration & Baselines  ')

        # Calibration
        cal = ttk.LabelFrame(tab, text='Conformal Calibration')
        cal.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cal, text='Cal table (JSON)').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cal, textvariable=self.cal_table_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cal, text='...', width=3, command=lambda: self._browse_open(self.cal_table_var, [('JSON', '*.json')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(cal, text='Load', command=self._load_cal_table).grid(row=0, column=3, sticky='w', padx=6, pady=4)
        cal.columnconfigure(1, weight=1)

        ttk.Label(cal, text='Build from JSONL').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cal, textvariable=self.calibrate_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cal, text='...', width=3, command=lambda: self._browse_open(self.calibrate_var, [('JSONL', '*.jsonl')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(cal, text='Build & Save', command=lambda: self._run_async(self._build_calibration)).grid(
            row=1, column=3, sticky='w', padx=6, pady=4)

        ttk.Button(cal, text='Rebuild from Memory', command=lambda: self._run_async(self._rebuild_calibration)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=6, pady=4)

        # Baseline analysis
        bl = ttk.LabelFrame(tab, text='Baseline Analysis')
        bl.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(bl, text='Baselines JSONL').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.baselines_jsonl_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_open(self.baselines_jsonl_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(bl, text='Output CSV').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.baselines_csv_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_save(self.baselines_csv_var, [('CSV', '*.csv')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(bl, text='Analyze Baselines', command=lambda: self._run_async(self._analyze_baselines)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=6, pady=4)
        bl.columnconfigure(1, weight=1)

    # ── Helpers ───────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[('Data files', '*.csv *.xlsx *.xlsm *.pdf'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _browse_open(self, var, filetypes=None):
        filetypes = filetypes or [('All files', '*.*')]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_save(self, var, filetypes=None):
        filetypes = filetypes or [('All files', '*.*')]
        path = filedialog.asksaveasfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _get_disabled_channels(self):
        disabled = [ch for ch, var in self.ablation_vars.items() if var.get()]
        return disabled or None

    def _get_dna_samples(self):
        try:
            return int(self.dna_samples_var.get())
        except ValueError:
            return 3

    def _get_cost(self):
        try:
            return float(self.cost_var.get())
        except ValueError:
            return 400.0

    def _get_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            env_var = ('ANTHROPIC_API_KEY' if self.provider_var.get() == 'anthropic'
                       else 'OPENAI_API_KEY')
            key = os.environ.get(env_var, '')
        return key or None

    def _get_sim_threshold(self):
        try:
            return float(self.sim_threshold_var.get())
        except ValueError:
            return 0.40

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
                self.root.after(0, lambda e=exc: messagebox.showerror('Error', str(e)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text, tag=None):
        def do_append():
            if tag:
                self.output.insert(tk.END, text, tag)
            else:
                self.output.insert(tk.END, text)
            self.output.see(tk.END)
        self.root.after(0, do_append)

    # ── Memory Store ──────────────────────────────────────────────────

    def _ensure_memory(self):
        if self._memory_store is not None:
            return True
        path = self.memory_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Memory required', 'Set a memory store directory first.'))
            return False
        self._load_memory()
        return self._memory_store is not None

    def _load_memory(self):
        path = self.memory_var.get().strip()
        if not path:
            messagebox.showinfo('Memory required', 'Set a memory store directory first.')
            return
        from llm_detector.memory import MemoryStore
        self._memory_store = MemoryStore(path)
        self.status_var.set(f'Memory store loaded: {path}')

    def _load_cal_table(self):
        path = self.cal_table_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showinfo('Cal table', 'Select a valid calibration table JSON file.')
            return
        from llm_detector.calibration import load_calibration
        self._cal_table = load_calibration(path)
        self.status_var.set(f"Calibration loaded: {self._cal_table['n_calibration']} records, "
                            f"{len(self._cal_table.get('strata', {}))} strata")

    # ── Analysis Actions ──────────────────────────────────────────────

    def _build_analyze_kwargs(self):
        kwargs = {
            'run_l3': not self.no_layer3_var.get(),
            'api_key': self._get_api_key(),
            'dna_provider': self.provider_var.get(),
            'dna_model': self.dna_model_var.get().strip() or None,
            'dna_samples': self._get_dna_samples(),
            'mode': self.mode_var.get(),
            'disabled_channels': self._get_disabled_channels(),
            'cal_table': self._cal_table,
            'memory_store': self._memory_store,
        }
        return kwargs

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Enter text to analyze.'))
            return
        kwargs = self._build_analyze_kwargs()
        result = analyze_prompt(text, **kwargs)

        # Shadow model check
        if self._memory_store:
            disagreement = self._memory_store.check_shadow_disagreement(result)
            result['shadow_disagreement'] = disagreement
            result['shadow_ai_prob'] = (disagreement or {}).get('shadow_ai_prob')

        self._last_results = [result]
        self._last_text_map = {'_single': text}
        self._display_result(result)

        # Collect baselines if configured
        collect_path = self.collect_var.get().strip()
        if collect_path:
            from llm_detector.baselines import collect_baselines
            collect_baselines([result], collect_path)
            self._append(f"  Baseline appended to {collect_path}\n", 'DIM')

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Choose a CSV/XLSX file.'))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            tasks = load_xlsx(path, sheet=self.sheet_var.get().strip() or None,
                              prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.csv':
            tasks = load_csv(path, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.pdf':
            if not HAS_PYPDF:
                self.root.after(0, lambda: messagebox.showerror(
                    'Missing dependency', 'PDF support requires pypdf: pip install pypdf'))
                return
            from llm_detector.io import load_pdf
            tasks = load_pdf(path)
        else:
            self.root.after(0, lambda: messagebox.showerror('Unsupported file', f'Unsupported: {ext}'))
            return

        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo('No tasks', 'No qualifying prompts found.'))
            return

        kwargs = self._build_analyze_kwargs()
        results = []
        text_map = {}
        counts = Counter()

        for i, task in enumerate(tasks, 1):
            r = analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                **kwargs,
            )
            results.append(r)
            tid = task.get('task_id', f'_row{i}')
            text_map[tid] = task['prompt']
            counts[r['determination']] += 1
            self._append(f"[{i}/{len(tasks)}] ")
            self._display_result(r)

        # Shadow model checks
        if self._memory_store:
            shadow_count = 0
            for r in results:
                disagreement = self._memory_store.check_shadow_disagreement(r)
                r['shadow_disagreement'] = disagreement
                r['shadow_ai_prob'] = (disagreement or {}).get('shadow_ai_prob')
                if disagreement:
                    shadow_count += 1
            if shadow_count:
                self._append(f"\nShadow model: {shadow_count} disagreements\n", 'ALERT')

        self._last_results = results
        self._last_text_map = text_map

        # Summary
        parts = []
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0 or det in ('RED', 'AMBER', 'YELLOW', 'GREEN'):
                parts.append(f"{det}={ct}")
        self._append(f"\nSummary: {' | '.join(parts)}\n", 'HEADER')

        # Similarity analysis
        if not self.no_similarity_var.get() and len(results) >= 2:
            self._run_similarity(results, text_map)

        # Cross-batch memory
        if self._memory_store:
            cross_flags = self._memory_store.cross_batch_similarity(results, text_map)
            if cross_flags:
                self._append(f"\nCross-batch memory: {len(cross_flags)} matches\n", 'HEADER')
                for cf in cross_flags[:5]:
                    self._append(f"  {cf['current_id'][:15]} <-> {cf['historical_id'][:15]} "
                                 f"(MH={cf['minhash_similarity']:.2f})\n", 'DIM')
            self._memory_store.record_batch(results, text_map)

        # Yellow alerts
        yellow = [r for r in results if r['determination'] == 'YELLOW']
        if yellow:
            self._append(f"\nYELLOW ({len(yellow)} minor signals):\n", 'YELLOW')
            for r in sorted(yellow, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                self._append(f"  {r.get('task_id', '')[:12]:12s} {r.get('occupation', '')[:40]:40s} | "
                             f"{r.get('reason', '')[:50]}\n", 'DIM')

        # Attempter profiling & channel pattern summary
        if len(results) >= 5:
            try:
                from llm_detector.reporting import (
                    profile_attempters, channel_pattern_summary,
                )
                profiles = profile_attempters(results)
                if profiles:
                    self._append(f"\nAttempter profiles: {len(profiles)} attempters\n", 'HEADER')
                    for p in profiles[:5]:
                        self._append(f"  {p['attempter'][:20]:20s} submissions={p['n_submissions']} "
                                     f"flag_rate={p['flag_rate']:.0f}%\n")
            except Exception:
                pass

            try:
                import io, sys
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    channel_pattern_summary(results)
                finally:
                    sys.stdout = old_stdout
                summary_text = buf.getvalue()
                if summary_text.strip():
                    self._append(f"\n{summary_text}", 'DIM')
            except Exception:
                pass

        # Financial impact
        if len(results) >= 10:
            try:
                from llm_detector.reporting import financial_impact
                impact = financial_impact(results, cost_per_prompt=self._get_cost())
                self._append(f"\nFinancial impact: waste={impact['waste_estimate']:.0f} "
                             f"projected_annual={impact.get('projected_annual_waste', 0):.0f}\n", 'HEADER')
            except Exception:
                pass

        # Collect baselines
        collect_path = self.collect_var.get().strip()
        if collect_path:
            from llm_detector.baselines import collect_baselines
            collect_baselines(results, collect_path)
            self._append(f"Baselines appended to {collect_path}\n", 'DIM')

    def _run_similarity(self, results, text_map):
        try:
            from llm_detector.similarity import (
                analyze_similarity, apply_similarity_adjustments,
                save_similarity_store, cross_batch_similarity,
            )
            instruction_text = None
            instr_path = self.instructions_var.get().strip()
            if instr_path and os.path.exists(instr_path):
                with open(instr_path, 'r') as f:
                    instruction_text = f.read()

            sim_pairs = analyze_similarity(
                results, text_map,
                jaccard_threshold=self._get_sim_threshold(),
                instruction_text=instruction_text,
            )
            if sim_pairs:
                self._append(f"\nSimilarity: {len(sim_pairs)} pairs flagged\n", 'HEADER')
                results[:] = apply_similarity_adjustments(results, sim_pairs, text_map)
                upgrades = [r for r in results if 'similarity_upgrade' in r]
                if upgrades:
                    self._append(f"  {len(upgrades)} determinations upgraded by similarity\n", 'ALERT')

            sim_store = self.sim_store_var.get().strip()
            if sim_store:
                cross_flags = cross_batch_similarity(results, text_map, sim_store)
                if cross_flags:
                    self._append(f"  Cross-batch similarity: {len(cross_flags)} matches\n", 'DIM')
                save_similarity_store(results, text_map, sim_store)
        except Exception:
            pass

    def _save_results_csv(self):
        if not self._last_results:
            messagebox.showinfo('No results', 'Run an analysis first.')
            return
        path = self.output_csv_var.get().strip()
        if not path:
            path = filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return

        import pandas as pd
        flat = []
        for r in self._last_results:
            row = {k: v for k, v in r.items() if k != 'preamble_details'}
            row['preamble_details'] = str(r.get('preamble_details', []))
            flat.append(row)
        pd.DataFrame(flat).to_csv(path, index=False)
        self.status_var.set(f'Results saved to {path}')

    def _save_html_reports(self):
        if not self._last_results:
            messagebox.showinfo('No results', 'Run an analysis first.')
            return
        report_dir = self.html_report_var.get().strip()
        if not report_dir:
            report_dir = filedialog.askdirectory(title='Select HTML report directory')
        if not report_dir:
            return

        flagged = [r for r in self._last_results if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if not flagged:
            messagebox.showinfo('No flagged', 'No flagged submissions to report.')
            return

        try:
            os.makedirs(report_dir, exist_ok=True)
            from llm_detector.html_report import generate_html_report
            for r in flagged:
                tid = r.get('task_id', 'unknown')[:20].replace('/', '_')
                path = os.path.join(report_dir, f"{tid}_{r['determination']}.html")
                generate_html_report(self._last_text_map.get(r.get('task_id', ''), ''), r, path)
            self.status_var.set(f'{len(flagged)} HTML reports written to {report_dir}')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ── Display ───────────────────────────────────────────────────────

    def _display_result(self, result):
        det = result.get('determination', 'GREEN')
        conf = result.get('confidence', 0)
        wc = result.get('word_count', 0)
        reason = result.get('reason', '')

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
                sufficient = info.get('data_sufficient', True)
                disabled = info.get('disabled', False)
                eligible = info.get('mode_eligible', True)

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

        # Verbose details
        if self.verbose_var.get():
            self._display_verbose(result)

        # Attack types
        attack_types = result.get('norm_attack_types', [])
        if attack_types:
            self._append(f"  Attacks neutralized: {', '.join(attack_types)}\n", 'ALERT')

        # Binoculars
        bino = result.get('binoculars_score', 0)
        bino_det = result.get('binoculars_determination')
        if bino and bino > 0:
            self._append(f"  Binoculars: {bino:.4f}", None)
            if bino_det:
                self._append(f" ({bino_det})", bino_det)
            self._append('\n')

        # Shadow model
        shadow = result.get('shadow_disagreement')
        if shadow:
            self._append(f"  Shadow: {shadow.get('interpretation', 'disagrees')}\n", 'ALERT')
            self._append(f"    Rule={shadow.get('rule_determination', '?')}, "
                         f"Model={shadow.get('shadow_ai_prob', 0):.1%} AI\n", 'DIM')

        # Detection spans
        spans = result.get('detection_spans', [])
        if spans:
            span_sources = Counter(s.get('source', '?') for s in spans)
            span_str = ', '.join(f"{src}={ct}" for src, ct in span_sources.items())
            self._append(f"  Detection spans: {len(spans)} ({span_str})\n", 'DIM')

        self._append('\n')

    def _display_verbose(self, r):
        delta = r.get('norm_obfuscation_delta', 0)
        lang = r.get('lang_support_level', 'SUPPORTED')
        if delta > 0 or lang != 'SUPPORTED':
            self._append(f"  NORM: delta={delta:.1%} invisible={r.get('norm_invisible_chars', 0)} "
                         f"homoglyphs={r.get('norm_homoglyphs', 0)}\n", 'DIM')
            self._append(f"  GATE: {lang} fw={r.get('lang_fw_coverage', 0):.2f} "
                         f"non_latin={r.get('lang_non_latin_ratio', 0):.2f}\n", 'DIM')
        self._append(f"  Preamble: {r.get('preamble_score', 0):.2f} ({r.get('preamble_severity', '-')})\n", 'DIM')
        self._append(f"  Fingerprints: {r.get('fingerprint_score', 0):.2f} ({r.get('fingerprint_hits', 0)} hits)\n", 'DIM')
        self._append(f"  PromptSig: composite={r.get('prompt_signature_composite', 0):.2f} "
                     f"CFD={r.get('prompt_signature_cfd', 0):.3f} MFSR={r.get('prompt_signature_mfsr', 0):.3f} "
                     f"frames={r.get('prompt_signature_distinct_frames', 0)}\n", 'DIM')
        self._append(f"  IDI: {r.get('instruction_density_idi', 0):.1f} "
                     f"(imp={r.get('instruction_density_imperatives', 0)} "
                     f"cond={r.get('instruction_density_conditionals', 0)})\n", 'DIM')
        self._append(f"  VSD: {r.get('voice_dissonance_vsd', 0):.1f} "
                     f"(voice={r.get('voice_dissonance_voice_score', 0):.1f} "
                     f"spec={r.get('voice_dissonance_spec_score', 0):.1f})\n", 'DIM')
        nssi = r.get('self_similarity_nssi_score', 0)
        if nssi > 0:
            self._append(f"  NSSI: {nssi:.3f} ({r.get('self_similarity_nssi_signals', 0)} sig, "
                         f"det={r.get('self_similarity_determination', 'n/a')})\n", 'DIM')
        bscore = r.get('continuation_bscore', 0)
        if bscore > 0:
            self._append(f"  DNA-GPT: BScore={bscore:.4f} "
                         f"(det={r.get('continuation_determination', 'n/a')})\n", 'DIM')

    # ── Memory & Learning Actions ─────────────────────────────────────

    def _memory_summary(self):
        if not self._ensure_memory():
            return
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self._memory_store.print_summary()
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _record_confirmation(self):
        if not self._ensure_memory():
            return
        task_id = self.confirm_task_var.get().strip()
        label = self.confirm_label_var.get()
        reviewer = self.confirm_reviewer_var.get().strip()
        if not task_id or not reviewer:
            messagebox.showinfo('Missing fields', 'Task ID and Reviewer are required.')
            return
        self._memory_store.record_confirmation(task_id, label, verified_by=reviewer)
        self.status_var.set(f'Confirmed: {task_id} = {label} by {reviewer}')

    def _show_attempter_history(self):
        if not self._ensure_memory():
            return
        name = self.attempter_history_var.get().strip()
        if not name:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Enter an attempter name.'))
            return
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self._memory_store.print_attempter_history(name)
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _rebuild_shadow(self):
        if not self._ensure_memory():
            return
        pkg = self._memory_store.rebuild_shadow_model()
        if pkg:
            self._append(f"Shadow model rebuilt: AUC={pkg['cv_auc']:.3f}\n", 'HEADER')
        else:
            self._append("Shadow model: insufficient labeled data\n", 'ALERT')

    def _rebuild_centroids(self):
        if not self._ensure_memory():
            return
        corpus = self.labeled_corpus_var.get().strip()
        if not corpus:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Set a labeled corpus JSONL path.'))
            return
        result = self._memory_store.rebuild_semantic_centroids(corpus)
        if result:
            self._append(f"Centroids rebuilt: separation={result['separation']:.4f}\n", 'HEADER')
        else:
            self._append("Centroids: insufficient labeled text\n", 'ALERT')

    def _discover_lexicon(self):
        if not self._ensure_memory():
            return
        corpus = self.labeled_corpus_var.get().strip()
        if not corpus:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Set a labeled corpus JSONL path.'))
            return
        candidates = self._memory_store.discover_lexicon_candidates(corpus)
        n_new = sum(1 for c in candidates
                    if not c.get('already_in_fingerprints') and not c.get('already_in_packs'))
        self._append(f"Lexicon discovery: {len(candidates)} candidates ({n_new} new)\n", 'HEADER')

    def _rebuild_all(self):
        if not self._ensure_memory():
            return
        self._append("Rebuilding all learned artifacts...\n", 'HEADER')

        cal = self._memory_store.rebuild_calibration()
        if cal:
            self._append(f"  Calibration: {cal['n_calibration']} samples\n")
            self._cal_table = cal
        else:
            self._append("  Calibration: insufficient data\n", 'ALERT')

        shadow = self._memory_store.rebuild_shadow_model()
        if shadow:
            self._append(f"  Shadow model: AUC={shadow['cv_auc']:.3f}\n")
        else:
            self._append("  Shadow model: insufficient data\n", 'ALERT')

        corpus = self.labeled_corpus_var.get().strip()
        if corpus:
            centroids = self._memory_store.rebuild_semantic_centroids(corpus)
            if centroids:
                self._append(f"  Centroids: separation={centroids['separation']:.4f}\n")
            else:
                self._append("  Centroids: insufficient text\n", 'ALERT')
            candidates = self._memory_store.discover_lexicon_candidates(corpus)
            n_new = sum(1 for c in candidates
                        if not c.get('already_in_fingerprints') and not c.get('already_in_packs'))
            self._append(f"  Lexicon: {len(candidates)} candidates ({n_new} new)\n")
        else:
            self._append("  Centroids/Lexicon: skipped (no labeled corpus)\n", 'DIM')

        self._append("Rebuild complete.\n", 'HEADER')

    # ── Calibration & Baselines Actions ───────────────────────────────

    def _build_calibration(self):
        jsonl_path = self.calibrate_var.get().strip()
        if not jsonl_path or not os.path.exists(jsonl_path):
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Select a valid baselines JSONL file.'))
            return
        from llm_detector.calibration import calibrate_from_baselines, save_calibration
        cal = calibrate_from_baselines(jsonl_path)
        if cal is None:
            self._append("Calibration failed: need >= 20 labeled human samples\n", 'ALERT')
            return
        cal_path = self.cal_table_var.get().strip()
        if not cal_path:
            cal_path = jsonl_path.replace('.jsonl', '_calibration.json')
            self.cal_table_var.set(cal_path)
        save_calibration(cal, cal_path)
        self._cal_table = cal
        self._append(f"Calibration built: {cal['n_calibration']} records, "
                     f"{len(cal.get('strata', {}))} strata\n", 'HEADER')
        self._append(f"  Global quantiles: {cal['global']}\n", 'DIM')
        self._append(f"  Saved to: {cal_path}\n", 'DIM')

    def _rebuild_calibration(self):
        if not self._ensure_memory():
            return
        cal = self._memory_store.rebuild_calibration()
        if cal:
            self._cal_table = cal
            self._append(f"Calibration rebuilt from memory: {cal['n_calibration']} samples\n", 'HEADER')
        else:
            self._append("Calibration rebuild: insufficient data\n", 'ALERT')

    def _analyze_baselines(self):
        jsonl_path = self.baselines_jsonl_var.get().strip()
        if not jsonl_path or not os.path.exists(jsonl_path):
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Select a valid baselines JSONL file.'))
            return
        csv_path = self.baselines_csv_var.get().strip() or None

        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            from llm_detector.baselines import analyze_baselines
            analyze_baselines(jsonl_path, output_csv=csv_path)
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())


    # ── Tab 5: Reports ─────────────────────────────────────────────────

    def _build_reports_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Reports  ')

        actions = ttk.Frame(tab)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Refresh Reports',
                   command=self._refresh_reports).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text='Export Baselines',
                   command=self._export_baselines).pack(side=tk.LEFT, padx=4)

        report_frame = ttk.Frame(tab)
        report_frame.pack(fill=tk.BOTH, expand=True)
        self.report_output = tk.Text(report_frame, wrap=tk.WORD,
                                     font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL,
                                  command=self.report_output.yview)
        self.report_output.configure(yscrollcommand=scrollbar.set)
        self.report_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _report_append(self, text):
        self.root.after(0, lambda: (
            self.report_output.insert(tk.END, text),
            self.report_output.see(tk.END)))

    def _refresh_reports(self):
        self.report_output.delete('1.0', tk.END)
        if not self._last_results:
            self._report_append('No results available. Run a batch analysis first.\n')
            return

        results = self._last_results
        counts = Counter(r['determination'] for r in results)
        self._report_append(f"{'=' * 60}\n")
        self._report_append(f"  BATCH SUMMARY (n={len(results)})\n")
        self._report_append(f"{'=' * 60}\n")
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0:
                pct = ct / len(results) * 100
                self._report_append(f"  {det:>8}: {ct:>4} ({pct:.1f}%)\n")

        # Attempter profiling
        if len(results) >= 5:
            try:
                from llm_detector.reporting import profile_attempters
                profiles = profile_attempters(results)
                if profiles:
                    self._report_append(f"\n{'=' * 60}\n")
                    self._report_append("  ATTEMPTER RISK PROFILES\n")
                    self._report_append(f"{'=' * 60}\n")
                    for p in profiles[:20]:
                        self._report_append(
                            f"  {p['attempter'][:20]:20s} "
                            f"n={p['n_submissions']:>3} "
                            f"flag={p['flag_rate']:.0f}% "
                            f"conf={p.get('mean_confidence', 0):.2f}\n")
            except Exception:
                pass

        # Channel pattern summary
        flagged = [r for r in results
                   if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if flagged:
            self._report_append(f"\n{'=' * 60}\n")
            self._report_append("  CHANNEL PATTERNS (flagged submissions)\n")
            self._report_append(f"{'=' * 60}\n")
            channel_counts = Counter()
            for r in flagged:
                cd = r.get('channel_details', {}).get('channels', {})
                for ch_name, info in cd.items():
                    if info.get('severity') not in ('GREEN', None):
                        channel_counts[ch_name] += 1
            for ch, ct in channel_counts.most_common():
                self._report_append(f"  {ch:20s}: {ct} flags\n")

        # Financial impact
        if len(results) >= 10:
            try:
                from llm_detector.reporting import financial_impact
                impact = financial_impact(results, cost_per_prompt=self._get_cost())
                self._report_append(f"\n{'=' * 60}\n")
                self._report_append("  FINANCIAL IMPACT ESTIMATE\n")
                self._report_append(f"{'=' * 60}\n")
                self._report_append(
                    f"  Total submissions:     {impact['total_submissions']}\n")
                self._report_append(
                    f"  Flag rate:             {impact['flag_rate']:.1%}\n")
                self._report_append(
                    f"  Waste estimate:        ${impact['waste_estimate']:,.0f}\n")
                self._report_append(
                    f"  Projected annual:      ${impact.get('projected_annual_waste', 0):,.0f}\n")
            except Exception:
                pass

    def _export_baselines(self):
        if not self._last_results:
            messagebox.showinfo('Export', 'No results to export. Run analysis first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.jsonl',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if not path:
            return
        try:
            from llm_detector.baselines import collect_baselines
            collect_baselines(self._last_results, path)
            messagebox.showinfo('Baselines', f'Results appended to {path}')
        except Exception as e:
            messagebox.showerror('Baselines Error', str(e))


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()
