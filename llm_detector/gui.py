"""Desktop GUI for the LLM Detection Pipeline v0.66."""

import os
import threading
from collections import Counter, defaultdict

from llm_detector.compat import HAS_TK, HAS_PYPDF
from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv

if HAS_TK:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox


class DetectorGUI:
    """Tabbed desktop GUI for v0.66 pipeline analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Detector Pipeline v0.66")
        self.root.geometry("1100x820")

        # State
        self.results = []
        self.text_map = {}
        self.cal_table = None
        self.memory_store = None

        # Variables
        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')

        # Settings variables
        self.mode_var = tk.StringVar(value='auto')
        self.run_l3_var = tk.BooleanVar(value=True)
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.dna_model_var = tk.StringVar()
        self.dna_samples_var = tk.IntVar(value=3)
        self.cal_path_var = tk.StringVar()
        self.similarity_var = tk.BooleanVar(value=True)
        self.sim_threshold_var = tk.StringVar(value='0.40')
        self.memory_dir_var = tk.StringVar()

        # Channel ablation
        self.disable_prompt_structure = tk.BooleanVar(value=False)
        self.disable_stylometric = tk.BooleanVar(value=False)
        self.disable_continuation = tk.BooleanVar(value=False)
        self.disable_windowed = tk.BooleanVar(value=False)

        # Similarity store & instructions
        self.sim_store_var = tk.StringVar()
        self.instructions_var = tk.StringVar()
        self.cost_var = tk.StringVar(value='400')

        # Memory management
        self.confirm_task_var = tk.StringVar()
        self.confirm_label_var = tk.StringVar(value='ai')
        self.confirm_reviewer_var = tk.StringVar()
        self.corpus_path_var = tk.StringVar()

        # Batch state
        self.sim_pairs = []

        self._build_layout()

    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1: Analyze
        analyze_tab = ttk.Frame(notebook, padding=8)
        notebook.add(analyze_tab, text=' Analyze ')
        self._build_analyze_tab(analyze_tab)

        # Tab 2: Settings
        settings_tab = ttk.Frame(notebook, padding=8)
        notebook.add(settings_tab, text=' Settings ')
        self._build_settings_tab(settings_tab)

        # Tab 3: Reports
        reports_tab = ttk.Frame(notebook, padding=8)
        notebook.add(reports_tab, text=' Reports ')
        self._build_reports_tab(reports_tab)

        # Status bar
        ttk.Label(self.root, textvariable=self.status_var).pack(
            anchor='w', padx=10, pady=(0, 6))

    # ── Tab 1: Analyze ──────────────────────────────────────────

    def _build_analyze_tab(self, parent):
        # File input row
        file_row = ttk.Frame(parent)
        file_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(file_row, text='Input file:').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(
            side=tk.LEFT)

        # File options row
        opts = ttk.Frame(parent)
        opts.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(opts, text='Prompt column').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=16).grid(
            row=0, column=1, sticky='w', padx=4)
        ttk.Label(opts, text='Sheet (xlsx)').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=14).grid(
            row=0, column=3, sticky='w', padx=4)
        ttk.Label(opts, text='Attempter filter').grid(
            row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=16).grid(
            row=0, column=5, sticky='w', padx=4)

        # Single text input
        ttk.Label(parent, text='Single text input:').pack(anchor='w')
        self.text_input = tk.Text(parent, height=8, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(2, 6))

        # Action buttons
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(actions, text='Analyze Text',
                   command=lambda: self._run_async(self._analyze_text)).pack(
            side=tk.LEFT)
        ttk.Button(actions, text='Analyze File',
                   command=lambda: self._run_async(self._analyze_file)).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(actions, text='Export CSV',
                   command=self._export_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text='Clear',
                   command=self._clear_output).pack(side=tk.LEFT)

        # Results
        ttk.Label(parent, text='Results:').pack(anchor='w')
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(result_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                  command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Tab 2: Settings ─────────────────────────────────────────

    def _build_settings_tab(self, parent):
        # Pipeline Options
        pipeline_frame = ttk.LabelFrame(parent, text='Pipeline Options',
                                        padding=8)
        pipeline_frame.pack(fill=tk.X, pady=(0, 8))

        row0 = ttk.Frame(pipeline_frame)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text='Detection mode:').pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.mode_var,
                     values=['auto', 'task_prompt', 'generic_aigt'],
                     width=14, state='readonly').pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(row0, text='Enable Layer 3 (NSSI + DNA-GPT)',
                        variable=self.run_l3_var).pack(side=tk.LEFT, padx=16)

        # Channel ablation
        ch_frame = ttk.LabelFrame(parent, text='Channel Ablation (disable)',
                                  padding=8)
        ch_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(ch_frame, text='prompt_structure',
                        variable=self.disable_prompt_structure).pack(
            side=tk.LEFT, padx=8)
        ttk.Checkbutton(ch_frame, text='stylometric',
                        variable=self.disable_stylometric).pack(
            side=tk.LEFT, padx=8)
        ttk.Checkbutton(ch_frame, text='continuation',
                        variable=self.disable_continuation).pack(
            side=tk.LEFT, padx=8)
        ttk.Checkbutton(ch_frame, text='windowed',
                        variable=self.disable_windowed).pack(
            side=tk.LEFT, padx=8)

        # DNA-GPT
        dna_frame = ttk.LabelFrame(parent, text='DNA-GPT Continuation',
                                   padding=8)
        dna_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(dna_frame, text='Provider').grid(
            row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Combobox(dna_frame, textvariable=self.provider_var,
                     values=['anthropic', 'openai'], width=12,
                     state='readonly').grid(row=0, column=1, sticky='w', pady=4)
        ttk.Label(dna_frame, text='API Key').grid(
            row=0, column=2, sticky='w', padx=(16, 4), pady=4)
        ttk.Entry(dna_frame, textvariable=self.api_key_var, show='*').grid(
            row=0, column=3, sticky='ew', padx=4, pady=4)
        dna_frame.columnconfigure(3, weight=1)

        row1 = ttk.Frame(dna_frame)
        row1.grid(row=1, column=0, columnspan=4, sticky='w', pady=4)
        ttk.Label(row1, text='Model (optional):').pack(side=tk.LEFT, padx=4)
        ttk.Entry(row1, textvariable=self.dna_model_var, width=20).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(row1, text='Samples:').pack(side=tk.LEFT, padx=(16, 4))
        ttk.Spinbox(row1, textvariable=self.dna_samples_var,
                    from_=1, to=10, width=5).pack(side=tk.LEFT, padx=4)

        # Calibration
        cal_frame = ttk.LabelFrame(parent, text='Calibration Table', padding=8)
        cal_frame.pack(fill=tk.X, pady=(0, 8))
        cal_row = ttk.Frame(cal_frame)
        cal_row.pack(fill=tk.X)
        ttk.Entry(cal_row, textvariable=self.cal_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(cal_row, text='Browse',
                   command=self._browse_cal).pack(side=tk.LEFT, padx=4)
        ttk.Button(cal_row, text='Load',
                   command=self._load_calibration).pack(side=tk.LEFT)
        ttk.Button(cal_row, text='Build from JSONL',
                   command=self._build_calibration).pack(side=tk.LEFT, padx=4)

        # Similarity
        sim_frame = ttk.LabelFrame(parent, text='Similarity Analysis',
                                   padding=8)
        sim_frame.pack(fill=tk.X, pady=(0, 8))
        sim_row = ttk.Frame(sim_frame)
        sim_row.pack(fill=tk.X)
        ttk.Checkbutton(sim_row, text='Enable cross-submission similarity',
                        variable=self.similarity_var).pack(side=tk.LEFT)
        ttk.Label(sim_row, text='Threshold:').pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(sim_row, textvariable=self.sim_threshold_var, width=6).pack(
            side=tk.LEFT)

        sim_row2 = ttk.Frame(sim_frame)
        sim_row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(sim_row2, text='Similarity store (JSONL):').pack(
            side=tk.LEFT)
        ttk.Entry(sim_row2, textvariable=self.sim_store_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(sim_row2, text='Browse',
                   command=self._browse_sim_store).pack(side=tk.LEFT)

        sim_row3 = ttk.Frame(sim_frame)
        sim_row3.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(sim_row3, text='Instructions file:').pack(side=tk.LEFT)
        ttk.Entry(sim_row3, textvariable=self.instructions_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(sim_row3, text='Browse',
                   command=self._browse_instructions).pack(side=tk.LEFT)

        # Financial
        fin_frame = ttk.LabelFrame(parent, text='Financial Impact', padding=8)
        fin_frame.pack(fill=tk.X, pady=(0, 8))
        fin_row = ttk.Frame(fin_frame)
        fin_row.pack(fill=tk.X)
        ttk.Label(fin_row, text='Cost per prompt ($):').pack(side=tk.LEFT)
        ttk.Entry(fin_row, textvariable=self.cost_var, width=8).pack(
            side=tk.LEFT, padx=4)

        # Memory Store
        mem_frame = ttk.LabelFrame(parent, text='Memory Store (BEET)',
                                   padding=8)
        mem_frame.pack(fill=tk.X, pady=(0, 8))
        mem_row = ttk.Frame(mem_frame)
        mem_row.pack(fill=tk.X)
        ttk.Entry(mem_row, textvariable=self.memory_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(mem_row, text='Browse',
                   command=self._browse_memory_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(mem_row, text='Connect',
                   command=self._connect_memory).pack(side=tk.LEFT)
        self.memory_status = ttk.Label(mem_frame, text='Not connected')
        self.memory_status.pack(anchor='w', pady=(4, 0))

    # ── Tab 3: Reports ──────────────────────────────────────────

    def _build_reports_tab(self, parent):
        # Actions row
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Generate HTML Reports',
                   command=self._generate_html_reports).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(actions, text='Export Baselines',
                   command=self._export_baselines).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text='Refresh Reports',
                   command=self._refresh_reports).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text='Analyze Baselines',
                   command=self._analyze_baselines_ui).pack(side=tk.LEFT, padx=4)

        # Attempter history row
        hist_row = ttk.Frame(parent)
        hist_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hist_row, text='Attempter history:').pack(side=tk.LEFT)
        self.hist_name_var = tk.StringVar()
        ttk.Entry(hist_row, textvariable=self.hist_name_var, width=20).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(hist_row, text='Show',
                   command=self._show_attempter_history).pack(side=tk.LEFT)

        # Memory management
        mem_mgmt = ttk.LabelFrame(parent, text='Memory Management', padding=8)
        mem_mgmt.pack(fill=tk.X, pady=(0, 8))

        # Confirmation row
        conf_row = ttk.Frame(mem_mgmt)
        conf_row.pack(fill=tk.X, pady=2)
        ttk.Label(conf_row, text='Task ID:').pack(side=tk.LEFT)
        ttk.Entry(conf_row, textvariable=self.confirm_task_var, width=16).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(conf_row, text='Label:').pack(side=tk.LEFT)
        ttk.Combobox(conf_row, textvariable=self.confirm_label_var,
                     values=['ai', 'human'], width=8,
                     state='readonly').pack(side=tk.LEFT, padx=4)
        ttk.Label(conf_row, text='Reviewer:').pack(side=tk.LEFT)
        ttk.Entry(conf_row, textvariable=self.confirm_reviewer_var,
                  width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(conf_row, text='Confirm',
                   command=self._record_confirmation).pack(side=tk.LEFT, padx=4)

        # Summary + rebuild row
        rebuild_row = ttk.Frame(mem_mgmt)
        rebuild_row.pack(fill=tk.X, pady=2)
        ttk.Button(rebuild_row, text='Memory Summary',
                   command=self._show_memory_summary).pack(side=tk.LEFT, padx=4)
        ttk.Button(rebuild_row, text='Rebuild Calibration',
                   command=self._rebuild_calibration).pack(side=tk.LEFT, padx=4)
        ttk.Button(rebuild_row, text='Rebuild Shadow',
                   command=self._rebuild_shadow).pack(side=tk.LEFT, padx=4)
        ttk.Button(rebuild_row, text='Rebuild All',
                   command=self._rebuild_all).pack(side=tk.LEFT, padx=4)

        # Corpus row (for centroids + lexicon)
        corpus_row = ttk.Frame(mem_mgmt)
        corpus_row.pack(fill=tk.X, pady=2)
        ttk.Label(corpus_row, text='Labeled corpus (JSONL):').pack(
            side=tk.LEFT)
        ttk.Entry(corpus_row, textvariable=self.corpus_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(corpus_row, text='Browse',
                   command=self._browse_corpus).pack(side=tk.LEFT, padx=4)
        ttk.Button(corpus_row, text='Rebuild Centroids',
                   command=self._rebuild_centroids).pack(side=tk.LEFT, padx=4)
        ttk.Button(corpus_row, text='Discover Lexicon',
                   command=self._discover_lexicon).pack(side=tk.LEFT, padx=4)

        # Report output
        report_frame = ttk.Frame(parent)
        report_frame.pack(fill=tk.BOTH, expand=True)
        self.report_output = tk.Text(report_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL,
                                  command=self.report_output.yview)
        self.report_output.configure(yscrollcommand=scrollbar.set)
        self.report_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Helpers ──────────────────────────────────────────────────

    def _browse_file(self):
        filetypes = [('Data files', '*.csv *.xlsx *.xlsm *.pdf'),
                     ('All files', '*.*')]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.file_var.set(path)

    def _browse_cal(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if path:
            self.cal_path_var.set(path)

    def _browse_memory_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.memory_dir_var.set(path)

    def _browse_sim_store(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if path:
            self.sim_store_var.set(path)

    def _browse_instructions(self):
        path = filedialog.askopenfilename(
            filetypes=[('Text', '*.txt *.md'), ('All', '*.*')])
        if path:
            self.instructions_var.set(path)

    def _browse_corpus(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if path:
            self.corpus_path_var.set(path)

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
                self.root.after(0, lambda e=exc: messagebox.showerror(
                    'Analysis Error', str(e)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text):
        self.root.after(0, lambda: (
            self.output.insert(tk.END, text), self.output.see(tk.END)))

    def _report_append(self, text):
        self.root.after(0, lambda: (
            self.report_output.insert(tk.END, text),
            self.report_output.see(tk.END)))

    def _get_disabled_channels(self):
        disabled = set()
        if self.disable_prompt_structure.get():
            disabled.add('prompt_structure')
        if self.disable_stylometric.get():
            disabled.add('stylometric')
        if self.disable_continuation.get():
            disabled.add('continuation')
        if self.disable_windowed.get():
            disabled.add('windowed')
        return disabled or None

    def _get_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            env_var = ('ANTHROPIC_API_KEY' if self.provider_var.get() == 'anthropic'
                       else 'OPENAI_API_KEY')
            key = os.environ.get(env_var, '')
        return key or None

    def _get_dna_model(self):
        return self.dna_model_var.get().strip() or None

    # ── Calibration ──────────────────────────────────────────────

    def _load_calibration(self):
        path = self.cal_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showinfo('Calibration', 'Select a valid calibration JSON file.')
            return
        try:
            from llm_detector.calibration import load_calibration
            self.cal_table = load_calibration(path)
            n = self.cal_table.get('n_calibration', '?')
            strata = len(self.cal_table.get('strata', {}))
            messagebox.showinfo('Calibration',
                                f'Loaded: {n} records, {strata} strata')
        except Exception as e:
            messagebox.showerror('Calibration Error', str(e))

    def _build_calibration(self):
        jsonl_path = filedialog.askopenfilename(
            title='Select labeled baseline JSONL',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if not jsonl_path:
            return
        try:
            from llm_detector.calibration import calibrate_from_baselines, save_calibration
            cal = calibrate_from_baselines(jsonl_path)
            if cal is None:
                messagebox.showerror('Calibration',
                                     'Insufficient labeled human data (need >=20)')
                return
            cal_path = filedialog.asksaveasfilename(
                title='Save calibration table',
                defaultextension='.json',
                initialfile=os.path.basename(jsonl_path).replace('.jsonl', '_calibration.json'),
                filetypes=[('JSON', '*.json'), ('All', '*.*')])
            if not cal_path:
                return
            save_calibration(cal, cal_path)
            self.cal_table = cal
            self.cal_path_var.set(cal_path)
            n = cal.get('n_calibration', '?')
            strata = len(cal.get('strata', {}))
            messagebox.showinfo('Calibration',
                                f'Built and saved: {n} records, {strata} strata')
        except Exception as e:
            messagebox.showerror('Calibration Error', str(e))

    # ── Memory Store ─────────────────────────────────────────────

    def _connect_memory(self):
        path = self.memory_dir_var.get().strip()
        if not path:
            messagebox.showinfo('Memory', 'Select a memory store directory.')
            return
        try:
            from llm_detector.memory import MemoryStore
            self.memory_store = MemoryStore(path)
            import json
            config = json.loads(self.memory_store.config_path.read_text())
            stats = (f"Connected: {config.get('total_submissions', 0)} submissions, "
                     f"{config.get('total_attempters', 0)} attempters, "
                     f"{config.get('total_confirmed', 0)} confirmed")
            self.memory_status.config(text=stats)
        except Exception as e:
            messagebox.showerror('Memory Error', str(e))

    # ── Analysis ─────────────────────────────────────────────────

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo(
                'Input required', 'Enter text to analyze.'))
            return
        result = analyze_prompt(
            text,
            run_l3=self.run_l3_var.get(),
            api_key=self._get_api_key(),
            dna_provider=self.provider_var.get(),
            dna_model=self._get_dna_model(),
            dna_samples=self.dna_samples_var.get(),
            mode=self.mode_var.get(),
            cal_table=self.cal_table,
            disabled_channels=self._get_disabled_channels(),
            memory_store=self.memory_store,
        )

        self.results = [result]
        self.text_map = {}
        self._append(self._format_result_verbose(result) + '\n')

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo(
                'Input required', 'Choose a file to analyze.'))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            tasks = load_xlsx(path,
                              sheet=self.sheet_var.get().strip() or None,
                              prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.csv':
            tasks = load_csv(path,
                             prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.pdf':
            if not HAS_PYPDF:
                self.root.after(0, lambda: messagebox.showerror(
                    'Missing dependency', 'PDF support requires pypdf: pip install pypdf'))
                return
            from llm_detector.io import load_pdf
            tasks = load_pdf(path)
        else:
            self.root.after(0, lambda: messagebox.showerror(
                'Unsupported file', f'Unsupported extension: {ext}'))
            return

        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo(
                'No tasks', 'No qualifying prompts found.'))
            return

        api_key = self._get_api_key()
        disabled = self._get_disabled_channels()
        results = []
        text_map = {}
        n = len(tasks)

        for i, task in enumerate(tasks):
            self.root.after(0, lambda idx=i: self.status_var.set(
                f'Processing {idx+1}/{n}...'))
            r = analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                run_l3=self.run_l3_var.get(),
                api_key=api_key,
                dna_provider=self.provider_var.get(),
                dna_model=self._get_dna_model(),
                dna_samples=self.dna_samples_var.get(),
                mode=self.mode_var.get(),
                cal_table=self.cal_table,
                disabled_channels=disabled,
                memory_store=self.memory_store,
            )
            results.append(r)
            tid = task.get('task_id', f'_row{i}')
            text_map[tid] = task['prompt']

            # Show flagged results in detail, others as one-line
            if r['determination'] in ('RED', 'AMBER', 'MIXED'):
                self._append(f"[{i+1}/{n}] {self._format_result_verbose(r)}\n")
            else:
                self._append(f"[{i+1}/{n}] {self._format_result_brief(r)}\n")

        # Shadow model summary (disagreements set by pipeline via memory_store)
        if self.memory_store:
            shadow_count = sum(1 for r in results if r.get('shadow_disagreement'))
            if shadow_count:
                self._append(f"\nSHADOW MODEL: {shadow_count} disagreements\n")

        # Load instruction text for similarity baseline
        instruction_text = None
        instr_path = self.instructions_var.get().strip()
        if instr_path and os.path.exists(instr_path):
            with open(instr_path, 'r') as f:
                instruction_text = f.read()

        # Similarity analysis
        sim_threshold = float(self.sim_threshold_var.get() or 0.40)
        sim_pairs = []
        if self.similarity_var.get() and len(results) >= 2:
            try:
                from llm_detector.similarity import (
                    analyze_similarity, apply_similarity_adjustments,
                )
                sim_pairs = analyze_similarity(
                    results, text_map, jaccard_threshold=sim_threshold,
                    instruction_text=instruction_text)
                if sim_pairs:
                    results = apply_similarity_adjustments(
                        results, sim_pairs, text_map)
                    self._append(f"\nSIMILARITY: {len(sim_pairs)} pairs flagged\n")
                    for p in sim_pairs[:10]:
                        self._append(
                            f"  {p['id_a'][:15]} <-> {p['id_b'][:15]} "
                            f"(J={p['jaccard']:.2f})\n")
                    upgrades = [r for r in results if 'similarity_upgrade' in r]
                    if upgrades:
                        self._append(
                            f"\n  SIMILARITY ADJUSTMENTS: "
                            f"{len(upgrades)} determinations upgraded\n")
                        for r in upgrades:
                            su = r['similarity_upgrade']
                            self._append(
                                f"    {r.get('task_id', '')[:15]:15s} "
                                f"{su['original_determination']} -> "
                                f"{su['upgraded_to']}  ({su['reason'][:60]})\n")
            except Exception:
                pass
        self.sim_pairs = sim_pairs

        # Similarity store (cross-batch fingerprints)
        sim_store = self.sim_store_var.get().strip()
        if sim_store:
            try:
                from llm_detector.similarity import (
                    cross_batch_similarity, save_similarity_store,
                )
                cross_flags = cross_batch_similarity(results, text_map, sim_store)
                if cross_flags:
                    self._append(
                        f"\nCROSS-BATCH SIMILARITY: "
                        f"{len(cross_flags)} matches to previous batches\n")
                    for cf in cross_flags[:10]:
                        self._append(
                            f"  {cf['current_id'][:15]} <-> "
                            f"{cf['historical_id'][:15]} "
                            f"(MH={cf['minhash_similarity']:.2f})\n")
                save_similarity_store(results, text_map, sim_store)
            except Exception:
                pass

        # Memory store: cross-batch + record
        if self.memory_store:
            try:
                cross_flags = self.memory_store.cross_batch_similarity(
                    results, text_map)
                if cross_flags:
                    self._append(
                        f"\nCROSS-BATCH: {len(cross_flags)} matches to history\n")
                self.memory_store.record_batch(results, text_map)
            except Exception:
                pass

        # Store for Reports tab
        self.results = results
        self.text_map = text_map

        # Summary
        counts = Counter(r['determination'] for r in results)
        self._append(f"\n{'='*70}\n")
        self._append(f"  PIPELINE v0.66 RESULTS (n={len(results)})\n")
        self._append(f"{'='*70}\n")
        icons = {'RED': '[!]', 'AMBER': '[!]', 'YELLOW': '[~]',
                 'GREEN': '[+]', 'MIXED': '[?]', 'REVIEW': '[ ]'}
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0 or det in ('RED', 'AMBER', 'YELLOW', 'GREEN'):
                pct = ct / len(results) * 100
                self._append(
                    f"  {icons.get(det, ' ')} {det:>8}: {ct:>4} ({pct:.1f}%)\n")

        # Yellow submissions list
        yellow = [r for r in results if r['determination'] == 'YELLOW']
        if yellow:
            self._append(f"\n  YELLOW ({len(yellow)} minor signals):\n")
            for r in sorted(yellow, key=lambda x: x['confidence'], reverse=True)[:10]:
                self._append(
                    f"    {r.get('task_id', '')[:12]:12} "
                    f"{r.get('occupation', '')[:40]:40} | "
                    f"{r.get('reason', '')[:50]}\n")

    # ── Result Formatting ────────────────────────────────────────

    @staticmethod
    def _format_result_brief(result):
        det = result.get('determination', '?')
        conf = result.get('confidence', 0)
        wc = result.get('word_count', 0)
        reason = result.get('reason', '')
        tid = result.get('task_id', '')[:20]
        return f"{det} | conf={conf:.2f} | {wc}w | {tid} | {reason[:60]}"

    @staticmethod
    def _format_result_verbose(r):
        """Format a result with full layer details (matches CLI print_result)."""
        lines = []
        det = r.get('determination', '?')
        tid = r.get('task_id', '')[:20]
        occ = r.get('occupation', '')[:45]
        lines.append(f"  [{det}] {tid}  |  {occ}")
        lines.append(
            f"     Attempter: {r.get('attempter') or '(unknown)'} "
            f"| Stage: {r.get('stage', '')} "
            f"| Words: {r.get('word_count', 0)} "
            f"| Mode: {r.get('mode', '?')}")
        lines.append(f"     Reason: {r.get('reason', '')}")

        # Calibration
        cal_conf = r.get('calibrated_confidence')
        p_val = r.get('conformity_level')
        if cal_conf is not None and cal_conf != r.get('confidence'):
            cal_str = f"     Calibrated: conf={cal_conf:.3f}"
            if p_val is not None:
                cal_str += f"  conf_level={p_val:.3f}"
            cal_str += f"  [{r.get('calibration_stratum', '?')}]"
            lines.append(cal_str)

        # Shadow disagreement
        shadow = r.get('shadow_disagreement')
        if shadow:
            lines.append(f"     SHADOW: {shadow['interpretation']}")
            lines.append(
                f"         Rule={shadow['rule_determination']}, "
                f"Model={shadow['shadow_ai_prob']:.1%} AI")

        # Layer details
        delta = r.get('norm_obfuscation_delta', 0)
        lang = r.get('lang_support_level', 'SUPPORTED')
        if delta > 0 or lang != 'SUPPORTED':
            lines.append(
                f"     NORM obfuscation: {delta:.1%}  "
                f"invisible={r.get('norm_invisible_chars', 0)} "
                f"homoglyphs={r.get('norm_homoglyphs', 0)}")
            lines.append(
                f"     GATE support:     {lang} "
                f"(fw_coverage={r.get('lang_fw_coverage', 0):.2f}, "
                f"non_latin={r.get('lang_non_latin_ratio', 0):.2f})")

        lines.append(
            f"     Preamble:         "
            f"{r.get('preamble_score', 0):.2f} "
            f"({r.get('preamble_severity', '?')}, "
            f"{r.get('preamble_hits', 0)} hits)")

        if r.get('preamble_details'):
            for name, sev in r['preamble_details']:
                lines.append(f"         -> [{sev}] {name}")

        lines.append(
            f"     Fingerprints:     "
            f"{r.get('fingerprint_score', 0):.2f} "
            f"({r.get('fingerprint_hits', 0)} hits)")
        lines.append(
            f"     Prompt Sig:       "
            f"{r.get('prompt_signature_composite', 0):.2f}")
        lines.append(
            f"         CFD={r.get('prompt_signature_cfd', 0):.3f} "
            f"frames={r.get('prompt_signature_distinct_frames', 0)} "
            f"MFSR={r.get('prompt_signature_mfsr', 0):.3f}")
        lines.append(
            f"         meta={r.get('prompt_signature_meta_design', 0)} "
            f"FC={r.get('prompt_signature_framing', 0)}/3 "
            f"must={r.get('prompt_signature_must_rate', 0):.3f}/sent")
        lines.append(
            f"         contractions={r.get('prompt_signature_contractions', 0)} "
            f"numbered_criteria={r.get('prompt_signature_numbered_criteria', 0)}")

        lines.append(
            f"     IDI:              "
            f"{r.get('instruction_density_idi', 0):.1f}  "
            f"(imp={r.get('instruction_density_imperatives', 0)} "
            f"cond={r.get('instruction_density_conditionals', 0)} "
            f"Y/N={r.get('instruction_density_binary_specs', 0)} "
            f"MISS={r.get('instruction_density_missing_refs', 0)} "
            f"flag={r.get('instruction_density_flag_count', 0)})")

        lines.append(
            f"     VSD:              "
            f"{r.get('voice_dissonance_vsd', 0):.1f}  "
            f"(voice={r.get('voice_dissonance_voice_score', 0):.1f} "
            f"x spec={r.get('voice_dissonance_spec_score', 0):.1f})")
        lines.append(
            f"         gated={'YES' if r.get('voice_dissonance_voice_gated') else 'no'} "
            f"casual={r.get('voice_dissonance_casual_markers', 0)} "
            f"typos={r.get('voice_dissonance_misspellings', 0)}")
        lines.append(
            f"         cols={r.get('voice_dissonance_camel_cols', 0)} "
            f"calcs={r.get('voice_dissonance_calcs', 0)} "
            f"hedges={r.get('voice_dissonance_hedges', 0)}")

        if r.get('ssi_triggered'):
            lines.append(
                f"     SSI:  TRIGGERED  "
                f"(spec={r.get('voice_dissonance_spec_score', 0):.1f})")

        nssi_score = r.get('self_similarity_nssi_score', 0.0)
        nssi_det = r.get('self_similarity_determination')
        if nssi_score > 0 or nssi_det:
            lines.append(
                f"     NSSI:             {nssi_score:.3f}  "
                f"({r.get('self_similarity_nssi_signals', 0)} signals, "
                f"det={nssi_det or 'n/a'})")
            lines.append(
                f"         formulaic={r.get('self_similarity_formulaic_density', 0):.3f} "
                f"power_adj={r.get('self_similarity_power_adj_density', 0):.3f} "
                f"demo={r.get('self_similarity_demonstrative_density', 0):.3f} "
                f"trans={r.get('self_similarity_transition_density', 0):.3f}")
            lines.append(
                f"         sent_cv={r.get('self_similarity_sent_length_cv', 0):.3f} "
                f"comp_ratio={r.get('self_similarity_comp_ratio', 0):.3f} "
                f"hapax={r.get('self_similarity_hapax_ratio', 0):.3f} "
                f"(unique={r.get('self_similarity_unique_words', 0)})")

        bscore = r.get('continuation_bscore', 0.0)
        dna_det = r.get('continuation_determination')
        if bscore > 0 or dna_det:
            lines.append(
                f"     DNA-GPT:          BScore={bscore:.4f}  "
                f"(max={r.get('continuation_bscore_max', 0):.4f}, "
                f"samples={r.get('continuation_n_samples', 0)}, "
                f"det={dna_det or 'n/a'})")

        # Channel summary
        cd = r.get('channel_details', {})
        if cd.get('channels'):
            lines.append("     -- Channels --")
            for ch_name, ch_info in cd['channels'].items():
                sev = ch_info.get('severity', 'GREEN')
                disabled = ch_info.get('disabled', False)
                data_ok = ch_info.get('data_sufficient', True)
                if sev != 'GREEN' or disabled or not data_ok:
                    eligible = 'Y' if ch_info.get('mode_eligible') else 'o'
                    tags = []
                    if disabled:
                        tags.append('[disabled]')
                    if not data_ok:
                        tags.append('[insufficient data]')
                    tag_str = ' '.join(tags)
                    lines.append(
                        f"     {eligible} {ch_name:18s} {sev:6s} "
                        f"score={ch_info.get('score', 0):.2f}  "
                        f"{ch_info.get('explanation', '')[:50]}"
                        f"{'  ' + tag_str if tag_str else ''}")

        if cd.get('short_text_adjustment'):
            lines.append(
                f"     [short-text adjustment active, "
                f"{cd.get('active_channels', '?')} channels active]")

        return '\n'.join(lines)

    # ── Export ───────────────────────────────────────────────────

    def _export_csv(self):
        if not self.results:
            messagebox.showinfo('Export', 'No results to export. Run analysis first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if not path:
            return
        try:
            import pandas as pd
            flat = []
            # Build similarity lookup for similarity_flags column
            sim_lookup = defaultdict(list)
            for p in self.sim_pairs:
                sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
                sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
            for r in self.results:
                row = {k: v for k, v in r.items()
                       if k not in ('preamble_details', 'detection_spans')}
                row['preamble_details'] = str(r.get('preamble_details', []))
                row['audit_trail'] = str(r.get('audit_trail', []))
                row['channel_details'] = str(r.get('channel_details', {}))
                tid = row.get('task_id', '')
                row['similarity_flags'] = '; '.join(sim_lookup.get(tid, []))
                flat.append(row)
            pd.DataFrame(flat).to_csv(path, index=False)
            messagebox.showinfo('Export', f'Saved {len(flat)} results to {path}')
        except Exception as e:
            messagebox.showerror('Export Error', str(e))

    def _export_baselines(self):
        if not self.results:
            messagebox.showinfo('Export', 'No results to export. Run analysis first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.jsonl',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if not path:
            return
        try:
            from llm_detector.baselines import collect_baselines
            n = collect_baselines(self.results, path)
            messagebox.showinfo('Baselines', f'{n} records appended to {path}')
        except Exception as e:
            messagebox.showerror('Baselines Error', str(e))

    # ── Memory Management ────────────────────────────────────────

    def _record_confirmation(self):
        if not self.memory_store:
            messagebox.showinfo('Confirm', 'Connect a memory store in Settings first.')
            return
        task_id = self.confirm_task_var.get().strip()
        label = self.confirm_label_var.get()
        reviewer = self.confirm_reviewer_var.get().strip()
        if not task_id:
            messagebox.showinfo('Confirm', 'Enter a task ID.')
            return
        if label not in ('ai', 'human'):
            messagebox.showinfo('Confirm', "Label must be 'ai' or 'human'.")
            return
        try:
            self.memory_store.record_confirmation(task_id, label, verified_by=reviewer)
            messagebox.showinfo('Confirm', f"Recorded: {task_id} = {label}")
        except Exception as e:
            messagebox.showerror('Confirm Error', str(e))

    def _show_memory_summary(self):
        if not self.memory_store:
            messagebox.showinfo('Summary', 'Connect a memory store in Settings first.')
            return
        try:
            import json
            config = json.loads(self.memory_store.config_path.read_text())
            self.report_output.delete('1.0', tk.END)
            self._report_append(f"{'='*60}\n")
            self._report_append(f"  MEMORY STORE SUMMARY\n")
            self._report_append(f"{'='*60}\n")
            self._report_append(f"  Directory:      {self.memory_store.base_dir}\n")
            self._report_append(f"  Submissions:    {config.get('total_submissions', 0)}\n")
            self._report_append(f"  Batches:        {config.get('total_batches', 0)}\n")
            self._report_append(f"  Attempters:     {config.get('total_attempters', 0)}\n")
            self._report_append(f"  Confirmed:      {config.get('total_confirmed', 0)}\n")
            occs = config.get('occupations', [])
            if occs:
                self._report_append(f"  Occupations:    {', '.join(occs[:10])}\n")
            self._report_append(f"  Last updated:   {config.get('last_update', '?')}\n")
        except Exception as e:
            self._report_append(f"Error: {e}\n")

    def _rebuild_calibration(self):
        if not self.memory_store:
            messagebox.showinfo('Rebuild', 'Connect a memory store in Settings first.')
            return
        try:
            cal = self.memory_store.rebuild_calibration()
            if cal:
                self.cal_table = cal
                messagebox.showinfo('Rebuild',
                                    f"Calibration rebuilt: {cal['n_calibration']} samples")
            else:
                messagebox.showinfo('Rebuild', 'Insufficient data for calibration.')
        except Exception as e:
            messagebox.showerror('Rebuild Error', str(e))

    def _rebuild_shadow(self):
        if not self.memory_store:
            messagebox.showinfo('Rebuild', 'Connect a memory store in Settings first.')
            return
        try:
            shadow = self.memory_store.rebuild_shadow_model()
            if shadow:
                messagebox.showinfo('Rebuild',
                                    f"Shadow model built: AUC={shadow['cv_auc']:.3f}")
            else:
                messagebox.showinfo('Rebuild', 'Insufficient labeled data for shadow model.')
        except Exception as e:
            messagebox.showerror('Rebuild Error', str(e))

    def _rebuild_centroids(self):
        if not self.memory_store:
            messagebox.showinfo('Rebuild', 'Connect a memory store in Settings first.')
            return
        corpus = self.corpus_path_var.get().strip()
        if not corpus or not os.path.exists(corpus):
            messagebox.showinfo('Rebuild', 'Select a labeled corpus JSONL file.')
            return
        try:
            result = self.memory_store.rebuild_semantic_centroids(corpus)
            if result:
                messagebox.showinfo('Rebuild',
                                    f"Centroids rebuilt: separation={result['separation']:.4f}")
            else:
                messagebox.showinfo('Rebuild', 'Insufficient labeled text for centroids.')
        except Exception as e:
            messagebox.showerror('Rebuild Error', str(e))

    def _discover_lexicon(self):
        if not self.memory_store:
            messagebox.showinfo('Lexicon', 'Connect a memory store in Settings first.')
            return
        corpus = self.corpus_path_var.get().strip()
        if not corpus or not os.path.exists(corpus):
            messagebox.showinfo('Lexicon', 'Select a labeled corpus JSONL file.')
            return
        try:
            candidates = self.memory_store.discover_lexicon_candidates(corpus)
            n_new = sum(1 for c in candidates
                        if not c.get('already_in_fingerprints')
                        and not c.get('already_in_packs'))
            messagebox.showinfo('Lexicon',
                                f"{len(candidates)} candidates ({n_new} new)")
        except Exception as e:
            messagebox.showerror('Lexicon Error', str(e))

    def _rebuild_all(self):
        if not self.memory_store:
            messagebox.showinfo('Rebuild', 'Connect a memory store in Settings first.')
            return
        self.report_output.delete('1.0', tk.END)
        self._report_append(f"{'='*60}\n")
        self._report_append(f"  REBUILDING ALL LEARNED ARTIFACTS\n")
        self._report_append(f"{'='*60}\n")

        try:
            cal = self.memory_store.rebuild_calibration()
            if cal:
                self.cal_table = cal
                self._report_append(f"  + Calibration: {cal['n_calibration']} samples\n")
            else:
                self._report_append(f"  - Calibration: insufficient data\n")
        except Exception as e:
            self._report_append(f"  - Calibration: error ({e})\n")

        try:
            shadow = self.memory_store.rebuild_shadow_model()
            if shadow:
                self._report_append(f"  + Shadow model: AUC={shadow['cv_auc']:.3f}\n")
            else:
                self._report_append(f"  - Shadow model: insufficient labeled data\n")
        except Exception as e:
            self._report_append(f"  - Shadow model: error ({e})\n")

        corpus = self.corpus_path_var.get().strip()
        if corpus and os.path.exists(corpus):
            try:
                centroids = self.memory_store.rebuild_semantic_centroids(corpus)
                if centroids:
                    self._report_append(
                        f"  + Centroids: separation={centroids['separation']:.4f}\n")
                else:
                    self._report_append(f"  - Centroids: insufficient labeled text\n")
            except Exception as e:
                self._report_append(f"  - Centroids: error ({e})\n")

            try:
                candidates = self.memory_store.discover_lexicon_candidates(corpus)
                n_new = sum(1 for c in candidates
                            if not c.get('already_in_fingerprints')
                            and not c.get('already_in_packs'))
                self._report_append(
                    f"  + Lexicon: {len(candidates)} candidates ({n_new} new)\n")
            except Exception as e:
                self._report_append(f"  - Lexicon: error ({e})\n")
        else:
            self._report_append(f"  - Centroids: skipped (no labeled corpus)\n")
            self._report_append(f"  - Lexicon: skipped (no labeled corpus)\n")

        self._report_append(f"\n{'='*60}\n")

    def _analyze_baselines_ui(self):
        jsonl_path = filedialog.askopenfilename(
            title='Select baseline JSONL',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if not jsonl_path:
            return
        try:
            from llm_detector.baselines import analyze_baselines
            rows = analyze_baselines(jsonl_path)
            self.report_output.delete('1.0', tk.END)
            if not rows:
                self._report_append('No baseline records found.\n')
                return
            self._report_append(f"{'='*60}\n")
            self._report_append(f"  BASELINE ANALYSIS ({len(rows)} records)\n")
            self._report_append(f"{'='*60}\n")
            for row in rows[:50]:
                parts = [f"{k}={v}" for k, v in row.items()]
                self._report_append(f"  {', '.join(parts[:6])}\n")

            # Offer CSV export
            save = messagebox.askyesno('Baselines',
                                       f'Loaded {len(rows)} records. Export to CSV?')
            if save:
                csv_path = filedialog.asksaveasfilename(
                    defaultextension='.csv',
                    filetypes=[('CSV', '*.csv'), ('All', '*.*')])
                if csv_path:
                    analyze_baselines(jsonl_path, output_csv=csv_path)
                    messagebox.showinfo('Baselines', f'CSV saved to {csv_path}')
        except Exception as e:
            messagebox.showerror('Baselines Error', str(e))

    # ── Reports ──────────────────────────────────────────────────

    def _refresh_reports(self):
        self.report_output.delete('1.0', tk.END)
        if not self.results:
            self._report_append('No results available. Run a batch analysis first.\n')
            return

        # Summary
        counts = Counter(r['determination'] for r in self.results)
        self._report_append(f"{'='*60}\n")
        self._report_append(f"  BATCH SUMMARY (n={len(self.results)})\n")
        self._report_append(f"{'='*60}\n")
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0:
                pct = ct / len(self.results) * 100
                self._report_append(f"  {det:>8}: {ct:>4} ({pct:.1f}%)\n")

        # Attempter profiling
        if len(self.results) >= 5:
            try:
                from llm_detector.reporting import (
                    profile_attempters, financial_impact,
                )
                profiles = profile_attempters(self.results)
                if profiles:
                    self._report_append(f"\n{'='*60}\n")
                    self._report_append("  ATTEMPTER RISK PROFILES\n")
                    self._report_append(f"{'='*60}\n")
                    for p in profiles[:20]:
                        self._report_append(
                            f"  {p['attempter'][:20]:20s} "
                            f"n={p['total']:>3} "
                            f"flag={p['flag_rate']:.0%} "
                            f"R={p['red']:>2} A={p['amber']:>2} "
                            f"Y={p['yellow']:>2} G={p['green']:>2} "
                            f"conf={p['mean_confidence']:.2f}\n")
            except Exception:
                pass

        # Channel pattern summary
        flagged = [r for r in self.results
                   if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if flagged:
            self._report_append(f"\n{'='*60}\n")
            self._report_append("  CHANNEL PATTERNS (flagged submissions)\n")
            self._report_append(f"{'='*60}\n")
            channel_counts = Counter()
            for r in flagged:
                cd = r.get('channel_details', {}).get('channels', {})
                for ch_name, info in cd.items():
                    if info.get('severity') not in ('GREEN', None):
                        channel_counts[ch_name] += 1
            for ch, ct in channel_counts.most_common():
                self._report_append(f"  {ch:20s}: {ct} flags\n")

        # Financial impact
        if len(self.results) >= 10:
            try:
                from llm_detector.reporting import financial_impact
                cost = float(self.cost_var.get() or 400)
                impact = financial_impact(self.results, cost_per_prompt=cost)
                self._report_append(f"\n{'='*60}\n")
                self._report_append("  FINANCIAL IMPACT ESTIMATE\n")
                self._report_append(f"{'='*60}\n")
                self._report_append(
                    f"  Total submissions:     {impact['total_submissions']}\n")
                self._report_append(
                    f"  Flag rate:             {impact['flag_rate']:.1%}\n")
                self._report_append(
                    f"  Waste estimate:        ${impact['waste_estimate']:,.0f}\n")
                self._report_append(
                    f"  Projected annual:      ${impact['projected_annual_waste']:,.0f}\n")
                self._report_append(
                    f"  Savings (60% catch):   ${impact['projected_annual_savings_60pct']:,.0f}\n")
            except Exception:
                pass

    def _generate_html_reports(self):
        flagged = [r for r in self.results
                   if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if not flagged:
            messagebox.showinfo('HTML Reports', 'No flagged results to report.')
            return
        output_dir = filedialog.askdirectory(title='Choose HTML output directory')
        if not output_dir:
            return
        try:
            from llm_detector.html_report import generate_html_report
            os.makedirs(output_dir, exist_ok=True)
            for r in flagged:
                tid = r.get('task_id', 'unknown')[:20].replace('/', '_')
                path = os.path.join(output_dir,
                                    f"{tid}_{r['determination']}.html")
                text = self.text_map.get(r.get('task_id', ''), '')
                generate_html_report(text, r, path)
            messagebox.showinfo(
                'HTML Reports',
                f'{len(flagged)} reports saved to {output_dir}')
        except Exception as e:
            messagebox.showerror('HTML Report Error', str(e))

    def _show_attempter_history(self):
        name = self.hist_name_var.get().strip()
        if not name:
            messagebox.showinfo('History', 'Enter an attempter name.')
            return
        if not self.memory_store:
            messagebox.showinfo('History',
                                'Connect a memory store in Settings first.')
            return
        try:
            history = self.memory_store.get_attempter_history(name)
            self.report_output.delete('1.0', tk.END)
            if history['profile'] is None:
                self._report_append(f"No profile found for '{name}'\n")
                return
            p = history['profile']
            self._report_append(f"{'='*60}\n")
            self._report_append(f"  ATTEMPTER: {name}\n")
            self._report_append(f"{'='*60}\n")
            self._report_append(f"  Total submissions:  {p.get('total_submissions', 0)}\n")
            self._report_append(f"  Flag rate:          {p.get('flag_rate', 0):.1%}\n")
            self._report_append(f"  Risk tier:          {p.get('risk_tier', '?')}\n")
            self._report_append(f"  Confirmed AI:       {p.get('confirmed_ai', 0)}\n")
            self._report_append(f"  Confirmed human:    {p.get('confirmed_human', 0)}\n")
            self._report_append(f"  First seen:         {p.get('first_seen', '?')}\n")
            self._report_append(f"  Last seen:          {p.get('last_seen', '?')}\n")
            shadow_flags = p.get('shadow_model_flags', 0)
            if shadow_flags:
                self._report_append(f"  Shadow model flags: {shadow_flags}\n")
            self._report_append(f"\n  Submissions: {len(history['submissions'])}\n")
            for s in history['submissions'][:20]:
                self._report_append(
                    f"    {s.get('task_id', '')[:15]:15s} "
                    f"{s.get('determination', '?'):6s} "
                    f"conf={s.get('confidence', 0):.2f} "
                    f"{s.get('batch_id', '')[:15]}\n")
        except Exception as e:
            self._report_append(f"Error: {e}\n")


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()
