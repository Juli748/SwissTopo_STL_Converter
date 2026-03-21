import json
import shutil
import subprocess
import sys
import threading
import queue
import struct
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from defaults import DEFAULTS


GEOMETRY_DATA_DIRNAME = "geometry_data"


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 16
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.configure(bg="#0b1220")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#0b1220",
            foreground="#e2e8f0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
            wraplength=360,
        )
        label.pack()
        tw.update_idletasks()
        screen_w = tw.winfo_screenwidth()
        screen_h = tw.winfo_screenheight()
        tip_w = tw.winfo_width()
        tip_h = tw.winfo_height()
        x = min(max(0, x), max(0, screen_w - tip_w - 10))
        y = min(max(0, y), max(0, screen_h - tip_h - 10))
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(DEFAULTS["window_title"])
        self.geometry(DEFAULTS["window_geometry"])
        self.configure(bg="#0f172a")
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        target_w = min(900, max(720, screen_w - 80))
        target_h = min(860, max(720, screen_h - 120))
        self.geometry(f"{target_w}x{target_h}")
        self.minsize(target_w, target_h)

        self.data_dir = Path(__file__).resolve().parent / "data"
        self.xyz_dir = self.data_dir / "xyz"
        self.output_dir = Path(__file__).resolve().parent / "output"
        self.tiles_dir = self.output_dir / "tiles"
        self.geometry_data_dir = Path(__file__).resolve().parent / GEOMETRY_DATA_DIRNAME

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.xyz_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.current_process: subprocess.Popen[str] | None = None
        self.tooltips: list[Tooltip] = []
        self.status_vars: dict[str, tk.StringVar] = {}
        self.status_labels: dict[str, ttk.Label] = {}
        self.current_status_key: str | None = None
        self.model_name_var = tk.StringVar(value=DEFAULTS["model_name"])
        self._auto_merge_out = ""
        self.download_progress_var = tk.DoubleVar(value=0.0)
        self.download_progress_label_var = tk.StringVar(value="")
        self.convert_progress_var = tk.DoubleVar(value=0.0)
        self.convert_progress_label_var = tk.StringVar(value="")
        self.merge_progress_var = tk.DoubleVar(value=0.0)
        self.merge_progress_label_var = tk.StringVar(value="")
        self.pipeline_status_var = tk.StringVar(value="")
        self.input_summary_var = tk.StringVar(value="")
        self.tiles_summary_var = tk.StringVar(value="")
        self.output_summary_var = tk.StringVar(value="")
        self.detail_summary_var = tk.StringVar(value="")
        self.merge_summary_var = tk.StringVar(value="")
        self.merge_border_mode_var = tk.StringVar(value=DEFAULTS["merge_border_mode"])
        self.border_scale_var = tk.StringVar(value=DEFAULTS["border_scale"])
        self.detail_preset_var = tk.StringVar(value=DEFAULTS["detail_preset"])
        self.show_convert_advanced_var = tk.BooleanVar(value=DEFAULTS["show_convert_advanced"])
        self.show_merge_advanced_var = tk.BooleanVar(value=DEFAULTS["show_merge_advanced"])
        self.border_options = self._discover_border_shp()
        default_border_label = self._select_default_border_label()
        self.border_shp_var = tk.StringVar(value=default_border_label)
        self.border_keep_var = tk.StringVar(value="")
        self.border_keep_options: list[str] = []
        self.border_keep_all_label = "(all touched)"
        self.pending_commands: list[tuple[list[str], str, str]] = []
        self.pipeline_running = False

        self._setup_theme()
        self._build_ui()
        self._load_default_csv()
        self._refresh_pipeline_summary()
        self._poll_log()

    def _setup_theme(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background="#0f172a")
        style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 12, "bold"))
        style.configure(
            "TButton",
            background="#2563eb",
            foreground="#f8fafc",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
        )
        style.map("TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "TEntry",
            fieldbackground="#0b1220",
            foreground="#e2e8f0",
            background="#0b1220",
            padding=(6, 4),
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", "#0b1220")],
            foreground=[("disabled", "#6b7280")],
        )
        style.configure(
            "TLabelframe",
            background="#0f172a",
            foreground="#e2e8f0",
            padding=10,
        )
        style.configure(
            "TLabelframe.Label",
            background="#0f172a",
            foreground="#f8fafc",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure("TCheckbutton", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("TRadiobutton", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("TSeparator", background="#1f2937")

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(root, bg="#0f172a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=main, anchor="nw")

        def _sync_scrollregion(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        main.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)
        self._scroll_canvas = canvas
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text=DEFAULTS["window_title"], style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Pick a CSV, choose a print size, then build a print-ready STL.",
        ).pack(anchor=tk.W)
        name_row = ttk.Frame(header)
        name_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(name_row, text="Model name:").pack(side=tk.LEFT)
        self.model_name_entry = ttk.Entry(name_row, textvariable=self.model_name_var, width=24)
        self.model_name_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.model_name_entry.bind("<KeyRelease>", self._on_model_name_change)
        ttk.Button(name_row, text="Refresh Status", command=self._refresh_pipeline_summary).pack(side=tk.RIGHT)
        self.tooltips.append(
            Tooltip(
                self.model_name_entry,
                "Used to name output STL files and the STL solid name.",
            )
        )

        summary = ttk.LabelFrame(main, text="Pipeline Overview")
        summary.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(summary, textvariable=self.pipeline_status_var, style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4)
        )
        ttk.Label(summary, text="Inputs").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(summary, textvariable=self.input_summary_var).grid(row=2, column=0, sticky=tk.W, padx=(0, 16))
        ttk.Label(summary, text="STL tiles").grid(row=1, column=1, sticky=tk.W)
        ttk.Label(summary, textvariable=self.tiles_summary_var).grid(row=2, column=1, sticky=tk.W, padx=(0, 16))
        ttk.Label(summary, text="Final STL").grid(row=1, column=2, sticky=tk.W)
        ttk.Label(summary, textvariable=self.output_summary_var).grid(row=2, column=2, sticky=tk.W)
        ttk.Button(summary, text="Run Full Pipeline", command=self._run_full_pipeline).grid(
            row=0, column=2, sticky=tk.E
        )
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        summary.columnconfigure(2, weight=1)

        steps = ttk.Frame(main)
        steps.pack(fill=tk.X)

        self._build_step_download(steps)
        ttk.Separator(steps, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self._build_step_convert(steps)
        ttk.Separator(steps, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self._build_step_merge(steps)

        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        log_frame = ttk.Frame(main)
        log_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Log", style="Header.TLabel").pack(anchor=tk.W)
        self.log_text = tk.Text(
            log_frame,
            height=16,
            wrap=tk.WORD,
            bg="#0b1220",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief=tk.FLAT,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=8)
        ttk.Button(buttons, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Open Output Folder", command=self._open_output_folder).pack(side=tk.RIGHT)

    def _on_mousewheel(self, event: tk.Event) -> None:
        canvas = getattr(self, "_scroll_canvas", None)
        if canvas is None:
            return
        target = canvas.winfo_containing(event.x_root, event.y_root)
        if target is None:
            return
        if isinstance(target, tk.Listbox):
            return
        widget = target
        inside_canvas = False
        while widget is not None:
            if widget is canvas:
                inside_canvas = True
                break
            widget = widget.master
        if not inside_canvas:
            return
        if hasattr(event, "delta") and event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")

    def _on_listbox_mousewheel(self, event: tk.Event) -> str:
        listbox = self.border_keep_list
        if listbox is None:
            return "break"
        if hasattr(event, "delta") and event.delta:
            listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            listbox.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            listbox.yview_scroll(3, "units")
        return "break"

    def _build_step_download(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1) Download XYZ/TIF tiles")
        frame.pack(fill=tk.X)

        csv_label = ttk.Label(frame, text="CSV with download URLs:")
        csv_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.csv_path_var = tk.StringVar(value=DEFAULTS["csv_path"])
        entry = ttk.Entry(frame, textvariable=self.csv_path_var)
        entry.grid(row=0, column=1, sticky=tk.EW, padx=6)
        browse_btn = ttk.Button(frame, text="Browse", command=self._browse_csv)
        browse_btn.grid(row=0, column=2, padx=4, pady=2)
        copy_btn = ttk.Button(frame, text="Copy to data/", command=self._copy_csv)
        copy_btn.grid(row=0, column=3, padx=4, pady=2)

        download_note = ttk.Label(
            frame,
            text="Download into data/xyz and data/tif using download_tiles.py",
        )
        workers_label = ttk.Label(frame, text="Max parallel downloads:")
        workers_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.download_workers_var = tk.StringVar(value=DEFAULTS["download_workers"])
        workers_entry = ttk.Entry(frame, textvariable=self.download_workers_var, width=8)
        workers_entry.grid(row=1, column=1, sticky=tk.W)

        download_note.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.download_btn = ttk.Button(frame, text="Run Download", command=self._run_download)
        self.download_btn.grid(row=2, column=3, padx=4, pady=6)
        self.status_vars["download"] = tk.StringVar(value=DEFAULTS["status_idle"])
        self.status_labels["download"] = ttk.Label(frame, textvariable=self.status_vars["download"])
        self.status_labels["download"].grid(
            row=2, column=4, sticky=tk.W, padx=(8, 0)
        )
        self.status_labels["download"].configure(width=12)
        self.download_progress = ttk.Progressbar(
            frame,
            variable=self.download_progress_var,
            maximum=100.0,
            mode="determinate",
            length=220,
        )
        self.download_progress.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.download_progress_label = ttk.Label(frame, textvariable=self.download_progress_label_var)
        self.download_progress_label.grid(row=3, column=3, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.download_progress_label.configure(width=26)

        frame.columnconfigure(0, minsize=200)
        frame.columnconfigure(1, weight=1, minsize=220)
        frame.columnconfigure(2, minsize=140)
        frame.columnconfigure(3, minsize=120)
        frame.columnconfigure(4, minsize=140)

        self.tooltips += [
            Tooltip(
                csv_label,
                "CSV file with one download URL per line. The tool downloads ZIPs and extracts XYZ files.",
            ),
            Tooltip(
                entry,
                "Path to the URLs CSV file used by download_tiles.py.",
            ),
            Tooltip(
                browse_btn,
                "Pick a CSV file from disk.",
            ),
            Tooltip(
                copy_btn,
                "Copies the CSV into ./data so it is picked up automatically next time.",
            ),
            Tooltip(
                workers_label,
                "How many downloads to run in parallel. Higher is faster but uses more bandwidth.",
            ),
            Tooltip(
                workers_entry,
                "Try 4-8 for faster downloads. Set to 1 for sequential.",
            ),
            Tooltip(
                self.download_btn,
                "Runs download_tiles.py to fetch tiles into ./data/xyz and ./data/tif.",
            ),
            Tooltip(
                download_note,
                "Tiles are stored under ./data/xyz as .xyz files and ./data/tif as GeoTIFF.",
            ),
        ]

    def _build_step_convert(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2) Convert XYZ/TIF to STL tiles")
        frame.pack(fill=tk.X)

        self.mode_var = tk.StringVar(value=DEFAULTS["mode"])
        self.scale_mode_var = tk.StringVar(value=DEFAULTS["scale_mode"])
        self.target_size_var = tk.StringVar(value=DEFAULTS["target_size_mm"])
        self.target_res_var = tk.StringVar(value=DEFAULTS["target_resolution_mm"])
        self.target_edge_var = tk.StringVar(value=DEFAULTS["target_edge"])
        self.tile_size_var = tk.StringVar(value=DEFAULTS["tile_size_mm"])
        self.scale_ratio_var = tk.StringVar(value=DEFAULTS["scale_ratio"])
        self.step_var = tk.StringVar(value=DEFAULTS["step"])
        self.tol_var = tk.StringVar(value=DEFAULTS["grid_tolerance"])
        self.z_scale_var = tk.StringVar(value=DEFAULTS["z_scale"])
        intro = ttk.Label(
            frame,
            text="Default path: pick a final print size and a detail preset. Advanced controls stay hidden until needed.",
        )
        intro.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        basic = ttk.Frame(frame)
        basic.grid(row=1, column=0, columnspan=4, sticky=tk.EW)

        self.target_size_label = ttk.Label(basic, text="Final model size (mm):")
        self.target_size_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.target_size_entry = ttk.Entry(basic, textvariable=self.target_size_var, width=10)
        self.target_size_entry.grid(row=0, column=1, sticky=tk.W)

        self.target_edge_label = ttk.Label(basic, text="Fit this edge:")
        self.target_edge_label.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
        self.target_edge_combo = ttk.Combobox(
            basic,
            textvariable=self.target_edge_var,
            values=["shortest", "longest"],
            state="readonly",
            width=10,
        )
        self.target_edge_combo.grid(row=0, column=3, sticky=tk.W)

        detail_label = ttk.Label(basic, text="Detail preset:")
        detail_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        detail_combo = ttk.Combobox(
            basic,
            textvariable=self.detail_preset_var,
            values=["draft", "balanced", "fine", "custom"],
            state="readonly",
            width=12,
        )
        detail_combo.grid(row=1, column=1, sticky=tk.W)
        detail_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_convert_mode())

        detail_hint = ttk.Label(basic, textvariable=self.detail_summary_var)
        detail_hint.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=(12, 0))

        self.target_res_label = ttk.Label(basic, text="Target XY spacing (mm):")
        self.target_res_label.grid(row=2, column=0, sticky=tk.W, pady=4)
        self.target_res_entry = ttk.Entry(basic, textvariable=self.target_res_var, width=10)
        self.target_res_entry.grid(row=2, column=1, sticky=tk.W)

        self.z_scale_label = ttk.Label(basic, text="Height exaggeration:")
        self.z_scale_label.grid(row=2, column=2, sticky=tk.W, padx=(12, 0))
        z_scale_entry = ttk.Entry(basic, textvariable=self.z_scale_var, width=10)
        z_scale_entry.grid(row=2, column=3, sticky=tk.W)

        basic.columnconfigure(1, weight=1)
        basic.columnconfigure(3, weight=1)

        advanced_toggle = ttk.Checkbutton(
            frame,
            text="Show advanced conversion settings",
            variable=self.show_convert_advanced_var,
            command=self._update_convert_mode,
        )
        advanced_toggle.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 4))

        self.convert_advanced_frame = ttk.Frame(frame)
        self.convert_advanced_frame.grid(row=3, column=0, columnspan=4, sticky=tk.EW)

        mode_label = ttk.Label(self.convert_advanced_frame, text="Detail control:")
        mode_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        auto_radio = ttk.Radiobutton(
            self.convert_advanced_frame,
            text="Auto",
            value="auto",
            variable=self.mode_var,
            command=self._update_convert_mode,
        )
        auto_radio.grid(row=0, column=1, sticky=tk.W)
        manual_radio = ttk.Radiobutton(
            self.convert_advanced_frame,
            text="Manual step",
            value="manual",
            variable=self.mode_var,
            command=self._update_convert_mode,
        )
        manual_radio.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        scale_mode_label = ttk.Label(self.convert_advanced_frame, text="Scale source:")
        scale_mode_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        scale_target_radio = ttk.Radiobutton(
            self.convert_advanced_frame,
            text="Final model size",
            value="target_size",
            variable=self.scale_mode_var,
            command=self._update_convert_mode,
        )
        scale_target_radio.grid(row=1, column=1, sticky=tk.W)
        scale_tile_radio = ttk.Radiobutton(
            self.convert_advanced_frame,
            text="Fixed 1 km tile size",
            value="tile_size",
            variable=self.scale_mode_var,
            command=self._update_convert_mode,
        )
        scale_tile_radio.grid(row=1, column=2, sticky=tk.W, padx=(12, 0))
        scale_ratio_radio = ttk.Radiobutton(
            self.convert_advanced_frame,
            text="Scale ratio",
            value="scale_ratio",
            variable=self.scale_mode_var,
            command=self._update_convert_mode,
        )
        scale_ratio_radio.grid(row=1, column=3, sticky=tk.W)

        self.step_label = ttk.Label(self.convert_advanced_frame, text="Downsample step:")
        self.step_label.grid(row=2, column=0, sticky=tk.W, pady=4)
        self.step_entry = ttk.Entry(self.convert_advanced_frame, textvariable=self.step_var, width=10)
        self.step_entry.grid(row=2, column=1, sticky=tk.W)

        self.tile_size_label = ttk.Label(self.convert_advanced_frame, text="Tile size (mm for 1 km):")
        self.tile_size_label.grid(row=2, column=2, sticky=tk.W, padx=(12, 0))
        self.tile_size_entry = ttk.Entry(self.convert_advanced_frame, textvariable=self.tile_size_var, width=10)
        self.tile_size_entry.grid(row=2, column=3, sticky=tk.W)

        self.scale_ratio_label = ttk.Label(self.convert_advanced_frame, text="Scale ratio (e.g. 1:100):")
        self.scale_ratio_label.grid(row=3, column=0, sticky=tk.W, pady=4)
        self.scale_ratio_entry = ttk.Entry(self.convert_advanced_frame, textvariable=self.scale_ratio_var, width=10)
        self.scale_ratio_entry.grid(row=3, column=1, sticky=tk.W)

        self.tol_label = ttk.Label(self.convert_advanced_frame, text="Grid tolerance:")
        self.tol_label.grid(row=3, column=2, sticky=tk.W, padx=(12, 0))
        tol_entry = ttk.Entry(self.convert_advanced_frame, textvariable=self.tol_var, width=10)
        tol_entry.grid(row=3, column=3, sticky=tk.W)

        convert_workers_label = ttk.Label(self.convert_advanced_frame, text="Max parallel conversions:")
        convert_workers_label.grid(row=4, column=0, sticky=tk.W, pady=4)
        self.convert_workers_var = tk.StringVar(value=DEFAULTS["convert_workers"])
        convert_workers_entry = ttk.Entry(self.convert_advanced_frame, textvariable=self.convert_workers_var, width=10)
        convert_workers_entry.grid(row=4, column=1, sticky=tk.W)

        self.convert_btn = ttk.Button(frame, text="Create STL Tiles", command=self._run_convert)
        self.convert_btn.grid(row=4, column=3, sticky=tk.E, padx=4, pady=6)
        self.status_vars["convert"] = tk.StringVar(value=DEFAULTS["status_idle"])
        self.status_labels["convert"] = ttk.Label(frame, textvariable=self.status_vars["convert"])
        self.status_labels["convert"].grid(row=4, column=2, sticky=tk.W, padx=(12, 0))
        self.convert_progress = ttk.Progressbar(
            frame,
            variable=self.convert_progress_var,
            maximum=100.0,
            mode="determinate",
            length=220,
        )
        self.convert_progress.grid(row=5, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.convert_progress_label = ttk.Label(frame, textvariable=self.convert_progress_label_var)
        self.convert_progress_label.grid(row=5, column=3, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.convert_progress_label.configure(width=26)

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        self._update_convert_mode()

        self.tooltips += [
            Tooltip(
                self.target_size_entry,
                "The chosen edge of the merged model will be scaled to about this size in millimeters.",
            ),
            Tooltip(
                detail_combo,
                "Draft is fastest, Balanced is the default, Fine keeps more terrain detail, Custom unlocks manual control.",
            ),
            Tooltip(
                self.target_res_entry,
                "Smaller spacing keeps more points and increases STL size. Ignored when using a manual step.",
            ),
            Tooltip(
                z_scale_entry,
                "Multiply terrain heights during tile conversion. Leave at 1.0 for true scale.",
            ),
            Tooltip(
                advanced_toggle,
                "Show manual step, alternative scale modes, tolerance, and worker settings.",
            ),
            Tooltip(
                self.convert_btn,
                "Runs build_stl.py --all with the simplified settings above.",
            ),
        ]

    def _build_step_merge(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3) Merge tiles into final STL")
        frame.pack(fill=tk.X)

        self._auto_merge_out = str(self.output_dir / DEFAULTS["merge_out"])
        self.merge_out_var = tk.StringVar(value=self._auto_merge_out)
        self.weld_tol_var = tk.StringVar(value=DEFAULTS["weld_tolerance"])
        self.merge_z_scale_var = tk.StringVar(value=DEFAULTS["merge_z_scale"])
        self.merge_lake_lower_mm_var = tk.StringVar(value=DEFAULTS["merge_lake_lower_mm"])
        self.make_solid_var = tk.BooleanVar(value=DEFAULTS["make_solid"])
        self.base_mode_var = tk.StringVar(value=DEFAULTS["base_mode"])
        self.base_thickness_var = tk.StringVar(value=DEFAULTS["base_thickness"])
        self.base_z_var = tk.StringVar(value=DEFAULTS["base_z"])
        intro = ttk.Label(
            frame,
            text="Default path: merge all converted tiles into one printable STL with a base.",
        )
        intro.grid(row=0, column=0, sticky=tk.W, pady=(0, 6))

        path_row = ttk.Frame(frame)
        path_row.grid(row=1, column=0, sticky=tk.EW, pady=(0, 4))
        merge_out_label = ttk.Label(path_row, text="Final STL path:")
        merge_out_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        merge_out_entry = ttk.Entry(path_row, textvariable=self.merge_out_var)
        merge_out_entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
        merge_out_btn = ttk.Button(path_row, text="Browse", command=self._browse_merge_out)
        merge_out_btn.grid(row=0, column=2, pady=2)
        path_row.columnconfigure(1, weight=1)

        base_row = ttk.Frame(frame)
        base_row.grid(row=2, column=0, sticky=tk.EW, pady=(0, 4))
        make_solid_chk = ttk.Checkbutton(
            base_row,
            text="Add printable base",
            variable=self.make_solid_var,
            command=self._update_merge_controls,
        )
        make_solid_chk.grid(row=0, column=0, sticky=tk.W, pady=4)

        self.base_mode_label = ttk.Label(base_row, text="Base mode:")
        self.base_mode_label.grid(row=0, column=1, sticky=tk.W, padx=(16, 0), pady=2)
        self.base_mode_combo = ttk.Combobox(
            base_row,
            textvariable=self.base_mode_var,
            values=["fixed", "sealevel"],
            state="readonly",
            width=10,
        )
        self.base_mode_combo.grid(row=0, column=2, sticky=tk.W, padx=(8, 16))
        self.base_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_merge_controls())

        self.base_thickness_label = ttk.Label(base_row, text="Fixed base depth (mm):")
        self.base_thickness_label.grid(row=0, column=3, sticky=tk.W, pady=4)
        self.base_thickness_entry = ttk.Entry(base_row, textvariable=self.base_thickness_var, width=10)
        self.base_thickness_entry.grid(row=0, column=4, sticky=tk.W, padx=(8, 16))
        base_row.columnconfigure(5, weight=1)

        clip_row = ttk.Frame(frame)
        clip_row.grid(row=3, column=0, sticky=tk.EW, pady=(0, 4))
        self.merge_lake_lower_label = ttk.Label(clip_row, text="Lake lowering (mm):")
        self.merge_lake_lower_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.merge_lake_lower_entry = ttk.Entry(clip_row, textvariable=self.merge_lake_lower_mm_var, width=10)
        self.merge_lake_lower_entry.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))

        border_mode_row = ttk.Frame(frame)
        border_mode_row.grid(row=4, column=0, sticky=tk.EW, pady=(0, 4))

        border_mode_label = ttk.Label(border_mode_row, text="Border clipping:")
        border_mode_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        border_all_radio = ttk.Radiobutton(
            border_mode_row,
            text="Merge all tiles",
            value="all",
            variable=self.merge_border_mode_var,
            command=self._update_merge_controls,
        )
        border_all_radio.grid(row=0, column=3, sticky=tk.W, padx=(8, 12))
        border_clip_radio = ttk.Radiobutton(
            border_mode_row,
            text="Clip to canton/border",
            value="clip",
            variable=self.merge_border_mode_var,
            command=self._update_merge_controls,
        )
        border_clip_radio.grid(row=0, column=4, sticky=tk.W)
        border_mode_row.columnconfigure(5, weight=1)

        border_row = ttk.Frame(frame)
        border_row.grid(row=5, column=0, sticky=tk.EW, pady=(0, 4))
        self.border_shp_label = ttk.Label(border_row, text="Border shapefile:")
        self.border_shp_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.border_shp_combo = ttk.Combobox(
            border_row,
            textvariable=self.border_shp_var,
            values=sorted(self.border_options.keys()),
            state="readonly",
            width=30,
        )
        self.border_shp_combo.grid(row=0, column=1, sticky=tk.EW, padx=(8, 12))
        self.border_shp_combo.bind("<<ComboboxSelected>>", self._on_border_shp_change)

        self.border_keep_label = ttk.Label(border_row, text="Keep canton/bezirk:")
        self.border_keep_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        self.border_keep_list = tk.Listbox(
            border_row,
            selectmode=tk.MULTIPLE,
            height=5,
            exportselection=False,
            bg="#0b1220",
            fg="#e2e8f0",
            highlightthickness=1,
            highlightbackground="#1f2937",
            selectbackground="#2563eb",
            selectforeground="#f8fafc",
        )
        self.border_keep_list.grid(row=0, column=3, sticky=tk.EW, padx=(8, 12))
        self.border_keep_list.bind("<MouseWheel>", self._on_listbox_mousewheel)
        self.border_keep_list.bind("<Button-4>", self._on_listbox_mousewheel)
        self.border_keep_list.bind("<Button-5>", self._on_listbox_mousewheel)
        self.border_keep_list.bind("<<ListboxSelect>>", self._on_border_keep_select)
        self.border_keep_refresh_btn = ttk.Button(
            border_row,
            text="Detect touched",
            command=self._refresh_border_keep_options,
        )
        self.border_keep_refresh_btn.grid(row=0, column=4, pady=2, sticky=tk.NW)
        border_row.columnconfigure(1, weight=1)
        border_row.columnconfigure(3, weight=1)

        merge_summary = ttk.Label(frame, textvariable=self.merge_summary_var)
        merge_summary.grid(row=6, column=0, sticky=tk.W, pady=(0, 4))

        advanced_toggle = ttk.Checkbutton(
            frame,
            text="Show advanced merge settings",
            variable=self.show_merge_advanced_var,
            command=self._update_merge_controls,
        )
        advanced_toggle.grid(row=7, column=0, sticky=tk.W, pady=(4, 4))

        self.merge_advanced_frame = ttk.Frame(frame)
        self.merge_advanced_frame.grid(row=8, column=0, sticky=tk.EW)

        weld_label = ttk.Label(self.merge_advanced_frame, text="Weld tolerance:")
        weld_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        weld_entry = ttk.Entry(self.merge_advanced_frame, textvariable=self.weld_tol_var, width=10)
        weld_entry.grid(row=0, column=1, sticky=tk.W)

        merge_z_label = ttk.Label(self.merge_advanced_frame, text="Merge Z scale:")
        merge_z_label.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
        merge_z_entry = ttk.Entry(self.merge_advanced_frame, textvariable=self.merge_z_scale_var, width=10)
        merge_z_entry.grid(row=0, column=3, sticky=tk.W)

        action_row = ttk.Frame(frame)
        action_row.grid(row=9, column=0, sticky=tk.EW, pady=(6, 0))
        self.status_vars["merge"] = tk.StringVar(value=DEFAULTS["status_idle"])
        self.status_labels["merge"] = ttk.Label(action_row, textvariable=self.status_vars["merge"])
        self.status_labels["merge"].grid(row=0, column=0, sticky=tk.W)
        self.status_labels["merge"].configure(width=12)
        self.merge_btn = ttk.Button(action_row, text="Build Final STL", command=self._run_merge)
        self.merge_btn.grid(row=0, column=1, sticky=tk.E)
        action_row.columnconfigure(0, weight=1)

        self.merge_progress = ttk.Progressbar(
            frame,
            variable=self.merge_progress_var,
            maximum=100.0,
            mode="determinate",
            length=220,
        )
        self.merge_progress.grid(row=10, column=0, sticky=tk.W, pady=(4, 0))
        self.merge_progress_label = ttk.Label(frame, textvariable=self.merge_progress_label_var)
        self.merge_progress_label.grid(row=10, column=0, sticky=tk.E, pady=(4, 0))
        self.merge_progress_label.configure(width=26)

        frame.columnconfigure(0, weight=1)
        self.merge_advanced_frame.columnconfigure(2, weight=1)

        self.tooltips += [
            Tooltip(
                merge_out_entry,
                "The merged terrain STL will be written here.",
            ),
            Tooltip(
                make_solid_chk,
                "Adds side walls and a flat base so the result is ready to print.",
            ),
            Tooltip(
                self.base_mode_combo,
                "Choose a fixed base depth in millimeters or align the base to sea level (Z=0).",
            ),
            Tooltip(
                self.base_thickness_entry,
                "Used when base mode is fixed. Sets how far below the terrain the printable base should sit.",
            ),
            Tooltip(
                self.merge_lake_lower_entry,
                "Lower detected lake surfaces by this many millimeters in the final model.",
            ),
            Tooltip(
                self.border_shp_combo,
                "Choose the Swiss border, canton, or bezirk shapefile used for clipping.",
            ),
            Tooltip(
                self.border_keep_list,
                "When using canton or bezirk data, choose which touched regions to keep.",
            ),
            Tooltip(
                advanced_toggle,
                "Show weld tolerance and merge Z scale overrides.",
            ),
            Tooltip(
                self.merge_btn,
                "Runs build_stl.py --merge-stl with the selected settings.",
            ),
        ]

        self._update_merge_controls()

    def _format_border_label(self, path: Path) -> str:
        name = path.stem.replace("_", " ").strip()
        if not name:
            return path.name
        return name.title()

    def _geometry_search_roots(self) -> list[Path]:
        roots = [self.geometry_data_dir, Path(__file__).resolve().parent / "borders"]
        seen: set[str] = set()
        unique: list[Path] = []
        for path in roots:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _is_border_shapefile(self, path: Path) -> bool:
        name = path.name.upper()
        return any(token in name for token in ("LANDES", "KANTON", "BEZIRK"))

    def _discover_border_shp(self) -> dict[str, Path]:
        options: dict[str, Path] = {}
        for root in self._geometry_search_roots():
            if not root.exists():
                continue
            for shp in sorted(root.rglob("*.shp")):
                if not shp.is_file() or not self._is_border_shapefile(shp):
                    continue
                label = self._format_border_label(shp)
                if label in options:
                    label = f"{label} ({shp.name})"
                options[label] = shp
        return options

    def _select_default_border_label(self) -> str:
        default_raw = DEFAULTS.get("border_shp", "").strip()
        if default_raw:
            default_path = Path(default_raw)
            if default_path.exists():
                for label, path in self.border_options.items():
                    if path.resolve() == default_path.resolve():
                        return label
                return self._format_border_label(default_path)
        if not self.border_options:
            return ""
        preferred = [label for label in self.border_options if "LANDES" in label.upper()]
        return sorted(preferred or list(self.border_options.keys()))[0]

    def _border_type_from_path(self, path: Path) -> str | None:
        name = path.name.upper()
        if "KANTON" in name:
            return "canton"
        if "BEZIRK" in name:
            return "bezirk"
        try:
            import shapefile  # pyshp
        except Exception:
            return None

        try:
            reader = shapefile.Reader(str(path))
        except Exception:
            return None

        field_defs = [f for f in reader.fields if f[0] != "DeletionFlag"]
        field_names = {name.upper() for name, _ftype, _len, _dec in field_defs}
        if {"KANTONSNAME", "KANTON"} & field_names:
            return "canton"
        if {"BEZIRKSNAME", "BEZIRK"} & field_names:
            return "bezirk"
        return None

    def _read_last_scale_info(self) -> float | None:
        scale_info = self.tiles_dir / "scale_info.json"
        if not scale_info.exists():
            return None
        try:
            data = json.loads(scale_info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return float(data.get("scale_xy"))
        except (TypeError, ValueError):
            return None

    def _parse_scale_ratio(self, raw: str) -> float:
        s = raw.strip()
        if not s:
            raise ValueError("Scale ratio is empty.")
        if ":" in s:
            parts = s.split(":")
        elif "/" in s:
            parts = s.split("/")
        else:
            parts = [s]
        if len(parts) == 1:
            ratio = float(parts[0])
        elif len(parts) == 2:
            a = float(parts[0])
            b = float(parts[1])
            if a == 0:
                raise ValueError("Scale ratio numerator cannot be zero.")
            ratio = b / a
        else:
            raise ValueError("Invalid scale ratio format.")
        if ratio <= 0:
            raise ValueError("Scale ratio must be > 0.")
        return ratio

    def _parse_border_scale(self, raw: str) -> float:
        value = raw.strip().lower()
        if value in {"", "auto"}:
            scale = self._read_last_scale_info()
            if scale is None:
                return 1.0
            return float(scale)
        if ":" in value or "/" in value:
            ratio = self._parse_scale_ratio(value)
            return 1000.0 / float(ratio)
        return float(value)

    def _iter_stl_vertices(self, path: Path):
        try:
            file_size = path.stat().st_size
        except OSError:
            return
        try:
            with path.open("rb") as handle:
                header = handle.read(80)
                count_bytes = handle.read(4)
                if len(count_bytes) == 4:
                    tri_count = struct.unpack("<I", count_bytes)[0]
                    expected = 84 + tri_count * 50
                    if file_size == expected:
                        for _ in range(tri_count):
                            data = handle.read(50)
                            if len(data) < 50:
                                break
                            vals = struct.unpack("<12f", data[:48])
                            yield (vals[3], vals[4], vals[5])
                            yield (vals[6], vals[7], vals[8])
                            yield (vals[9], vals[10], vals[11])
                        return
        except OSError:
            return

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    s = line.strip()
                    if s.lower().startswith("vertex"):
                        parts = s.split()
                        if len(parts) >= 4:
                            try:
                                yield (float(parts[1]), float(parts[2]), float(parts[3]))
                            except ValueError:
                                continue
        except OSError:
            return

    def _stl_bounds(self, path: Path) -> tuple[float, float, float, float] | None:
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        found = False
        for x, y, _z in self._iter_stl_vertices(path):
            found = True
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        if not found:
            return None
        return min_x, min_y, max_x, max_y

    def _collect_tile_union(self):
        try:
            from shapely.geometry import box
            from shapely.ops import unary_union
        except Exception as exc:
            raise RuntimeError(
                "Border selection requires 'shapely'. Install with: pip install shapely"
            ) from exc

        stl_files = sorted(self.tiles_dir.glob("*.stl"))
        if not stl_files:
            return None
        boxes = []
        for path in stl_files:
            bounds = self._stl_bounds(path)
            if bounds is None:
                continue
            min_x, min_y, max_x, max_y = bounds
            boxes.append(box(min_x, min_y, max_x, max_y))
        if not boxes:
            return None
        return unary_union(boxes)

    def _guess_border_label_field(self, field_defs, border_type: str) -> str | None:
        field_names = [name for name, _ftype, _len, _dec in field_defs]
        string_fields = [name for name, ftype, _len, _dec in field_defs if ftype in {"C", "M"}]
        upper_names = {name.upper(): name for name in field_names}

        if border_type == "bezirk":
            preferred = [
                "BEZIRKSNAME",
                "BEZIRK",
                "NAME",
                "NAME_DE",
                "NAME_FR",
                "NAME_IT",
                "NAME_EN",
            ]
        elif border_type == "canton":
            preferred = [
                "KANTONSNAME",
                "KANTON",
                "NAME",
                "NAME_DE",
                "NAME_FR",
                "NAME_IT",
                "NAME_EN",
            ]
        else:
            preferred = [
                "NAME",
                "NAME_DE",
                "NAME_FR",
                "NAME_IT",
                "NAME_EN",
            ]

        for candidate in preferred:
            if candidate in upper_names:
                return upper_names[candidate]

        for name in field_names:
            if "NAME" in name.upper():
                return name

        if string_fields:
            return string_fields[0]

        return None

    def _load_border_features(self, shp_path: Path, border_scale: float, border_type: str):
        try:
            import shapefile  # pyshp
            from shapely.geometry import shape as shapely_shape
            from shapely.affinity import scale as shapely_scale
        except Exception as exc:
            raise RuntimeError(
                "Border selection requires 'pyshp' and 'shapely'. Install with: pip install pyshp shapely"
            ) from exc

        reader = shapefile.Reader(str(shp_path))
        field_defs = [f for f in reader.fields if f[0] != "DeletionFlag"]
        field_names = [name for name, _ftype, _len, _dec in field_defs]
        label_field = self._guess_border_label_field(field_defs, border_type)
        if not label_field:
            raise RuntimeError(f"Could not find a label field in {shp_path.name}.")

        features = []
        for shape_record in reader.iterShapeRecords():
            geom = shapely_shape(shape_record.shape.__geo_interface__)
            if geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
                continue
            if border_scale != 1.0:
                geom = shapely_scale(geom, xfact=border_scale, yfact=border_scale, origin=(0.0, 0.0))
            record_dict = dict(zip(field_names, shape_record.record))
            value = record_dict.get(label_field, "")
            label = str(value).strip() if value is not None else ""
            if not label:
                continue
            features.append((label, geom))
        return features

    def _refresh_border_keep_options(self) -> None:
        if self.merge_border_mode_var.get() != "clip":
            return
        border_label = self.border_shp_var.get().strip()
        if not border_label:
            return
        border_path = self.border_options.get(border_label, Path(border_label))
        border_type = self._border_type_from_path(border_path)
        if border_type not in {"canton", "bezirk"}:
            self.border_keep_options = []
            self.border_keep_var.set("")
            self.border_keep_list.delete(0, tk.END)
            return

        try:
            tile_union = self._collect_tile_union()
        except Exception as exc:
            messagebox.showerror("Missing dependency", str(exc))
            return

        if tile_union is None:
            messagebox.showwarning(
                "No tiles found",
                "No STL tiles were found in output/tiles. Convert tiles first.",
            )
            return

        try:
            border_scale = self._parse_border_scale(self.border_scale_var.get())
            features = self._load_border_features(border_path, border_scale, border_type)
        except Exception as exc:
            messagebox.showerror("Border selection failed", str(exc))
            return

        touched = sorted({name for name, geom in features if geom.intersects(tile_union)})
        if not touched:
            messagebox.showwarning(
                "No touched regions",
                "No cantons/bezirk regions intersect the current tiles.",
            )
            self.border_keep_options = []
            self.border_keep_var.set("")
            self.border_keep_list.delete(0, tk.END)
            return

        self.border_keep_options = [self.border_keep_all_label] + touched
        self.border_keep_list.delete(0, tk.END)
        for item in self.border_keep_options:
            self.border_keep_list.insert(tk.END, item)
        self.border_keep_list.selection_set(0)

    def _on_border_shp_change(self, _event: tk.Event) -> None:
        self._refresh_border_keep_options()
        self._update_merge_controls()

    def _on_border_keep_select(self, _event: tk.Event) -> None:
        selections = list(self.border_keep_list.curselection())
        if not selections:
            return
        labels = [self.border_keep_list.get(i) for i in selections]
        if self.border_keep_all_label in labels and len(labels) > 1:
            all_index = self.border_keep_options.index(self.border_keep_all_label)
            active_index = self.border_keep_list.index(tk.ACTIVE)
            if active_index == all_index:
                for idx in selections:
                    if idx != all_index:
                        self.border_keep_list.selection_clear(idx)
            else:
                self.border_keep_list.selection_clear(all_index)

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)
            self._refresh_pipeline_summary()

    def _load_default_csv(self) -> None:
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if csv_files and not self.csv_path_var.get().strip():
            self.csv_path_var.set(str(csv_files[0]))

    def _count_input_files(self) -> tuple[int, int]:
        tif_dir = self.data_dir / "tif"
        xyz_count = len(list(self.xyz_dir.glob("*.xyz")))
        tif_count = len(list(tif_dir.glob("*.tif"))) + len(list(tif_dir.glob("*.tiff")))
        tif_count += len(list(self.data_dir.glob("*.tif"))) + len(list(self.data_dir.glob("*.tiff")))
        return xyz_count, tif_count

    def _count_tiles(self) -> int:
        return len(list(self.tiles_dir.glob("*.stl")))

    def _refresh_pipeline_summary(self) -> None:
        if not hasattr(self, "csv_path_var") or not hasattr(self, "merge_out_var"):
            return
        csv_path = self.csv_path_var.get().strip()
        xyz_count, tif_count = self._count_input_files()
        tile_count = self._count_tiles()
        final_path = Path(self.merge_out_var.get().strip()) if self.merge_out_var.get().strip() else None
        final_exists = bool(final_path and final_path.exists())

        ready_parts = []
        if csv_path:
            ready_parts.append("CSV selected")
        if xyz_count or tif_count:
            ready_parts.append("input tiles ready")
        if tile_count:
            ready_parts.append("STL tiles ready")
        if final_exists:
            ready_parts.append("final STL exists")
        self.pipeline_status_var.set(" -> ".join(ready_parts) if ready_parts else "Pick a CSV or place tiles into data/.")

        csv_name = Path(csv_path).name if csv_path else "no CSV selected"
        self.input_summary_var.set(f"{csv_name} | {xyz_count} XYZ, {tif_count} TIF")
        self.tiles_summary_var.set(f"{tile_count} tile STL files in output/tiles")
        self.output_summary_var.set(final_path.name if final_exists and final_path else "not built yet")
        self.detail_summary_var.set(self._detail_summary_text())
        self.merge_summary_var.set(self._merge_summary_text())

    def _detail_summary_text(self) -> str:
        preset = self.detail_preset_var.get().strip()
        if preset == "draft":
            return "Fastest export with lighter meshes."
        if preset == "fine":
            return "Higher detail and larger STL output."
        if preset == "custom":
            return "Custom mode uses your advanced conversion settings."
        return "Balanced for typical 3D print workflows."

    def _merge_summary_text(self) -> str:
        if self.make_solid_var.get():
            return f"Printable base enabled, thickness {self.base_thickness_var.get().strip() or '5.0'} mm."
        return "Surface-only merge, no printable base."

    def _apply_detail_preset(self) -> None:
        preset = self.detail_preset_var.get().strip()
        if preset == "draft":
            self.mode_var.set("auto")
            self.scale_mode_var.set("target_size")
            self.target_res_var.set("0.8")
        elif preset == "fine":
            self.mode_var.set("auto")
            self.scale_mode_var.set("target_size")
            self.target_res_var.set("0.2")
        elif preset == "balanced":
            self.mode_var.set("auto")
            self.scale_mode_var.set("target_size")
            self.target_res_var.set("0.3")

    def _queue_pipeline_commands(self, commands: list[tuple[list[str], str, str]]) -> None:
        if self.current_process is not None:
            messagebox.showinfo("Busy", "Another task is running.")
            return
        if not commands:
            return
        self.pending_commands = list(commands)
        self.pipeline_running = True
        self._start_next_command()

    def _start_next_command(self) -> None:
        if not self.pending_commands:
            self.pipeline_running = False
            self._refresh_pipeline_summary()
            return
        args, label, status_key = self.pending_commands.pop(0)
        self._run_command(args, label, status_key)

    def _run_full_pipeline(self) -> None:
        commands: list[tuple[list[str], str, str]] = []
        csv_path = self.csv_path_var.get().strip()
        xyz_count, tif_count = self._count_input_files()
        if csv_path:
            download_args = self._build_download_args()
            if download_args is None:
                return
            commands.append((download_args, "Download tiles", "download"))
        elif not (xyz_count or tif_count):
            messagebox.showwarning(
                "Missing input",
                "Select a CSV to download tiles, or place XYZ/TIF tiles into data/ before running the full pipeline.",
            )
            return

        convert_args = self._build_convert_args()
        if convert_args is None:
            return
        merge_args = self._build_merge_args()
        if merge_args is None:
            return
        commands.append((convert_args, "Convert tiles", "convert"))
        commands.append((merge_args, "Merge tiles", "merge"))
        self._queue_pipeline_commands(commands)

    def _copy_csv(self) -> None:
        src = self.csv_path_var.get().strip()
        if not src:
            messagebox.showwarning("Missing CSV", "Select a CSV file first.")
            return
        src_path = Path(src)
        if not src_path.exists():
            messagebox.showerror("Missing CSV", f"File not found: {src_path}")
            return
        dest = self.data_dir / src_path.name
        if dest.exists():
            if not messagebox.askyesno("Overwrite CSV", f"{dest} exists. Overwrite?"):
                return
        shutil.copy2(src_path, dest)
        self.csv_path_var.set(str(dest))
        self._log(f"Copied CSV to {dest}")
        self._refresh_pipeline_summary()

    def _browse_merge_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save merged STL",
            defaultextension=".stl",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            initialfile=Path(self.merge_out_var.get()).name,
        )
        if path:
            self.merge_out_var.set(path)
            self._auto_merge_out = path
            self._refresh_pipeline_summary()

    def _build_download_args(self) -> list[str] | None:
        args = [sys.executable, "download_tiles.py"]
        csv_path = self.csv_path_var.get().strip()
        if csv_path:
            args += ["--csv", csv_path]
        workers = self.download_workers_var.get().strip()
        if workers:
            args += ["--workers", workers]

        tif_dir = self.data_dir / "tif"
        tif_dir.mkdir(parents=True, exist_ok=True)
        existing_xyz = [p for p in self.xyz_dir.iterdir() if p.is_file()]
        existing_tif = [p for p in tif_dir.iterdir() if p.is_file()]
        if existing_xyz or existing_tif:
            if messagebox.askyesno(
                "Replace existing files?",
                "Existing XYZ or TIF files found in data/xyz or data/tif. Delete them before downloading?",
            ):
                args.append("--clean-xyz")

        return args

    def _build_convert_args(self) -> list[str] | None:
        args = [sys.executable, "build_stl.py", "--all"]
        self._apply_detail_preset()

        existing_tiles = []
        if self.tiles_dir.exists():
            existing_tiles = [p for p in self.tiles_dir.iterdir() if p.is_file()]
        if existing_tiles:
            if messagebox.askyesno(
                "Replace STL tiles?",
                "Existing STL tiles found in output/tiles. Delete them before converting?",
            ):
                args.append("--clean-tiles")

        if self.mode_var.get() == "auto":
            scale_mode = self.scale_mode_var.get().strip()
            if scale_mode == "target_size":
                target_size = self.target_size_var.get().strip()
                if target_size:
                    args += ["--target-size-mm", target_size]
                    target_edge = self.target_edge_var.get().strip()
                    if target_edge:
                        args += ["--target-edge", target_edge]
            elif scale_mode == "tile_size":
                tile_size = self.tile_size_var.get().strip()
                if tile_size:
                    args += ["--tile-size-mm", tile_size]
            elif scale_mode == "scale_ratio":
                scale_ratio = self.scale_ratio_var.get().strip()
                if scale_ratio:
                    args += ["--scale-ratio", scale_ratio]
            target_res = self.target_res_var.get().strip()
            if target_res:
                args += ["--target-resolution-mm", target_res]
        else:
            step = self.step_var.get().strip()
            if step:
                args += ["--step", step]
            scale_mode = self.scale_mode_var.get().strip()
            if scale_mode == "tile_size":
                tile_size = self.tile_size_var.get().strip()
                if tile_size:
                    args += ["--tile-size-mm", tile_size]
            elif scale_mode == "scale_ratio":
                scale_ratio = self.scale_ratio_var.get().strip()
                if scale_ratio:
                    args += ["--scale-ratio", scale_ratio]

        tol = self.tol_var.get().strip()
        if tol:
            args += ["--tol", tol]

        z_scale = self.z_scale_var.get().strip()
        if z_scale:
            args += ["--z-scale", z_scale]

        workers = self.convert_workers_var.get().strip()
        if workers:
            args += ["--workers", workers]

        model_name = self.model_name_var.get().strip()
        if model_name:
            args += ["--model-name", model_name]

        return args

    def _build_merge_args(self) -> list[str] | None:
        out_path = self.merge_out_var.get().strip()
        if not out_path:
            messagebox.showwarning("Missing output", "Please choose an output STL path.")
            return None

        out_parent = Path(out_path).resolve().parent
        out_parent.mkdir(parents=True, exist_ok=True)

        args = [sys.executable, "build_stl.py", "--merge-stl", out_path]

        weld = self.weld_tol_var.get().strip()
        if weld:
            args += ["--weld-tol", weld]

        merge_z = self.merge_z_scale_var.get().strip()
        if merge_z:
            args += ["--merge-z-scale", merge_z]

        lake_lower_mm = self.merge_lake_lower_mm_var.get().strip()
        if lake_lower_mm:
            args += ["--lake-lower-mm", lake_lower_mm]

        if self.make_solid_var.get():
            args.append("--make-solid")
            base_mode = self.base_mode_var.get().strip()
            if base_mode:
                args += ["--base-mode", base_mode]
            base_thick = self.base_thickness_var.get().strip()
            if base_thick:
                args += ["--base-thickness", base_thick]

        if self.merge_border_mode_var.get() == "clip":
            if not self.border_options:
                messagebox.showerror("Missing borders", "No border shapefiles found in ./geometry_data.")
                return None
            args.append("--clip-border")
            border_label = self.border_shp_var.get().strip()
            if border_label:
                border_path = self.border_options.get(border_label, Path(border_label))
                args += ["--border-shp", str(border_path)]
            if border_label:
                border_path = self.border_options.get(border_label, Path(border_label))
                border_type = self._border_type_from_path(border_path)
                if border_type in {"canton", "bezirk"}:
                    selections = [self.border_keep_list.get(i) for i in self.border_keep_list.curselection()]
                    selections = [s for s in selections if s and s != self.border_keep_all_label]
                    if selections:
                        args += ["--border-keep", ",".join(selections)]

        model_name = self.model_name_var.get().strip()
        if model_name:
            args += ["--model-name", model_name]

        return args

    def _run_download(self) -> None:
        self.download_progress_var.set(0.0)
        self.download_progress_label_var.set("")
        args = self._build_download_args()
        if args is None:
            return
        self._run_command(args, "Download tiles", status_key="download")

    def _run_convert(self) -> None:
        self.convert_progress_var.set(0.0)
        self.convert_progress_label_var.set("")
        args = self._build_convert_args()
        if args is None:
            return
        self._run_command(args, "Convert tiles", status_key="convert")

    def _run_merge(self) -> None:
        self.merge_progress_var.set(0.0)
        self.merge_progress_label_var.set("")
        args = self._build_merge_args()
        if args is None:
            return
        self._run_command(args, "Merge tiles", status_key="merge")

    def _on_model_name_change(self, _event: tk.Event) -> None:
        name = self.model_name_var.get().strip()
        if not name:
            return
        auto_path = str(self.output_dir / f"{name}.stl")
        if self.merge_out_var.get().strip() in {"", self._auto_merge_out}:
            self.merge_out_var.set(auto_path)
            self._auto_merge_out = auto_path
        self._refresh_pipeline_summary()

    def _run_command(self, args: list[str], label: str, status_key: str) -> None:
        if self.current_process is not None:
            messagebox.showinfo("Busy", "Another task is running.")
            return
        if args and args[0] == sys.executable and "-u" not in args:
            args.insert(1, "-u")
        self._set_status(status_key, "Running...", "#38bdf8")
        self.current_status_key = status_key

        self._log(f"[RUN] {label}")
        self._log(f"Command: {' '.join(args)}")

        def worker() -> None:
            try:
                self.current_process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path(__file__).resolve().parent,
                )
                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    self.log_queue.put(line.rstrip())
                self.current_process.wait()
                exit_code = self.current_process.returncode
                self.log_queue.put(f"[DONE] Exit code: {exit_code}")
                self.after(0, self._finalize_status, status_key, exit_code)
            except Exception as exc:
                self.log_queue.put(f"[ERROR] {exc}")
                self.after(0, self._finalize_status, status_key, 1)
            finally:
                self.current_process = None
                self.current_status_key = None

        threading.Thread(target=worker, daemon=True).start()

    def _poll_log(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_progress_line(line)
            self._log(line)
        self.after(100, self._poll_log)

    def _handle_progress_line(self, line: str) -> None:
        if not line.startswith("[PROGRESS]"):
            return
        try:
            payload = line[len("[PROGRESS]"):].strip()
            counter, name = payload.split(" ", 1)
            current_str, total_str = counter.split("/", 1)
            current = int(current_str)
            total = int(total_str)
        except ValueError:
            return
        if total <= 0:
            return
        pct = (current / total) * 100.0
        if self.current_status_key == "download":
            self.download_progress_var.set(pct)
            self.download_progress_label_var.set(f"Downloaded {current} of {total} ({name})")
        elif self.current_status_key == "convert":
            self.convert_progress_var.set(pct)
            self.convert_progress_label_var.set(f"Converted {current} of {total} ({name})")
        elif self.current_status_key == "merge":
            self.merge_progress_var.set(pct)
            self.merge_progress_label_var.set(f"Merged {current} of {total} ({name})")

    def _log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def _open_output_folder(self) -> None:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(self.output_dir.resolve())])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.output_dir.resolve())])
        else:
            subprocess.Popen(["xdg-open", str(self.output_dir.resolve())])

    def _update_convert_mode(self) -> None:
        if self.detail_preset_var.get().strip() != "custom":
            self.mode_var.set("auto")
            self.scale_mode_var.set("target_size")
        elif self.mode_var.get() != "auto" and self.scale_mode_var.get() == "target_size":
            self.scale_mode_var.set("tile_size")
        self._update_scale_mode()
        self.detail_summary_var.set(self._detail_summary_text())
        if self.show_convert_advanced_var.get() or self.detail_preset_var.get().strip() == "custom":
            self.convert_advanced_frame.grid()
        else:
            self.convert_advanced_frame.grid_remove()
        self._refresh_pipeline_summary()

    def _update_scale_mode(self) -> None:
        auto_mode = self.mode_var.get() == "auto"
        scale_mode = self.scale_mode_var.get()

        enable_target = auto_mode and scale_mode == "target_size"
        enable_tile = scale_mode == "tile_size"
        enable_ratio = scale_mode == "scale_ratio"
        enable_step = not auto_mode
        enable_res = auto_mode

        self.target_size_entry.configure(state="normal" if enable_target else "disabled")
        self.target_res_entry.configure(state="normal" if enable_res else "disabled")
        self.target_edge_combo.configure(state="readonly" if enable_target else "disabled")
        self.step_entry.configure(state="normal" if enable_step else "disabled")
        self.tile_size_entry.configure(state="normal" if enable_tile else "disabled")
        self.scale_ratio_entry.configure(state="normal" if enable_ratio else "disabled")

        self.target_size_label.configure(foreground="#e2e8f0" if enable_target else "#6b7280")
        self.target_res_label.configure(foreground="#e2e8f0" if enable_res else "#6b7280")
        self.target_edge_label.configure(foreground="#e2e8f0" if enable_target else "#6b7280")
        self.step_label.configure(foreground="#e2e8f0" if enable_step else "#6b7280")
        self.tile_size_label.configure(foreground="#e2e8f0" if enable_tile else "#6b7280")
        self.scale_ratio_label.configure(foreground="#e2e8f0" if enable_ratio else "#6b7280")

    def _update_merge_controls(self) -> None:
        make_solid = self.make_solid_var.get()
        base_mode = self.base_mode_var.get()
        border_mode = self.merge_border_mode_var.get()
        border_available = bool(self.border_options)
        clip_border = border_mode == "clip" and border_available
        border_path = None
        if border_available:
            border_label = self.border_shp_var.get().strip()
            if border_label:
                border_path = self.border_options.get(border_label, Path(border_label))
        border_type = self._border_type_from_path(border_path) if border_path else None
        border_keep_enabled = border_available and border_type in {"canton", "bezirk"}

        if make_solid:
            self.base_mode_combo.configure(state="readonly")
            self.base_mode_label.configure(foreground="#e2e8f0")
            if base_mode == "sealevel":
                self.base_thickness_entry.configure(state="disabled")
                self.base_thickness_label.configure(foreground="#6b7280")
            else:
                self.base_thickness_entry.configure(state="normal")
                self.base_thickness_label.configure(foreground="#e2e8f0")
        else:
            self.base_mode_combo.configure(state="disabled")
            self.base_thickness_entry.configure(state="disabled")
            self.base_mode_label.configure(foreground="#6b7280")
            self.base_thickness_label.configure(foreground="#6b7280")

        if self.show_merge_advanced_var.get():
            self.merge_advanced_frame.grid()
        else:
            self.merge_advanced_frame.grid_remove()

        if not border_available:
            self.merge_border_mode_var.set("all")
            clip_border = False

        self.border_shp_combo.configure(state="readonly" if clip_border else "disabled")
        self.border_shp_label.configure(foreground="#e2e8f0" if clip_border else "#6b7280")
        self.border_keep_list.configure(state="normal" if border_keep_enabled else "disabled")
        self.border_keep_refresh_btn.configure(state="normal" if border_keep_enabled else "disabled")
        self.border_keep_label.configure(foreground="#e2e8f0" if border_keep_enabled else "#6b7280")
        if not border_keep_enabled:
            self.border_keep_var.set("")
            self.border_keep_list.delete(0, tk.END)
        elif not self.border_keep_options:
            self._refresh_border_keep_options()
        self.merge_summary_var.set(self._merge_summary_text())
        self._refresh_pipeline_summary()

    def _set_status(self, key: str, text: str, color: str) -> None:
        var = self.status_vars.get(key)
        label = self.status_labels.get(key)
        if var is None or label is None:
            return
        var.set(text)
        label.configure(foreground=color)

    def _finalize_status(self, key: str, exit_code: int) -> None:
        if exit_code == 0:
            self._set_status(key, "Success", "#22c55e")
        else:
            self._set_status(key, "Failed", "#ef4444")
        self._refresh_pipeline_summary()
        if self.pipeline_running:
            if exit_code == 0:
                self._start_next_command()
            else:
                self.pending_commands = []
                self.pipeline_running = False


if __name__ == "__main__":
    app = App()
    app.mainloop()
