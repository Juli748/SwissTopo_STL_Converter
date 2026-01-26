import shutil
import subprocess
import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from defaults import DEFAULTS


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

        self._setup_theme()
        self._build_ui()
        self._load_default_csv()
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
        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text=DEFAULTS["window_title"], style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Download tiles, convert to STL, then merge into a print-ready model.",
        ).pack(anchor=tk.W)
        name_row = ttk.Frame(header)
        name_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(name_row, text="Model name:").pack(side=tk.LEFT)
        self.model_name_entry = ttk.Entry(name_row, textvariable=self.model_name_var, width=24)
        self.model_name_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.model_name_entry.bind("<KeyRelease>", self._on_model_name_change)
        self.tooltips.append(
            Tooltip(
                self.model_name_entry,
                "Used to name output STL files and the STL solid name.",
            )
        )

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

    def _build_step_download(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1) Download XYZ tiles")
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
            text="Download into data/xyz using download_tiles.py",
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
                "Runs download_tiles.py to fetch tiles into ./data/xyz.",
            ),
            Tooltip(
                download_note,
                "Tiles are stored under ./data/xyz as .xyz files (X Y Z per line).",
            ),
        ]

    def _build_step_convert(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2) Convert XYZ to STL tiles")
        frame.pack(fill=tk.X)

        self.mode_var = tk.StringVar(value=DEFAULTS["mode"])
        self.target_size_var = tk.StringVar(value=DEFAULTS["target_size_mm"])
        self.target_res_var = tk.StringVar(value=DEFAULTS["target_resolution_mm"])
        self.step_var = tk.StringVar(value=DEFAULTS["step"])
        self.tol_var = tk.StringVar(value=DEFAULTS["grid_tolerance"])
        self.z_scale_var = tk.StringVar(value=DEFAULTS["z_scale"])

        mode_label = ttk.Label(frame, text="Choose how detail is set:")
        mode_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        auto_radio = ttk.Radiobutton(
            frame,
            text="Auto (size in mm)",
            value="auto",
            variable=self.mode_var,
            command=self._update_convert_mode,
        )
        auto_radio.grid(row=0, column=1, sticky=tk.W)
        manual_radio = ttk.Radiobutton(
            frame,
            text="Manual (step)",
            value="manual",
            variable=self.mode_var,
            command=self._update_convert_mode,
        )
        manual_radio.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        self.target_size_label = ttk.Label(frame, text="Target smallest edge (mm):")
        self.target_size_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.target_size_entry = ttk.Entry(frame, textvariable=self.target_size_var, width=10)
        self.target_size_entry.grid(row=1, column=1, sticky=tk.W)
        self.target_res_label = ttk.Label(frame, text="Target XY spacing (mm):")
        self.target_res_label.grid(row=1, column=2, sticky=tk.W, padx=(12, 0))
        self.target_res_entry = ttk.Entry(frame, textvariable=self.target_res_var, width=10)
        self.target_res_entry.grid(row=1, column=3, sticky=tk.W)

        self.step_label = ttk.Label(frame, text="Downsample step:")
        self.step_label.grid(row=2, column=0, sticky=tk.W, pady=4)
        self.step_entry = ttk.Entry(frame, textvariable=self.step_var, width=10)
        self.step_entry.grid(row=2, column=1, sticky=tk.W)
        tol_label = ttk.Label(frame, text="Grid tolerance:")
        tol_label.grid(row=2, column=2, sticky=tk.W, padx=(12, 0))
        tol_entry = ttk.Entry(frame, textvariable=self.tol_var, width=10)
        tol_entry.grid(row=2, column=3, sticky=tk.W)

        z_scale_label = ttk.Label(frame, text="Z scale (tile conversion):")
        z_scale_label.grid(row=3, column=0, sticky=tk.W, pady=4)
        z_scale_entry = ttk.Entry(frame, textvariable=self.z_scale_var, width=10)
        z_scale_entry.grid(row=3, column=1, sticky=tk.W)
        auto_hint = ttk.Label(frame, text="Auto = program picks step to hit print size.")
        auto_hint.grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=(12, 0))

        convert_workers_label = ttk.Label(frame, text="Max parallel conversions:")
        convert_workers_label.grid(row=4, column=0, sticky=tk.W, pady=4)
        self.convert_workers_var = tk.StringVar(value=DEFAULTS["convert_workers"])
        convert_workers_entry = ttk.Entry(frame, textvariable=self.convert_workers_var, width=10)
        convert_workers_entry.grid(row=4, column=1, sticky=tk.W)

        self.convert_btn = ttk.Button(frame, text="Run Conversion", command=self._run_convert)
        self.convert_btn.grid(row=5, column=3, sticky=tk.E, padx=4, pady=6)
        self.status_vars["convert"] = tk.StringVar(value=DEFAULTS["status_idle"])
        self.status_labels["convert"] = ttk.Label(frame, textvariable=self.status_vars["convert"])
        self.status_labels["convert"].grid(
            row=5, column=2, sticky=tk.W, padx=(12, 0)
        )
        self.convert_progress = ttk.Progressbar(
            frame,
            variable=self.convert_progress_var,
            maximum=100.0,
            mode="determinate",
            length=220,
        )
        self.convert_progress.grid(row=6, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.convert_progress_label = ttk.Label(frame, textvariable=self.convert_progress_label_var)
        self.convert_progress_label.grid(row=6, column=3, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.convert_progress_label.configure(width=26)

        self._update_convert_mode()

        self.tooltips += [
            Tooltip(
                mode_label,
                "Auto chooses a downsample step to hit the requested print size. Manual lets you set the step.",
            ),
            Tooltip(
                auto_radio,
                "Auto: set a target print size; the tool computes an appropriate step.",
            ),
            Tooltip(
                manual_radio,
                "Manual: set the downsample step directly (keep every Nth point).",
            ),
            Tooltip(
                self.target_size_label,
                "Smallest edge of the final print in mm. Used with Auto mode (--target-size-mm).",
            ),
            Tooltip(
                self.target_size_entry,
                "Example: 150 means the smallest edge of the merged model is ~150 mm.",
            ),
            Tooltip(
                self.target_res_label,
                "Desired XY point spacing in mm when using Auto (--target-resolution-mm).",
            ),
            Tooltip(
                self.target_res_entry,
                "Lower values keep more detail but create larger STL files.",
            ),
            Tooltip(
                self.step_label,
                "Keep every Nth grid point in X and Y (--step). Larger step = fewer points.",
            ),
            Tooltip(
                self.step_entry,
                "Example: 10 keeps every 10th point.",
            ),
            Tooltip(
                tol_label,
                "Snap X/Y to a grid to fix tiny coordinate noise (--tol). Use for messy data.",
            ),
            Tooltip(
                tol_entry,
                "Example: 0.001 snaps to 1 mm in map units if units are meters.",
            ),
            Tooltip(
                z_scale_label,
                "Multiply elevations by this factor during tile conversion (--z-scale).",
            ),
            Tooltip(
                z_scale_entry,
                "Example: 2.0 doubles height for exaggeration.",
            ),
            Tooltip(
                auto_hint,
                "Auto mode ignores the manual step field.",
            ),
            Tooltip(
                convert_workers_label,
                "How many tiles to convert in parallel. Higher is faster but uses more CPU/RAM.",
            ),
            Tooltip(
                convert_workers_entry,
                "Try 2-4 on a typical laptop. Set to 1 for sequential conversion.",
            ),
            Tooltip(
                self.convert_btn,
                "Runs build_stl.py --all with the selected options.",
            ),
        ]

    def _build_step_merge(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3) Merge tiles into final STL")
        frame.pack(fill=tk.X)

        self._auto_merge_out = str(self.output_dir / DEFAULTS["merge_out"])
        self.merge_out_var = tk.StringVar(value=self._auto_merge_out)
        self.weld_tol_var = tk.StringVar(value=DEFAULTS["weld_tolerance"])
        self.merge_z_scale_var = tk.StringVar(value=DEFAULTS["merge_z_scale"])
        self.make_solid_var = tk.BooleanVar(value=DEFAULTS["make_solid"])
        self.base_thickness_var = tk.StringVar(value=DEFAULTS["base_thickness"])
        self.base_z_var = tk.StringVar(value=DEFAULTS["base_z"])

        merge_out_label = ttk.Label(frame, text="Output STL path:")
        merge_out_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        merge_out_entry = ttk.Entry(frame, textvariable=self.merge_out_var)
        merge_out_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW)
        merge_out_btn = ttk.Button(frame, text="Browse", command=self._browse_merge_out)
        merge_out_btn.grid(row=0, column=4, padx=4, pady=2)

        weld_label = ttk.Label(frame, text="Weld tolerance:")
        weld_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        weld_entry = ttk.Entry(frame, textvariable=self.weld_tol_var, width=10)
        weld_entry.grid(row=1, column=1, sticky=tk.W)
        merge_z_label = ttk.Label(frame, text="Merge Z scale:")
        merge_z_label.grid(row=1, column=2, sticky=tk.W, padx=(12, 0))
        merge_z_entry = ttk.Entry(frame, textvariable=self.merge_z_scale_var, width=10)
        merge_z_entry.grid(row=1, column=3, sticky=tk.W)

        make_solid_chk = ttk.Checkbutton(frame, text="Make solid", variable=self.make_solid_var)
        make_solid_chk.grid(row=2, column=0, sticky=tk.W)
        base_thickness_label = ttk.Label(frame, text="Base thickness:")
        base_thickness_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        base_thickness_entry = ttk.Entry(frame, textvariable=self.base_thickness_var, width=10)
        base_thickness_entry.grid(row=2, column=2, sticky=tk.W)
        base_z_label = ttk.Label(frame, text="Base Z (optional):")
        base_z_label.grid(row=2, column=3, sticky=tk.W, pady=2)
        base_z_entry = ttk.Entry(frame, textvariable=self.base_z_var, width=10)
        base_z_entry.grid(row=2, column=4, sticky=tk.W)

        self.merge_btn = ttk.Button(frame, text="Run Merge", command=self._run_merge)
        self.merge_btn.grid(row=3, column=4, sticky=tk.E, padx=4, pady=6)
        self.status_vars["merge"] = tk.StringVar(value=DEFAULTS["status_idle"])
        self.status_labels["merge"] = ttk.Label(frame, textvariable=self.status_vars["merge"])
        self.status_labels["merge"].grid(
            row=3, column=3, sticky=tk.W, padx=(12, 0)
        )

        frame.columnconfigure(1, weight=1)

        self.tooltips += [
            Tooltip(
                merge_out_label,
                "Where the merged STL will be saved (--merge-stl).",
            ),
            Tooltip(
                merge_out_entry,
                "All tiles in ./output/tiles are merged into this file.",
            ),
            Tooltip(
                merge_out_btn,
                "Choose a filename for the merged STL.",
            ),
            Tooltip(
                weld_label,
                "Vertices within this distance are welded to remove seams (--weld-tol).",
            ),
            Tooltip(
                weld_entry,
                "Example: 0.001 is a typical tolerance in map units.",
            ),
            Tooltip(
                merge_z_label,
                "Scale elevations during merge only (--merge-z-scale).",
            ),
            Tooltip(
                merge_z_entry,
                "Use to exaggerate or reduce relief on the final model.",
            ),
            Tooltip(
                make_solid_chk,
                "Adds side walls and a flat base to make the model printable (--make-solid).",
            ),
            Tooltip(
                base_thickness_label,
                "Distance below the minimum terrain to place the base (--base-thickness).",
            ),
            Tooltip(
                base_thickness_entry,
                "Used only when Make solid is enabled.",
            ),
            Tooltip(
                base_z_label,
                "Explicit base plane elevation (--base-z). Overrides base thickness.",
            ),
            Tooltip(
                base_z_entry,
                "Leave blank to use base thickness instead.",
            ),
            Tooltip(
                self.merge_btn,
                "Runs build_stl.py --merge-stl with the selected options.",
            ),
        ]

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def _load_default_csv(self) -> None:
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if csv_files and not self.csv_path_var.get().strip():
            self.csv_path_var.set(str(csv_files[0]))

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

    def _run_download(self) -> None:
        self.download_progress_var.set(0.0)
        self.download_progress_label_var.set("")
        args = [sys.executable, "download_tiles.py"]
        csv_path = self.csv_path_var.get().strip()
        if csv_path:
            args += ["--csv", csv_path]
        workers = self.download_workers_var.get().strip()
        if workers:
            args += ["--workers", workers]

        existing_xyz = [p for p in self.xyz_dir.iterdir() if p.is_file()]
        if existing_xyz:
            if messagebox.askyesno(
                "Replace XYZ files?",
                "Existing XYZ files found in data/xyz. Delete them before downloading?",
            ):
                args.append("--clean-xyz")

        self._run_command(args, "Download tiles", status_key="download")

    def _run_convert(self) -> None:
        self.convert_progress_var.set(0.0)
        self.convert_progress_label_var.set("")
        args = [sys.executable, "build_stl.py", "--all"]

        if self.mode_var.get() == "auto":
            target_size = self.target_size_var.get().strip()
            if target_size:
                args += ["--target-size-mm", target_size]
            target_res = self.target_res_var.get().strip()
            if target_res:
                args += ["--target-resolution-mm", target_res]
        else:
            step = self.step_var.get().strip()
            if step:
                args += ["--step", step]

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

        self._run_command(args, "Convert tiles", status_key="convert")

    def _run_merge(self) -> None:
        out_path = self.merge_out_var.get().strip()
        if not out_path:
            messagebox.showwarning("Missing output", "Please choose an output STL path.")
            return

        out_parent = Path(out_path).resolve().parent
        out_parent.mkdir(parents=True, exist_ok=True)

        args = [sys.executable, "build_stl.py", "--merge-stl", out_path]

        weld = self.weld_tol_var.get().strip()
        if weld:
            args += ["--weld-tol", weld]

        merge_z = self.merge_z_scale_var.get().strip()
        if merge_z:
            args += ["--merge-z-scale", merge_z]

        if self.make_solid_var.get():
            args.append("--make-solid")
            base_thick = self.base_thickness_var.get().strip()
            if base_thick:
                args += ["--base-thickness", base_thick]
            base_z = self.base_z_var.get().strip()
            if base_z:
                args += ["--base-z", base_z]

        model_name = self.model_name_var.get().strip()
        if model_name:
            args += ["--model-name", model_name]

        self._run_command(args, "Merge tiles", status_key="merge")

    def _on_model_name_change(self, _event: tk.Event) -> None:
        name = self.model_name_var.get().strip()
        if not name:
            return
        auto_path = str(self.output_dir / f"{name}.stl")
        if self.merge_out_var.get().strip() in {"", self._auto_merge_out}:
            self.merge_out_var.set(auto_path)
            self._auto_merge_out = auto_path

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
        if self.mode_var.get() == "auto":
            self.target_size_entry.configure(state="normal")
            self.target_res_entry.configure(state="normal")
            self.step_entry.configure(state="disabled")
            self.target_size_label.configure(foreground="#e2e8f0")
            self.target_res_label.configure(foreground="#e2e8f0")
            self.step_label.configure(foreground="#6b7280")
        else:
            self.target_size_entry.configure(state="disabled")
            self.target_res_entry.configure(state="disabled")
            self.step_entry.configure(state="normal")
            self.target_size_label.configure(foreground="#6b7280")
            self.target_res_label.configure(foreground="#6b7280")
            self.step_label.configure(foreground="#e2e8f0")

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


if __name__ == "__main__":
    app = App()
    app.mainloop()
