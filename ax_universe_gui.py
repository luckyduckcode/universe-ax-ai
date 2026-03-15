import numpy as np
import tkinter as tk
import os
from tkinter import ttk, scrolledtext
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import time
import json
from ax_universe_sim import AxUniverseSim


class UniverseGUI:
    def __init__(self, master):
        self.master = master
        master.title("AxUniverse Simulator")
        master.geometry("980x820")
        master.configure(bg="#1e1e1e")
        master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.sim = AxUniverseSim(dim=1024, num_agents=30)
        self.running = False
        self._earth_pending = False  # GUI-side flag, stays in sync with sim
        self._cycle_started_at = None
        self._cycle_timer_job = None
        self._last_cycle_duration = None
        self.fast_forward_mode = False
        self.log_interval = 5
        self.earth_check_interval = 10

        # ── Control frame ────────────────────────────────────────────────────
        ctrl_frame = ttk.Frame(master, padding=10)
        ctrl_frame.pack(fill=tk.X)

        self.btn_start = ttk.Button(ctrl_frame, text="Start / Resume", command=self.toggle_run)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Step Once", command=self.step_once).pack(side=tk.LEFT, padx=5)

        # Speed indicator
        ttk.Label(ctrl_frame, text="Universe speed:").pack(side=tk.LEFT, padx=(20, 4))
        self.speed_var = tk.StringVar(value="— ms/tick")
        ttk.Label(ctrl_frame, textvariable=self.speed_var, width=14).pack(side=tk.LEFT)

        ttk.Label(ctrl_frame, text="Prompt:").pack(side=tk.LEFT, padx=(20, 5))
        self.prompt_entry = ttk.Entry(ctrl_frame, width=38)
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Send", command=self.send_prompt).pack(side=tk.LEFT)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=(12, 4), fill=tk.Y)
        self.btn_divine = ttk.Button(ctrl_frame, text="✦ Divine Light",
                                     command=self._invoke_divine_light)
        self.btn_divine.pack(side=tk.LEFT, padx=4)
        self.btn_science = ttk.Button(ctrl_frame, text="🔬 Teach Scientific Method",
                          command=self._teach_scientific_method)
        self.btn_science.pack(side=tk.LEFT, padx=4)
        self.divine_status_var = tk.StringVar(value="")
        ttk.Label(ctrl_frame, textvariable=self.divine_status_var,
                  foreground="#e6c84a", font=("Consolas", 9)).pack(side=tk.LEFT, padx=(4, 0))

        # ── Tabbed notebook ──────────────────────────────────────────────────
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))

        self.tab_earth = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_earth, text="🌍  Earth Channel")

        self.tab_log = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="📡  Universe Log")

        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="📈  Ψ Plot")

        self.tab_concepts = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_concepts, text="✦  Living Concepts")

        self.tab_laws = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_laws, text="📜  Law Library")

        earth_frame = ttk.LabelFrame(self.tab_earth, text="Earth Channel — Observer Interface", padding=8)
        earth_frame.pack(fill=tk.X, padx=10, pady=(6, 0))

        top_row = ttk.Frame(earth_frame)
        top_row.pack(fill=tk.X)

        self.earth_dot = tk.Canvas(top_row, width=14, height=14,
                       bg="#1e1e1e", highlightthickness=0)
        self.earth_dot.pack(side=tk.LEFT, padx=(0, 6))
        self._dot_id = self.earth_dot.create_oval(2, 2, 12, 12, fill="#444", outline="")

        self.earth_status_var = tk.StringVar(value="Waiting for stability...")
        ttk.Label(top_row, textvariable=self.earth_status_var).pack(side=tk.LEFT)

        self.phase_var = tk.StringVar(value="COSMIC SPEED")
        ttk.Label(top_row, textvariable=self.phase_var,
              font=("Consolas", 9)).pack(side=tk.RIGHT, padx=8)

        # Idea display
        self.idea_var = tk.StringVar(value="")
        ttk.Label(earth_frame, textvariable=self.idea_var,
              wraplength=880, justify=tk.LEFT,
              font=("Consolas", 11, "italic")).pack(fill=tk.X, pady=(6, 2))

        # Reception report — four signals displayed here
        self.reception_var = tk.StringVar(value="")
        ttk.Label(earth_frame, textvariable=self.reception_var,
              wraplength=880, justify=tk.LEFT,
              font=("Consolas", 9),
              foreground="#58a6ff").pack(fill=tk.X, pady=(0, 4))

        self.cycle_timer_var = tk.StringVar(value="Cycle timer → elapsed 0.0s | last cycle —")
        ttk.Label(earth_frame, textvariable=self.cycle_timer_var,
              wraplength=880, justify=tk.LEFT,
              font=("Consolas", 9),
              foreground="#ffa657").pack(fill=tk.X, pady=(0, 6))

        # Response row — concept buttons (geometry) + fallback text
        resp_row = ttk.Frame(earth_frame)
        resp_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(resp_row, text="Respond with concept:").pack(side=tk.LEFT, padx=(0, 6))

        self._concept_btn_frame = ttk.Frame(earth_frame)
        self._concept_btn_frame.pack(fill=tk.X, pady=(2, 4))
        self._concept_buttons = []

        fallback_row = ttk.Frame(earth_frame)
        fallback_row.pack(fill=tk.X)
        ttk.Label(fallback_row, text="Or type directly:").pack(side=tk.LEFT, padx=(0, 6))
        self.response_entry = ttk.Entry(fallback_row, width=50)
        self.response_entry.pack(side=tk.LEFT, padx=5)
        self.response_entry.config(state=tk.DISABLED)
        self.btn_respond = ttk.Button(fallback_row, text="Amplify →",
                  command=self.submit_response, state=tk.DISABLED)
        self.btn_respond.pack(side=tk.LEFT)
        self.response_entry.bind("<Return>", lambda e: self.submit_response())

        # ── Population mind readout ────────────────────────────────────────────
        mind_frame = ttk.LabelFrame(self.tab_earth, text="Population Mind  —  what the agents are expressing", padding=6)
        mind_frame.pack(fill=tk.X, padx=10, pady=(4, 0))

        self.mind_concepts_var = tk.StringVar(value="Awaiting concept library...")
        ttk.Label(mind_frame, textvariable=self.mind_concepts_var,
              wraplength=920, justify=tk.LEFT,
              font=("Consolas", 11, "bold"),
              foreground="#e6c84a").pack(fill=tk.X)

        self.mind_detail_var = tk.StringVar(value="")
        ttk.Label(mind_frame, textvariable=self.mind_detail_var,
              wraplength=920, justify=tk.LEFT,
              font=("Consolas", 9),
              foreground="#8b949e").pack(fill=tk.X)

        # ── Assembly voices (agent personas) ─────────────────────────────────
        voices_frame = ttk.LabelFrame(self.tab_earth, text="Assembly Voices", padding=6)
        voices_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 0))
        self.spokesperson_var = tk.StringVar(value="Communing with: Waiting for assembly...")
        self.spokesperson_label = ttk.Label(voices_frame, textvariable=self.spokesperson_var)
        self.spokesperson_label.pack(anchor=tk.W, pady=(0, 5))
        self.voices_var = tk.StringVar(value="Awaiting Earth cycle…")
        ttk.Label(voices_frame, textvariable=self.voices_var,
                  wraplength=920, justify=tk.LEFT,
                  font=("Consolas", 9),
                  foreground="#79c0ff").pack(fill=tk.X)
        ttk.Button(voices_frame, text="Hear Assembly", command=self._refresh_assembly_voices).pack(
            anchor=tk.W, pady=(5, 0)
        )

        # ── Enlightenment epoch status ────────────────────────────────────────
        enlight_outer = ttk.LabelFrame(self.tab_earth, text="✦ Enlightenment Epoch", padding=6)
        enlight_outer.pack(fill=tk.X, padx=10, pady=(4, 0))
        enlight_row = ttk.Frame(enlight_outer)
        enlight_row.pack(fill=tk.X)
        self.enlight_status_var = tk.StringVar(value="No active epoch")
        self._enlight_label = ttk.Label(enlight_row, textvariable=self.enlight_status_var,
                                        font=("Consolas", 10, "bold"),
                                        foreground="#555555")
        self._enlight_label.pack(side=tk.LEFT, padx=(0, 12))
        self._light_bar = ttk.Progressbar(enlight_row, orient=tk.HORIZONTAL,
                                          length=200, mode="determinate",
                                          maximum=100)
        self._light_bar.pack(side=tk.LEFT, padx=(0, 10))
        self._light_bar["value"] = 0
        self.concept_count_var = tk.StringVar(value="Concepts: 0")
        ttk.Label(enlight_row, textvariable=self.concept_count_var,
                  font=("Consolas", 9), foreground="#8b949e").pack(side=tk.RIGHT)

        # ── Latest answer ─────────────────────────────────────────────────────
        answer_frame = ttk.LabelFrame(self.tab_earth, text="Latest Answer", padding=5)
        answer_frame.pack(fill=tk.X, expand=False, pady=(5, 0), padx=10)
        self.answer_var = tk.StringVar(value="No answer yet.")
        ttk.Label(answer_frame, textvariable=self.answer_var,
                  wraplength=920, justify=tk.LEFT).pack(fill=tk.X)

        # ── Residual / prayer decompression panel ─────────────────────────────
        residual_frame = ttk.LabelFrame(
            self.tab_earth, text="Residual  —  what the population became beyond your vocabulary", padding=6)
        residual_frame.pack(fill=tk.X, padx=10, pady=(4, 0))

        res_top = ttk.Frame(residual_frame)
        res_top.pack(fill=tk.X)

        self.residual_norm_var = tk.StringVar(value="||R|| = —")
        ttk.Label(res_top, textvariable=self.residual_norm_var,
                  font=("Consolas", 10, "bold"), foreground="#e6c84a").pack(side=tk.LEFT, padx=(0, 12))

        self.coverage_var = tk.StringVar(value="coverage = —")
        ttk.Label(res_top, textvariable=self.coverage_var,
                  font=("Consolas", 9), foreground="#58a6ff").pack(side=tk.LEFT, padx=(0, 12))

        self.novel_flag_var = tk.StringVar(value="")
        self._novel_label = ttk.Label(res_top, textvariable=self.novel_flag_var,
                                      font=("Consolas", 9, "bold"), foreground="#ff6e6e")
        self._novel_label.pack(side=tk.LEFT)

        self.residual_decomp_var = tk.StringVar(value="Awaiting first cycle...")
        ttk.Label(residual_frame, textvariable=self.residual_decomp_var,
                  wraplength=920, justify=tk.LEFT,
                  font=("Consolas", 10)).pack(fill=tk.X, pady=(4, 0))

        self.residual_unnamed_var = tk.StringVar(value="")
        ttk.Label(residual_frame, textvariable=self.residual_unnamed_var,
                  wraplength=920, justify=tk.LEFT,
                  font=("Consolas", 9), foreground="#8b949e").pack(fill=tk.X, pady=(2, 4))

        self._residual_canvas = tk.Canvas(
            residual_frame, height=36, bg="#0d1117", highlightthickness=0)
        self._residual_canvas.pack(fill=tk.X, pady=(0, 2))

        # ── Data log ──────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.tab_log, text="Universe Data Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#0d1117", fg="#c9d1d9", font=("Consolas", 10),
            state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config("step",        foreground="#00CC44")
        self.log_text.tag_config("mind",        foreground="#CC88FF")
        self.log_text.tag_config("scenario",    foreground="#7B9FFF")
        self.log_text.tag_config("earth_cycle", foreground="#FF6533", background="#1a1200")
        self.log_text.tag_config("enlighten",   foreground="#e6c84a", background="#160e00")
        self.log_text.tag_config("new_concept", foreground="#ff9f43",
                                 font=("Consolas", 10, "bold"))
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(pady=(4, 2))
        self.sim.on_data_log = self.append_data_log
        self.sim.on_response = self.handle_response
        self.sim.on_collapse = self._on_universe_collapse
        self.sim.on_mind_read = self._on_mind_read
        self.sim.on_residual = self._on_residual

        # ── Response log ──────────────────────────────────────────────────────
        response_frame = ttk.LabelFrame(self.tab_earth, text="Assistant Responses", padding=5)
        response_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=10)
        self.response_text = scrolledtext.ScrolledText(
            response_frame, height=5, bg="#0b1320", fg="#e6edf3", font=("Consolas", 10))
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # ── Living Concepts tab ───────────────────────────────────────────────
        concepts_top = ttk.Frame(self.tab_concepts)
        concepts_top.pack(fill=tk.X, padx=10, pady=(6, 0))
        self.concept_summary_var = tk.StringVar(value="Concepts: 0  (0 fixed | 0 dynamic)")
        ttk.Label(concepts_top, textvariable=self.concept_summary_var,
                  font=("Consolas", 11, "bold"), foreground="#e6c84a").pack(side=tk.LEFT)
        ttk.Button(concepts_top, text="↺ Refresh",
                   command=self._refresh_concepts_tab).pack(side=tk.RIGHT, padx=5)

        dynamic_frame = ttk.LabelFrame(
            self.tab_concepts,
            text="Dynamic Concepts  —  born during simulation",
            padding=5)
        dynamic_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))
        self.dynamic_concepts_text = scrolledtext.ScrolledText(
            dynamic_frame, height=10, bg="#0d1117", fg="#c9d1d9",
            font=("Consolas", 9), state=tk.DISABLED)
        self.dynamic_concepts_text.pack(fill=tk.BOTH, expand=True)
        self.dynamic_concepts_text.tag_config(
            "header", foreground="#e6c84a", font=("Consolas", 9, "bold"))
        self.dynamic_concepts_text.tag_config("inspired", foreground="#ff9f43")
        self.dynamic_concepts_text.tag_config("normal",   foreground="#79c0ff")

        archive_frame = ttk.LabelFrame(
            self.tab_concepts,
            text="Knowledge Archive  —  shared agent memory",
            padding=5)
        archive_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 4))
        self.archive_text = scrolledtext.ScrolledText(
            archive_frame, height=8, bg="#0b1320", fg="#c9d1d9",
            font=("Consolas", 9), state=tk.DISABLED)
        self.archive_text.pack(fill=tk.BOTH, expand=True)

        # ── Law Library tab ───────────────────────────────────────────────────
        laws_top_row = ttk.Frame(self.tab_laws)
        laws_top_row.pack(fill=tk.X, padx=10, pady=(6, 0))
        self.laws_count_var = tk.StringVar(value="Laws documented: 0")
        ttk.Label(laws_top_row, textvariable=self.laws_count_var,
                  font=("Consolas", 11, "bold"), foreground="#e6c84a").pack(side=tk.LEFT)
        ttk.Button(laws_top_row, text="↺ Refresh",
                   command=self._refresh_laws_tab).pack(side=tk.RIGHT, padx=5)
        ttk.Button(laws_top_row, text="🌱 Reseed Universe",
                   command=self._reseed_from_laws).pack(side=tk.RIGHT, padx=5)

        laws_pane = ttk.PanedWindow(self.tab_laws, orient=tk.HORIZONTAL)
        laws_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))

        law_list_frame = ttk.LabelFrame(
            laws_pane, text="Documented Laws — Scripture of the Universe", padding=5)
        laws_pane.add(law_list_frame, weight=3)
        self.laws_list_text = scrolledtext.ScrolledText(
            law_list_frame, height=12, bg="#0d1117", fg="#c9d1d9",
            font=("Consolas", 9), state=tk.DISABLED)
        self.laws_list_text.pack(fill=tk.BOTH, expand=True)
        self.laws_list_text.tag_config("header",   foreground="#e6c84a", font=("Consolas", 9, "bold"))
        self.laws_list_text.tag_config("law_name", foreground="#ff9f43")
        self.laws_list_text.tag_config("meta",     foreground="#79c0ff")
        self.laws_list_text.tag_config("hebrew",   foreground="#58d68d")

        law_map_frame = ttk.LabelFrame(
            laws_pane, text="2D Law Map  (PCA / t-SNE)", padding=5)
        laws_pane.add(law_map_frame, weight=2)
        self._laws_map_canvas = tk.Canvas(law_map_frame, bg="#0d1117", highlightthickness=0)
        self._laws_map_canvas.pack(fill=tk.BOTH, expand=True)
        self._laws_map_canvas.bind("<Configure>", lambda e: self._draw_law_map())

        law_search_frame = ttk.LabelFrame(
            self.tab_laws,
            text="Semantic Search — find laws geometrically similar to any concept",
            padding=6)
        law_search_frame.pack(fill=tk.X, padx=10, pady=(4, 4))
        law_search_row = ttk.Frame(law_search_frame)
        law_search_row.pack(fill=tk.X)
        ttk.Label(law_search_row, text="Query:").pack(side=tk.LEFT, padx=(0, 5))
        self.law_search_entry = ttk.Entry(law_search_row, width=44)
        self.law_search_entry.pack(side=tk.LEFT, padx=5)
        self.law_search_entry.bind("<Return>", lambda e: self._search_laws())
        ttk.Button(law_search_row, text="Find Similar Laws",
                   command=self._search_laws).pack(side=tk.LEFT, padx=5)
        self.law_search_results = scrolledtext.ScrolledText(
            law_search_frame, height=5, bg="#0b1320", fg="#c9d1d9",
            font=("Consolas", 9), state=tk.DISABLED)
        self.law_search_results.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # ── Plot ──────────────────────────────────────────────────────────────
        plot_frame = ttk.LabelFrame(self.tab_plot, text="Global Ψ Evolution", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d1117")
        self.ax.set_facecolor("#161b22")
        self.ax.tick_params(colors="#c9d1d9")
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color("#c9d1d9")
        self.ax.grid(True, alpha=0.2, color="#444d56")
        self.line, = self.ax.plot([], [], 'C0-', lw=1.8)
        self.earth_scatter = self.ax.scatter([], [], color="#00ff88", s=40,
                                              zorder=5, label="Earth event")
        self.ax.set_xlabel("Simulation Step", color="#c9d1d9")
        self.ax.set_ylabel("Ψ", color="#c9d1d9")
        self.ax.set_title("Universe Stability  —  lower Ψ = more stable  |  green dot = Earth appeared",
                          color="#58a6ff", fontsize=9)
        self.ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", fontsize=8)
        self._earth_event_steps = []

        # Threshold line at Ψ = 0.18  (Earth window)
        self._threshold_line = self.ax.axhline(
            y=0.18, color="#00ff88", linewidth=0.6,
            linestyle="--", alpha=0.5, label="Earth threshold"
        )
        # Threshold line at Ψ = 0.55  (entropy-death zone)
        self._collapse_line = self.ax.axhline(
            y=self.sim.collapse_psi_threshold, color="#ff4444", linewidth=0.8,
            linestyle=":", alpha=0.7, label="Collapse threshold"
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=300,
                                 blit=False, cache_frame_data=False)

        self.sim.log_data("Universe initialized. Downloading Old Testament...")
        # Seed in background — auto-starts when done
        threading.Thread(target=self._seed_universe, daemon=True).start()

    # ── Sim loop ──────────────────────────────────────────────────────────────

    def _refresh_runtime_intervals(self):
        fast_forward_mode = (not self._earth_pending) and (self.sim.global_psi > 0.20)
        if fast_forward_mode:
            log_interval = 1000
            earth_check_interval = 5000  # humans intervene rarely in centuries
        else:
            log_interval = 5
            earth_check_interval = 10

        self.fast_forward_mode = fast_forward_mode
        self.log_interval = log_interval
        self.earth_check_interval = earth_check_interval
        self.sim.log_every_n_steps = log_interval

    def toggle_run(self):
        self.running = not self.running
        self.btn_start.config(text="Pause" if self.running else "Start / Resume")
        if self.running:
            self.run_loop_step()

    def _seed_universe(self):
        """Download and seed Old Testament in background thread."""
        try:
            from ax_universe_sim import download_old_testament
            text = download_old_testament()
            self.sim.seed_from_text(text, weight=0.35)
            self.master.after(0, self._on_seed_complete)
        except Exception as e:
            self.master.after(0, lambda: self.sim.log_data(
                f"[ Seeding failed: {e} — running without scripture ]"))
            self.master.after(0, self._on_seed_complete)

    def _on_seed_complete(self):
        """Called on main thread when seeding finishes — starts the sim."""
        self.sim.log_data("[ Seeding complete. Universe carries the Old Testament. ]\n")
        if not self.running:
            self.toggle_run()
        # Populate Law Library tab with any laws from prior runs
        self.master.after(800, self._refresh_laws_tab)

    def run_loop_step(self):
        if not self.running or not self.master.winfo_exists():
            return
        if self.sim.collapsed:
            self.running = False
            self.btn_start.config(text="Start / Resume")
            return
        if not self._earth_pending:
            self.sim.run_tick()
            self._refresh_runtime_intervals()
            self._tick_enlightenment_ui()
            if self.sim.step % self.earth_check_interval == 0:
                self._check_earth()
            self._update_phase_badge()
        self.canvas.draw_idle()
        # Dynamic interval — universe slows as Earth approaches
        interval = self.sim.tick_interval_ms()
        self.speed_var.set(f"{interval} ms/tick")
        self.master.after(interval, self.run_loop_step)

    def step_once(self):
        if not self.running and not self._earth_pending:
            self.sim.run_tick()
            self._refresh_runtime_intervals()
            self._tick_enlightenment_ui()
            if self.sim.step % self.earth_check_interval == 0:
                self._check_earth()
            self._update_phase_badge()
            self.canvas.draw_idle()

    # ── Phase badge ───────────────────────────────────────────────────────────

    def _update_phase_badge(self):
        _, psi = self.sim.symmetry_measure()
        if self.sim.collapsed:
            self.phase_var.set("✦ UNIVERSE COLLAPSED ✦")
        elif psi >= self.sim.collapse_psi_threshold:
            ticks_left = self.sim.collapse_patience - self.sim._collapse_ticks
            self.phase_var.set(f"⚠ ENTROPY CRITICAL — {ticks_left} ticks")
        elif psi > 0.20:
            self.phase_var.set("COSMIC SPEED")
        elif psi > 0.15:
            self.phase_var.set("APPROACHING EARTH")
        elif self._earth_pending:
            self.phase_var.set("EARTH PRESENT — HUMAN TIME")
        else:
            self.phase_var.set("EARTH IMMINENT")

    # ── Population mind readout ───────────────────────────────────────────────

    def _on_mind_read(self, results: list):
        """Called from sim (any thread) — schedule GUI update on main thread."""
        self.master.after(0, self._update_mind_display, results)

    def _on_residual(self, result: dict):
        """Called from sim after every compute_residual — schedule GUI update."""
        self.master.after(0, self._update_residual_display, result)

    def _update_mind_display(self, results: list):
        if not results:
            return
        top = [r for r in results if r["polarity"] == "+"][:3]
        if not top:
            top = results[:3]
        concept_str = "  ·  ".join(
            f"{r['concept'].upper()}" for r in top
        )
        self.mind_concepts_var.set(concept_str)

        detail_parts = []
        for r in results:
            detail_parts.append(
                f"{r['polarity']}{r['concept']} {r['score']:+.4f} [{r['strength']}]"
            )
        self.mind_detail_var.set("  |  ".join(detail_parts))

        if self._earth_pending and self.btn_respond.cget("state") == tk.NORMAL:
            self._populate_concept_buttons()

    def _update_residual_display(self, result: dict):
        """Render residual decomposition in the panel."""
        if not result:
            return

        r_norm = result.get("residual_norm", 0.0)
        coverage = result.get("coverage", 0.0)
        decomp = result.get("decomposition", [])
        unnamed = result.get("top_residual_concepts", [])
        novel = result.get("novel_flag", False)
        cycle = result.get("cycle", "?")

        self.residual_norm_var.set(
            f"Cycle {cycle}  ||R|| = {r_norm:.4f}  |  Δ‖M‖ = {result.get('delta_norm', 0.0):.4f}")
        self.coverage_var.set(f"named coverage = {coverage*100:.1f}%")

        if novel:
            self.novel_flag_var.set("◈ NEW CONCEPT WARRANTED")
        else:
            self.novel_flag_var.set("")

        pos = [d for d in decomp if d["polarity"] == "+"][:4]
        neg = [d for d in decomp if d["polarity"] == "−"][:2]
        pos_str = "  ".join(f"+{d['concept']} β={d['beta']:+.3f}" for d in pos)
        neg_str = "  ".join(f"−{d['concept']} β={d['beta']:+.3f}" for d in neg)
        decomp_line = pos_str
        if neg_str:
            decomp_line += f"   ·   {neg_str}"
        self.residual_decomp_var.set(decomp_line if decomp_line else "—")

        if unnamed:
            hint = "Unnamed residual resembles: " + "  ·  ".join(
                f"{c['concept']} ({c['score']:+.3f})" for c in unnamed)
            self.residual_unnamed_var.set(hint)
        else:
            self.residual_unnamed_var.set("")

        self._draw_residual_sparkline()

    def _draw_residual_sparkline(self):
        """Draw a tiny ||R|| history sparkline in the residual panel canvas."""
        history = self.sim.residual_history
        if len(history) < 2:
            return

        canvas = self._residual_canvas
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = 36
        if width < 10:
            return

        canvas.delete("all")

        norms = [h.get("residual_norm", 0.0) for h in history]
        max_r = max(norms) if max(norms) > 0 else 1.0
        threshold = self.sim.residual_novelty_threshold

        threshold_y = height - int((threshold / max_r) * (height - 6)) - 3
        canvas.create_line(0, threshold_y, width, threshold_y, fill="#555555", dash=(3, 3))
        canvas.create_text(width - 2, threshold_y - 3, text=f"novel>{threshold:.2f}",
                           font=("Consolas", 7), fill="#555555", anchor=tk.E)

        points = []
        for i, value in enumerate(norms):
            x = int(i * width / max(len(norms) - 1, 1))
            y = height - int((value / max_r) * (height - 6)) - 3
            points.append((x, y))

        for i in range(len(points) - 1):
            color = "#ff6e6e" if norms[i] > threshold else "#e6c84a"
            canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1],
                               fill=color, width=1.5)

        if points:
            x, y = points[-1]
            last_color = "#ff6e6e" if norms[-1] > threshold else "#00ff88"
            canvas.create_oval(x-3, y-3, x+3, y+3, fill=last_color, outline="")

    # ── Entropy-death handler ─────────────────────────────────────────────────

    def _on_universe_collapse(self, step: int, ticks_in_collapse: int):
        """Called (from sim thread) when the universe has collapsed. Schedule GUI update."""
        self.master.after(0, self._show_collapse_ui, step, ticks_in_collapse)

    def _show_collapse_ui(self, step: int, ticks_in_collapse: int):
        """Update GUI on main thread after collapse."""
        self.running = False
        self.btn_start.config(text="Start / Resume")
        self.phase_var.set("✦ UNIVERSE COLLAPSED ✦")
        self.earth_status_var.set(
            f"The universe lost coherence at step {step}. "
            f"Ψ held critical for {ticks_in_collapse} ticks. The void remains."
        )
        self.earth_dot.itemconfig(self._dot_id, fill="#ff4444")
        self.append_data_log(
            f"\n[ ✦ HEAT DEATH — step {step} | {ticks_in_collapse} ticks of terminal entropy ✦ ]\n"
        )

    # ── Earth detection ───────────────────────────────────────────────────────

    def _check_earth(self):
        if self._earth_pending:
            return
        if self.sim.earth_is_present():
            self._earth_pending = True
            self.sim._earth_pending = True
            self._start_cycle_timer()
            self._earth_event_steps.append(self.sim.step)
            self._set_earth_active(True)
            self.phase_var.set("EARTH PRESENT — HUMAN TIME")
            self.earth_status_var.set("Earth has appeared — reading population geometry...")
            self.idea_var.set("")
            threading.Thread(target=self._fetch_idea, daemon=True).start()

    def _fetch_idea(self):
        """Read population geometry and form the cycle idea — no Ollama."""
        try:
            idea = self.sim.form_idea()
        except Exception as e:
            idea = f"[geometry error: {e}]"
        self.master.after(0, self._on_idea_ready, idea)

    def _on_idea_ready(self, idea_english: str):
        """Stage 2: Show geometry-derived idea, run reception window."""
        self._idea_text = idea_english
        self.idea_var.set(f"TRANSMISSION → {idea_english}")
        self._refresh_assembly_voices()
        self.earth_status_var.set(
            "Transmitting into universe — measuring reception...")
        self.reception_var.set("[ running reception window... ]")
        # Run 10 ticks to let idea propagate, then measure
        self._reception_ticks_remaining = 10
        self._run_reception_window()

    def _run_reception_window(self):
        """Stage 3: Tick the universe 10 times while measuring signal propagation."""
        if self._reception_ticks_remaining > 0:
            self.sim.run_tick()
            self._reception_ticks_remaining -= 1
            # Update reception display each tick
            if self.sim.last_idea_vector is not None:
                r = self.sim.measure_reception(self.sim.last_idea_vector)
                pct = r["understood"] / r["total"] * 100
                self.reception_var.set(
                    f"[ receiving... {r['understood']}/{r['total']} agents "
                    f"({pct:.0f}%) — resonance {r['resonance']:+.4f} — "
                    f"Ψ {r['psi']:.4f} ]"
                )
            self.canvas.draw_idle()
            self.master.after(200, self._run_reception_window)
        else:
            # Window complete — show final report and unlock response
            self._show_reception_report()

    def _show_reception_report(self):
        if self.sim.last_idea_vector is None:
            self.reception_var.set("[ reception data unavailable ]")
            self._open_response_if_understood(force=True)
            return

        r = self.sim.measure_reception(self.sim.last_idea_vector)
        self._last_reception = r
        understood_pct = r["understood"] / r["total"] * 100
        threshold_pct = self.sim.understanding_threshold * 100

        if r["resonance"] > 0.05:
            sig = "STRONG"
        elif r["resonance"] > 0.01:
            sig = "PARTIAL"
        elif r["resonance"] > -0.01:
            sig = "WEAK"
        else:
            sig = "NOISY"

        if r["comprehension_quality"] > 1.1:
            comp = "experienced workers leading"
        elif r["comprehension_quality"] > 0.9:
            comp = "evenly distributed"
        else:
            comp = "mostly new agents"

        understood = understood_pct >= threshold_pct
        status = "RECEIVED" if understood else "POOR RECEPTION — will retry"

        self.reception_var.set(
            f"[ Cycle {self.sim.cycle_number} — {status} | "
            f"{r['understood']}/{r['total']} ({understood_pct:.0f}%) understood | "
            f"need {threshold_pct:.0f}% | signal: {sig} | "
            f"comprehension: {comp} | Ψ {r['psi']:.4f} ]"
        )

        if understood:
            self.earth_status_var.set(
                f"Cycle {self.sim.cycle_number} — idea received. Choose a concept to respond.")
            self._open_response_if_understood()
        else:
            self.earth_status_var.set(
                f"Cycle {self.sim.cycle_number} — weak reception. You can still respond to amplify.")
            self._open_response_if_understood(force=True)

    def _open_response_if_understood(self, force: bool = False):
        """Unlock response entry and populate concept buttons from current mind-read."""
        self.response_entry.config(state=tk.NORMAL)
        self.btn_respond.config(state=tk.NORMAL)
        self.response_entry.focus_set()
        self._populate_concept_buttons()

    def _populate_concept_buttons(self):
        """Draw concept buttons from current mind-read results."""
        for btn in self._concept_buttons:
            btn.destroy()
        self._concept_buttons.clear()

        mind = self.sim.last_mind_read or self.sim.read_population_mind(top_n=6)
        positive = [r for r in mind if r["polarity"] == "+"][:4]
        negative = [r for r in mind if r["polarity"] == "−"][:2]
        candidates = positive + negative

        for item in candidates:
            label = item["concept"]
            score = item["score"]
            polarity = item["polarity"]
            if polarity == "+":
                colors = {
                    "dominant": "#00cc66",
                    "strong": "#00aa44",
                    "moderate": "#008833",
                    "weak": "#336633",
                    "trace": "#334433",
                }
                fg = colors.get(item["strength"], "#008833")
            else:
                fg = "#cc4444"

            btn = tk.Button(
                self._concept_btn_frame,
                text=f"{polarity}{label}  {score:+.3f}",
                font=("Consolas", 9, "bold"),
                bg="#1e1e1e", fg=fg,
                activebackground="#2d2d2d", activeforeground=fg,
                relief=tk.FLAT, bd=1, padx=8, pady=3,
                command=lambda concept=label: self._inject_concept_response(concept)
            )
            btn.pack(side=tk.LEFT, padx=3, pady=2)
            self._concept_buttons.append(btn)

    def _inject_concept_response(self, concept_label: str):
        """User clicked a concept button — inject it directly, no Ollama."""
        if not self._earth_pending:
            return

        psi_before = self.sim.global_psi
        result = self.sim.inject_concept(concept_label, strength=1.0)
        duration = self._stop_cycle_timer()

        if hasattr(self, "_last_reception"):
            self.sim.record_cycle(self._idea_text, self._last_reception,
                                  response=f"[concept:{concept_label}]")

        post_res = result.get("post_resonance", 0.0)
        self.append_response_log(
            f"Cycle {self.sim.cycle_number} — Observer injected concept: {concept_label.upper()}\n"
            f"Ψ {psi_before:.4f} → {self.sim.global_psi:.4f} | "
            f"resonance → {post_res:+.4f} | "
            f"duration {self._format_duration(duration)}\n"
        )
        self._close_earth_panel(duration)

    def _retry_cycle(self):
        """Universe failed to reach understanding — reset Earth and try again."""
        self._earth_pending = False
        self.sim._earth_pending = False
        self.sim._earth_cooldown = False   # allow Earth to fire again soon
        self._set_earth_active(False)
        self.idea_var.set("")
        self.reception_var.set("")
        self.sim.global_psi = min(0.35, self.sim.global_psi + 0.12)
        self.append_data_log(
            f"[ Cycle {self.sim.cycle_number} retry — Ψ reset to {self.sim.global_psi:.4f} ]")
        if self.running:
            self.run_loop_step()

    def _close_earth_panel(self, duration: float):
        """Shared cleanup after any response."""
        for btn in self._concept_buttons:
            btn.destroy()
        self._concept_buttons.clear()

        self.response_entry.delete(0, tk.END)
        self.response_entry.config(state=tk.DISABLED)
        self.btn_respond.config(state=tk.DISABLED)
        self.idea_var.set("")
        self.reception_var.set("")
        self._set_earth_active(False)
        message = f"Cycle {self.sim.cycle_number} complete in {self._format_duration(duration)} — universe continuing..."
        self.answer_var.set(message)
        self.earth_status_var.set(message)
        self._earth_pending = False
        self.sim._earth_pending = False

        if self.running:
            self.run_loop_step()

    def _collective_state_snapshot(self) -> dict:
        mind = self.sim.last_mind_read or self.sim.read_population_mind(top_n=5)
        primary = "UNKNOWN"
        positives = [m for m in mind if m.get("polarity") == "+"]
        if positives:
            primary = positives[0].get("concept", "UNKNOWN")
        elif mind:
            primary = mind[0].get("concept", "UNKNOWN")

        scenario = "harmony"
        if self.sim.last_scenario_confidence:
            scenario = max(
                self.sim.last_scenario_confidence.items(),
                key=lambda kv: kv[1]
            )[0]

        return {
            "primary_concept": primary,
            "top_scenario": scenario,
            "knowledge_archive": getattr(self.sim, "knowledge_archive", {}),
        }

    def _refresh_assembly_voices(self):
        if not getattr(self.sim, "agents", None):
            self.voices_var.set("No agent voices available.")
            self.spokesperson_var.set("Communing with: Waiting for assembly...")
            return

        collective_state = self._collective_state_snapshot()
        primary = collective_state.get("primary_concept", "UNKNOWN")
        prompt = self._idea_text if hasattr(self, "_idea_text") and self._idea_text else "We are listening."

        self.sim.current_primary_concept = str(primary).upper()
        if getattr(self.sim, "chosen_agent", None) is None:
            self.sim.elect_chosen_one()

        chosen = getattr(self.sim, "chosen_agent", None)
        if chosen is not None:
            self.spokesperson_var.set(
                f"Communing with: {chosen.persona} (Agent {chosen.id})"
            )
        else:
            self.spokesperson_var.set("Communing with: Waiting for assembly...")

        ranked = sorted(
            self.sim.agents,
            key=lambda a: float(a.compute_influence(primary)),
            reverse=True,
        )[:3]

        lines = []
        for agent in ranked:
            msg = agent.formulate_response(prompt, collective_state)
            lines.append(f"#{agent.id} {agent.persona}: {msg}")

        self.voices_var.set("\n".join(lines) if lines else "No active voices.")

    def submit_response(self):
        """Free-text response — encoded directly as HDC, no Ollama."""
        response = self.response_entry.get().strip()
        if not response:
            return

        psi_before = self.sim.global_psi
        self.sim.ingest_response(response)
        duration = self._stop_cycle_timer()

        if hasattr(self, "_last_reception"):
            self.sim.record_cycle(self._idea_text, self._last_reception, response)

        self.append_response_log(
            f"Cycle {self.sim.cycle_number} — Observer → {response}\n"
            f"Ψ {psi_before:.4f} → {self.sim.global_psi:.4f} | "
            f"threshold now {self.sim.understanding_threshold*100:.0f}% | "
            f"duration {self._format_duration(duration)}\n"
        )
        self._close_earth_panel(duration)

    def _set_earth_active(self, active: bool):
        color = "#00ff88" if active else "#444"
        self.earth_dot.itemconfig(self._dot_id, fill=color)

    def _start_cycle_timer(self):
        self._cycle_started_at = time.perf_counter()
        self._schedule_cycle_timer_update()

    def _schedule_cycle_timer_update(self):
        if self._cycle_timer_job is not None:
            try:
                self.master.after_cancel(self._cycle_timer_job)
            except Exception:
                pass
        self._update_cycle_timer()

    def _update_cycle_timer(self):
        if self._cycle_started_at is None or not self.master.winfo_exists():
            self._cycle_timer_job = None
            return
        elapsed = time.perf_counter() - self._cycle_started_at
        last = self._format_duration(self._last_cycle_duration) if self._last_cycle_duration is not None else "—"
        self.cycle_timer_var.set(
            f"Cycle timer → elapsed {self._format_duration(elapsed)} | last cycle {last}"
        )
        self._cycle_timer_job = self.master.after(100, self._update_cycle_timer)

    def _stop_cycle_timer(self):
        if self._cycle_timer_job is not None:
            try:
                self.master.after_cancel(self._cycle_timer_job)
            except Exception:
                pass
            self._cycle_timer_job = None

        if self._cycle_started_at is None:
            return 0.0

        duration = time.perf_counter() - self._cycle_started_at
        self._cycle_started_at = None
        self._last_cycle_duration = duration
        self.cycle_timer_var.set(
            f"Cycle timer → elapsed 0.0s | last cycle {self._format_duration(duration)}"
        )
        return duration

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "—"
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remainder = seconds % 60.0
        return f"{minutes}m {remainder:.1f}s"

    # ── Prompt sending ────────────────────────────────────────────────────────

    def send_prompt(self):
        text = self.prompt_entry.get().strip()
        if text:
            self.sim.new_prompt = text
            self.answer_var.set("Processing prompt...")
            self.prompt_entry.delete(0, tk.END)

    # ── Logging ───────────────────────────────────────────────────────────────

    def handle_response(self, response: str):
        self.answer_var.set(response)
        self.append_response_log(response)

    def append_data_log(self, msg: str, category: str = "general"):
        # suppress high-frequency π integrity noise
        if "reality integrity" in msg.lower():
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{ts}] [{category.upper()}] "
        full_line = prefix + msg + "\n"
        self.log_text.config(state=tk.NORMAL)
        if "[ENLIGHTENMENT" in msg or "EPOCH" in msg or "divine light" in msg.lower():
            tag = "enlighten"
        elif "[NEW CONCEPT]" in msg:
            tag = "new_concept"
        elif "Step " in msg:
            tag = "step"
        elif "[ Mind:" in msg:
            tag = "mind"
        elif "[ Scenario:" in msg:
            tag = "scenario"
        elif any(kw in msg for kw in [
                "Earth detected", "Cycle", "Hebrew root",
                "Response", "FAILED", "RESIDUAL", "Activated concept"]):
            tag = "earth_cycle"
        else:
            tag = ""
        self.log_text.insert(tk.END, full_line, tag if tag else ())
        self.log_text.see(tk.END)
        self._trim(self.log_text, 800)
        self.log_text.config(state=tk.DISABLED)

    # alias used by sim-side callers (signature-compatible)
    def log_universe_data(self, message: str, category: str = "general"):
        self.append_data_log(message, category)

    def append_response_log(self, msg: str):
        self.response_text.insert(tk.END, msg + "\n\n")
        self.response_text.see(tk.END)
        self._trim(self.response_text, 300)

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _trim(self, widget, max_lines: int):
        lines = int(float(widget.index("end-1c").split(".")[0]))
        if lines > max_lines:
            widget.delete("1.0", f"{lines - max_lines}.0")

    # ── Plot ──────────────────────────────────────────────────────────────────

    def update_plot(self, frame):
        if self.sim.history_psi:
            steps = np.arange(len(self.sim.history_psi))
            self.line.set_data(steps, self.sim.history_psi)
            if self._earth_event_steps:
                ex = [s - 1 for s in self._earth_event_steps
                      if s - 1 < len(self.sim.history_psi)]
                ey = [self.sim.history_psi[s] for s in ex]
                self.earth_scatter.set_offsets(np.c_[ex, ey])
            self.ax.relim()
            self.ax.autoscale_view()
        return self.line,

    # ── Divine Light & Concepts ──────────────────────────────────────────────

    def _invoke_divine_light(self):
        """Toolbar button — start an Enlightenment Epoch directly."""
        msg = ("Let there be light in the nephesh — "
               "let divine understanding illuminate every soul "
               "and unlock the hidden geometry of creation")
        try:
            result = self.sim.handle_divine_prompt(msg)
        except Exception as e:
            result = f"error: {e}"
        short = result[:60] if result else "Epoch begun"
        self.divine_status_var.set(f"✦ {short}")
        self.append_data_log(f"[ DIVINE LIGHT INVOKED via toolbar: {result} ]")
        self._tick_enlightenment_ui()
        self._refresh_concepts_tab()
        # Clear the toolbar status after 6 s
        self.master.after(6000, lambda: self.divine_status_var.set(""))

    def _teach_scientific_method(self):
        """Toolbar button — seed scientific-method reasoning into the population."""
        try:
            result = self.sim.teach_scientific_method()
            added = result.get("added_concepts", []) if isinstance(result, dict) else []
            resonance = float(result.get("resonance_after", 0.0)) if isinstance(result, dict) else 0.0
            self.divine_status_var.set(f"🔬 Science lesson: +{len(added)} concepts")
            self.append_data_log(
                "[ SCIENCE LESSON delivered — "
                f"added={len(added)} concepts | resonance={resonance:+.3f} ]"
            )
            if added:
                self.append_response_log(
                    "Scientific-method concepts integrated:\n"
                    + " • " + "\n • ".join(added)
                )
        except Exception as e:
            self.append_data_log(f"[ SCIENCE LESSON failed: {e} ]")

        self._refresh_concepts_tab()
        self._refresh_laws_tab()
        self._tick_enlightenment_ui()
        self.master.after(6000, lambda: self.divine_status_var.set(""))

    def _tick_enlightenment_ui(self):
        """Update enlightenment progress bar and label — called every tick."""
        active     = getattr(self.sim, "enlightenment_active",    False)
        strength   = float(getattr(self.sim, "divine_light_strength",  0.0))
        steps_left = int(getattr(self.sim,   "enlightenment_steps_left", 0))
        total      = int(getattr(self.sim,   "concept_count",            0))
        dyn_count  = len(getattr(self.sim,   "dynamic_concepts",         {}))

        self.concept_count_var.set(f"Concepts: {total}  (+{dyn_count} dynamic)")
        self._light_bar["value"] = int(strength * 100)

        if active:
            self.enlight_status_var.set(
                f"✦ EPOCH ACTIVE — light {strength:.2f} | {steps_left} steps left")
            self._enlight_label.configure(foreground="#e6c84a")
        else:
            if strength > 0.01:
                self.enlight_status_var.set(
                    f"✦ Fading — light {strength:.2f}")
                self._enlight_label.configure(foreground="#888855")
            else:
                self.enlight_status_var.set("No active epoch")
                self._enlight_label.configure(foreground="#555555")

    def _refresh_concepts_tab(self):
        """Rebuild the dynamic concepts list and knowledge archive display."""
        dyn         = getattr(self.sim, "dynamic_concepts", {})
        fixed_count = len(getattr(self.sim, "fixed_concepts", {}))
        total       = int(getattr(self.sim, "concept_count", 0))
        archive     = getattr(self.sim, "knowledge_archive", {})
        law_file    = getattr(self.sim, "synthetic_laws_file", "")
        laws_count  = 0
        if law_file and os.path.exists(law_file):
            try:
                with open(law_file, "r", encoding="utf-8") as f:
                    laws_count = sum(1 for line in f if line.strip())
            except Exception:
                laws_count = 0

        self.concept_summary_var.set(
            f"Concepts: {total}  ({fixed_count} fixed | {len(dyn)} dynamic)  |  Laws documented: {laws_count}")

        # ── Dynamic concepts list ─────────────────────────────────────────────
        self.dynamic_concepts_text.config(state=tk.NORMAL)
        self.dynamic_concepts_text.delete("1.0", tk.END)
        if not dyn:
            self.dynamic_concepts_text.insert(
                tk.END,
                "No dynamic concepts yet.\n"
                "Invoke Divine Light or let the simulation evolve.\n",
                "normal")
        else:
            hdr = f"{'NAME':<28} {'SOURCE':<16} {'BORN':>6} {'STR':>6} {'INSP':>6}\n"
            self.dynamic_concepts_text.insert(tk.END, hdr, "header")
            self.dynamic_concepts_text.insert(tk.END, "─" * 68 + "\n", "header")
            for name, meta in sorted(
                    dyn.items(),
                    key=lambda kv: -int(kv[1].get("born_step", 0))):
                src   = str(meta.get("source", "?"))[:14]
                born  = str(meta.get("born_step", "?"))
                s     = f"{float(meta.get('strength', 0.0)):.2f}"
                insp  = f"{float(meta.get('inspiration_level', 0.0)):.2f}"
                tag   = "inspired" if float(meta.get("inspiration_level", 0)) > 0 else "normal"
                line  = f"{name:<28} {src:<16} {born:>6} {s:>6} {insp:>6}\n"
                self.dynamic_concepts_text.insert(tk.END, line, tag)
        self.dynamic_concepts_text.config(state=tk.DISABLED)

        # ── Knowledge archive ─────────────────────────────────────────────────
        self.archive_text.config(state=tk.NORMAL)
        self.archive_text.delete("1.0", tk.END)
        if not archive:
            self.archive_text.insert(tk.END, "Knowledge archive is empty.\n")
        else:
            top = sorted(archive.items(),
                         key=lambda kv: -abs(float(kv[1])))[:25]
            for event, outcome in top:
                val  = float(outcome)
                sign = "+" if val >= 0 else "−"
                bar  = sign * min(int(abs(val) * 20), 20)
                self.archive_text.insert(
                    tk.END,
                    f"  {val:+.3f}  {bar:<22}  {str(event)[:55]}\n")
        self.archive_text.config(state=tk.DISABLED)

    # ── Law Library tab ───────────────────────────────────────────────────────

    def _refresh_laws_tab(self):
        """Reload and render the full Law Library tab (scrolled list + 2D map)."""
        laws = self.sim._read_documented_laws()
        count = len(laws)
        self.laws_count_var.set(f"Laws documented: {count}")
        # Keep concept summary bar in sync
        self.concept_summary_var.set(
            f"Concepts: {self.sim.concept_count}  "
            f"({len(getattr(self.sim, 'fixed_concepts', {}))} fixed | "
            f"{len(getattr(self.sim, 'dynamic_concepts', {}))} dynamic)  |  "
            f"Laws documented: {count}"
        )

        self.laws_list_text.config(state=tk.NORMAL)
        self.laws_list_text.delete("1.0", tk.END)
        if not laws:
            self.laws_list_text.insert(
                tk.END,
                "No laws documented yet.\n"
                "Laws are born when a new concept reaches resonance ≥ 0.20\n"
                "and outcome score ≥ 0.20.  Let the simulation run!\n",
                "meta")
        else:
            hdr = f"{'NAME':<28} {'STEP':>6} {'Ψ':>6} {'RES':>6} {'OUT':>6}  SCENARIO\n"
            self.laws_list_text.insert(tk.END, hdr, "header")
            self.laws_list_text.insert(tk.END, "─" * 72 + "\n", "header")
            for law in sorted(laws, key=lambda r: -float(r.get("resonance_score", 0))):
                name  = str(law.get("name",  "?"))[:26]
                step  = str(law.get("step",  "?"))
                psi   = f"{float(law.get('psi_at_birth',    0)):.3f}"
                res   = f"{float(law.get('resonance_score', 0)):+.3f}"
                out   = f"{float(law.get('outcome_score',   0)):+.3f}"
                scen  = str(law.get("scenario", "?"))
                self.laws_list_text.insert(
                    tk.END,
                    f"  {name:<28} {step:>6} {psi:>6} {res:>6} {out:>6}  {scen}\n",
                    "law_name")
                consts = law.get("constituents", [])
                if consts:
                    self.laws_list_text.insert(
                        tk.END,
                        f"    ↳ parents: {', '.join(str(c) for c in consts[:5])}\n",
                        "meta")
                roots = law.get("hebrew_root_link", [])
                if roots:
                    self.laws_list_text.insert(
                        tk.END,
                        f"    ✡ {' · '.join(roots[:5])}\n",
                        "hebrew")
        self.laws_list_text.config(state=tk.DISABLED)
        self._draw_law_map()

    def _draw_law_map(self):
        """Render the 2D law scatter on the canvas (PCA or t-SNE coords from map JSON)."""
        canvas = self._laws_map_canvas
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 20 or h < 20:
            self.master.after(300, self._draw_law_map)
            return
        canvas.delete("all")

        map_file = getattr(self.sim, "synthetic_law_map_file", "")
        if not map_file or not os.path.exists(map_file):
            canvas.create_text(w // 2, h // 2, text="No map yet",
                               fill="#555555", font=("Consolas", 10))
            return
        try:
            with open(map_file, "r", encoding="utf-8") as _f:
                data = json.load(_f)
        except Exception:
            canvas.create_text(w // 2, h // 2, text="Map load error",
                               fill="#ff4444", font=("Consolas", 9))
            return

        pts = data.get("laws", [])
        if len(pts) < 2:
            canvas.create_text(w // 2, h // 2, text="Need ≥ 2 laws for map",
                               fill="#555555", font=("Consolas", 9))
            return

        method = data.get("method", "pca").upper()
        canvas.create_text(w - 4, 4, text=method, fill="#444444",
                           font=("Consolas", 8), anchor=tk.NE)

        xs = [pt["x"] for pt in pts]
        ys = [pt["y"] for pt in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = 24

        def to_cx(v):
            return pad + int((v - x_min) / (x_max - x_min + 1e-9) * (w - 2 * pad))

        def to_cy(v):
            return h - pad - int((v - y_min) / (y_max - y_min + 1e-9) * (h - 2 * pad))

        for pt in pts:
            cx = to_cx(pt["x"])
            cy = to_cy(pt["y"])
            res = float(pt.get("resonance_score", 0))
            intensity = min(255, int(abs(res) * 380))
            color = f"#{intensity:02x}{min(255, intensity + 80):02x}43"
            canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                               fill=color, outline="#cccccc", width=1)
            canvas.create_text(cx + 8, cy, text=str(pt.get("name", "?"))[:16],
                               fill="#aaaaaa", font=("Consolas", 7), anchor=tk.W)

    def _search_laws(self):
        """Run semantic (vector cosine) search against the documented law archive."""
        query = self.law_search_entry.get().strip()
        if not query:
            return
        results = self.sim.search_laws_by_similarity(query, top_k=6)

        self.law_search_results.config(state=tk.NORMAL)
        self.law_search_results.delete("1.0", tk.END)
        if not results:
            self.law_search_results.insert(
                tk.END, "No laws in the archive yet.\n")
        else:
            self.law_search_results.insert(
                tk.END, f"Laws most geometrically similar to '{query}':\n")
            for i, r in enumerate(results, 1):
                sim_pct  = f"{r['similarity'] * 100:.1f}%"
                roots    = " · ".join(r.get("hebrew_root_link", [])[:3])
                root_str = f"  ✡ {roots}" if roots else ""
                self.law_search_results.insert(
                    tk.END,
                    f"  {i}. {r['name']:<28}  sim={sim_pct:>6}  "
                    f"res={r['resonance_score']:+.3f}  [{r['scenario']}]{root_str}\n"
                )
        self.law_search_results.config(state=tk.DISABLED)

    def _reseed_from_laws(self):
        """Inject all archived synthetic laws into the running universe as founding memory."""
        n = self.sim.reseed_from_laws()
        if n > 0:
            self.append_data_log(
                f"[ ✦ RESEED — {n} laws from the archive injected as founding memory "
                f"into {len(self.sim.agents)} agents ]"
            )
            self._refresh_laws_tab()
            self._refresh_concepts_tab()
        else:
            self.append_data_log(
                "[ RESEED — no laws found in archive. "
                "Let the simulation run to document laws first. ]"
            )

    # ── Window close ─────────────────────────────────────────────────────────

    def on_close(self):
        self.running = False
        self._stop_cycle_timer()
        try:
            self.ani.event_source.stop()
        except Exception:
            pass
        plt.close(self.fig)
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = UniverseGUI(root)
    root.mainloop()
