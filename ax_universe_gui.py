import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import time
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

        earth_frame = ttk.LabelFrame(master, text="Earth Channel — Observer Interface", padding=8)
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
        mind_frame = ttk.LabelFrame(master, text="Population Mind  —  what the agents are expressing", padding=6)
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

        # ── Latest answer ─────────────────────────────────────────────────────
        answer_frame = ttk.LabelFrame(master, text="Latest Answer", padding=5)
        answer_frame.pack(fill=tk.X, expand=False, pady=(5, 0), padx=10)
        self.answer_var = tk.StringVar(value="No answer yet.")
        ttk.Label(answer_frame, textvariable=self.answer_var,
                  wraplength=920, justify=tk.LEFT).pack(fill=tk.X)

        # ── Residual / prayer decompression panel ─────────────────────────────
        residual_frame = ttk.LabelFrame(
            master, text="Residual  —  what the population became beyond your vocabulary", padding=6)
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
        log_frame = ttk.LabelFrame(master, text="Universe Data Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=5, padx=10)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#0d1117", fg="#c9d1d9", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.sim.on_data_log = self.append_data_log
        self.sim.on_response = self.handle_response
        self.sim.on_collapse = self._on_universe_collapse
        self.sim.on_mind_read = self._on_mind_read
        self.sim.on_residual = self._on_residual

        # ── Response log ──────────────────────────────────────────────────────
        response_frame = ttk.LabelFrame(master, text="Assistant Responses", padding=5)
        response_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5), padx=10)
        self.response_text = scrolledtext.ScrolledText(
            response_frame, height=5, bg="#0b1320", fg="#e6edf3", font=("Consolas", 10))
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # ── Plot ──────────────────────────────────────────────────────────────
        plot_frame = ttk.LabelFrame(master, text="Global Ψ Evolution", padding=5)
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

    def run_loop_step(self):
        if not self.running or not self.master.winfo_exists():
            return
        if self.sim.collapsed:
            self.running = False
            self.btn_start.config(text="Start / Resume")
            return
        if not self._earth_pending:
            self.sim.run_tick()
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
        self.log_text.insert(tk.END,
            f"[ Cycle {self.sim.cycle_number} retry — "
            f"Ψ reset to {self.sim.global_psi:.4f} ]\n")
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

    def append_data_log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self._trim(self.log_text, 800)

    def append_response_log(self, msg: str):
        self.response_text.insert(tk.END, msg + "\n\n")
        self.response_text.see(tk.END)
        self._trim(self.response_text, 300)

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
