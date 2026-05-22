"""
Enregistre la présentation demo_presentation.html en vidéo MP4.

Navigation manuelle slide par slide avec timing précis par fragment :
  - Slide 01 · Intro              : 20s
  - Slide 02 · Graphe (3 frags)   : 6s → frag1 4s → frag2 4s → frag3 5s → avance
  - Slide 03 · Agent              : 20s
  - Slide 04 · Demo live (pause)  : 6s
  - Slide 05 · Juges (4 frags)    : 7s → frag1 3s → frag2 3s → frag3 3s → frag4 5s → avance
  - Slide 06 · Prompt Registry    : 5s → frag1 3s → frag2 2s → frag3 2s → frag4 2s → frag5 4s → avance
  - Slide 07 · GEPA               : 22s
  - Slide 08 · Conclusion         : 20s + 4s de marge finale
  Total estimé : ~2min30

Usage :
  python -m demo_client.record_video
  python demo_client\\record_video.py

Sortie :
  demo_client/artifact/demo_presentation.webm   (toujours)
  demo_client/artifact/demo_presentation.mp4    (si ffmpeg installé)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

HTML_FILE = HERE / "demo_presentation.html"
OUTPUT_DIR = HERE / "artifact"
OUTPUT_WEBM = OUTPUT_DIR / "demo_presentation.webm"
OUTPUT_MP4  = OUTPUT_DIR / "demo_presentation.mp4"

WIDTH  = 1920
HEIGHT = 1080

# ── Scénario de navigation ────────────────────────────────────────────────────
# Chaque étape = (durée en secondes, action, label)
#   action "wait"  → attendre sans bouger (animations en cours)
#   action "next"  → Reveal.next() (avance fragment ou slide suivante)
STEPS: list[tuple[float, str, str]] = [
    # ── Slide 01 · Intro ─────────────────────────────────────────────────────
    # Hero + pills + tagline apparaissent via CSS animate
    (3.0,  "wait", "01 · Intro — apparition hero"),
    (17.0, "wait", "01 · Intro — lecture"),
    (0.0,  "next", "→ Slide 02"),

    # ── Slide 02 · Graphe (3 fragments : 3 stats cards) ──────────────────────
    # iframe graphe + titre apparaissent, compteurs à 0
    (6.0,  "wait", "02 · Graphe — chargement iframe"),
    (0.0,  "next", "02 · frag1 — nœuds (14 996)"),
    (4.0,  "wait", "02 · compteur nœuds animé (1.4s)"),
    (0.0,  "next", "02 · frag2 — relations (28 129)"),
    (4.0,  "wait", "02 · compteur relations animé"),
    (0.0,  "next", "02 · frag3 — layers (11)"),
    (5.0,  "wait", "02 · lecture stats"),
    (0.0,  "next", "→ Slide 03"),

    # ── Slide 03 · Agent ─────────────────────────────────────────────────────
    # Diagramme de flux Question → Agent → Tools → Réponse
    (3.0,  "wait", "03 · Agent — apparition flow"),
    (17.0, "wait", "03 · Agent — lecture flow"),
    (0.0,  "next", "→ Slide 04"),

    # ── Slide 04 · Demo live (pause visuelle) ────────────────────────────────
    (6.0,  "wait", "04 · Demo live — pause"),
    (0.0,  "next", "→ Slide 05"),

    # ── Slide 05 · Juges LLM (radar Chart.js + 4 fragments recommendations) ──
    # Radar anime en 1.8s, score 12%, barres
    (3.0,  "wait", "05 · Juges — radar animation (1.8s)"),
    (5.0,  "wait", "05 · Juges — lecture scores"),
    (0.0,  "next", "05 · frag1 — reco JugeExploration"),
    (3.0,  "wait", "05 · frag1 lecture"),
    (0.0,  "next", "05 · frag2 — reco JugePrecision"),
    (3.0,  "wait", "05 · frag2 lecture"),
    (0.0,  "next", "05 · frag3 — reco JugeRaisonnement"),
    (3.0,  "wait", "05 · frag3 lecture"),
    (0.0,  "next", "05 · frag4 — reco JugeAmeliorations"),
    (5.0,  "wait", "05 · frag4 lecture"),
    (0.0,  "next", "→ Slide 06"),

    # ── Slide 06 · Prompt Registry (diff v1→v2, 5 fragments) ─────────────────
    # Diff apparaît : code v1 visible, lignes v2 cachées
    (5.0,  "wait", "06 · Prompt Registry — lecture v1"),
    (0.0,  "next", "06 · frag1 — ligne supprimée (rouge)"),
    (3.0,  "wait", "06 · frag1 lecture"),
    (0.0,  "next", "06 · frag2 — ajout ligne 1 (vert)"),
    (2.0,  "wait", "06 · frag2 lecture"),
    (0.0,  "next", "06 · frag3 — ajout ligne 2"),
    (2.0,  "wait", "06 · frag3 lecture"),
    (0.0,  "next", "06 · frag4 — ajout ligne 3"),
    (2.0,  "wait", "06 · frag4 lecture"),
    (0.0,  "next", "06 · frag5 — ajout ligne 4 (dernière)"),
    (5.0,  "wait", "06 · lecture diff complète"),
    (0.0,  "next", "→ Slide 07"),

    # ── Slide 07 · GEPA (line Chart.js + pipeline ring) ──────────────────────
    # Line chart anime en 1.9s : Iter0→1→2 score monte
    (3.0,  "wait", "07 · GEPA — line chart animation (1.9s)"),
    (19.0, "wait", "07 · GEPA — lecture pipeline"),
    (0.0,  "next", "→ Slide 08"),

    # ── Slide 08 · Conclusion (3 panels + banner) ────────────────────────────
    (3.0,  "wait", "08 · Conclusion — apparition panels"),
    (17.0, "wait", "08 · Conclusion — lecture"),
    (4.0,  "wait", "08 · Conclusion — marge finale"),
]

TOTAL_S = sum(d for d, _, _ in STEPS)


def record() -> Path:
    from playwright.sync_api import sync_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_tmp_dir = OUTPUT_DIR / "_video_tmp"
    video_tmp_dir.mkdir(parents=True, exist_ok=True)

    html_url = HTML_FILE.as_uri()
    print(f"  Présentation : {html_url}")
    print(f"  Durée totale : {TOTAL_S:.0f}s (~{TOTAL_S/60:.1f}min)")
    print(f"  Résolution   : {WIDTH}×{HEIGHT}")
    print(f"  Étapes       : {len(STEPS)} (slides + fragments)")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--start-maximized",
                f"--window-size={WIDTH},{HEIGHT}",
                "--disable-infobars",
                "--no-default-browser-check",
            ],
        )
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(video_tmp_dir),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = context.new_page()

        print("  Ouverture de la présentation...")
        page.goto(html_url, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector(".reveal.ready", timeout=15_000)
        print("  Reveal.js prêt ✓")

        # Désactive l'auto-avance pour contrôle manuel total
        page.evaluate("() => { Reveal.configure({ autoSlide: 0 }); }")

        # Plein écran
        page.evaluate("() => { document.documentElement.requestFullscreen().catch(() => {}); }")
        time.sleep(1.5)

        print("  Enregistrement en cours...\n")
        elapsed_total = 0.0

        for wait_s, action, label in STEPS:
            if action == "next":
                page.evaluate("() => { Reveal.next(); }")
                print(f"  ▶  {label}")
            if wait_s > 0:
                print(f"     ⏱  {label} ({wait_s}s)")
                time.sleep(wait_s)
                elapsed_total += wait_s

        print(f"\n  ✓ Enregistrement terminé ({elapsed_total:.0f}s)")
        context.close()
        browser.close()

    # Récupère le .webm généré
    webm_files = list(video_tmp_dir.glob("*.webm"))
    if not webm_files:
        raise RuntimeError(f"Aucun fichier .webm trouvé dans {video_tmp_dir}")

    shutil.move(str(webm_files[0]), str(OUTPUT_WEBM))
    shutil.rmtree(video_tmp_dir, ignore_errors=True)

    size_mb = OUTPUT_WEBM.stat().st_size / (1024 * 1024)
    print(f"  Vidéo WebM : {OUTPUT_WEBM} ({size_mb:.1f} MB)")
    return OUTPUT_WEBM


def convert_to_mp4(webm_path: Path) -> Path | None:
    """Convertit .webm en .mp4 via ffmpeg si disponible."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("\n  [INFO] ffmpeg non trouvé — vidéo disponible en .webm uniquement")
        print("         Installer : winget install ffmpeg  (terminal admin)")
        return None

    print("\n  Conversion MP4 via ffmpeg...")
    cmd = [
        ffmpeg, "-y",
        "-i", str(webm_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUTPUT_MP4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = OUTPUT_MP4.stat().st_size / (1024 * 1024)
        print(f"  ✓ MP4 : {OUTPUT_MP4} ({size_mb:.1f} MB)")
        return OUTPUT_MP4
    print(f"  [ERREUR ffmpeg] {result.stderr[-400:]}")
    return None


if __name__ == "__main__":
    if not HTML_FILE.exists():
        print(f"[ERREUR] Présentation introuvable : {HTML_FILE}")
        sys.exit(1)

    print("=" * 65)
    print("  Enregistrement vidéo — GraphAgent Demo")
    print("=" * 65)
    print()

    # Affiche le scénario complet avant de lancer
    print("  SCÉNARIO :")
    for i, (wait_s, action, label) in enumerate(STEPS):
        icon = "▶" if action == "next" else "⏱"
        val  = "" if action == "next" else f" {wait_s}s"
        print(f"  {icon} {label}{val}")
    print(f"\n  Durée totale estimée : {TOTAL_S:.0f}s (~{TOTAL_S/60:.1f}min)")
    print()

    webm = record()
    mp4  = convert_to_mp4(webm)

    print("\n" + "=" * 65)
    print("  RÉSULTAT")
    print("=" * 65)
    if mp4 and mp4.exists():
        print(f"  ✅ MP4  : {mp4}")
    print(f"  ✅ WebM : {webm}")
    print("\n  Lis avec VLC, Windows Media Player, ou importe dans Teams/LinkedIn.")
