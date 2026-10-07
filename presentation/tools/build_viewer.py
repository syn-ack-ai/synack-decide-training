"""Build one self-contained HTML file from the deck's slide files: fonts embedded, keyboard/click/swipe navigation,
speaker notes on N, full screen on F, and print-to-PDF gives one slide per page.

  python build_viewer.py DECK_PROJECT_DIR FONT_DIR OUT.html
"""
import base64
import json
import sys
from pathlib import Path

proj, fonts, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
deck = json.loads((proj / "deck.json").read_text())


def font(file, family, weight):
    b64 = base64.b64encode((fonts / file).read_bytes()).decode()
    return f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};font-display:block;src:url(data:font/woff2;base64,{b64}) format('woff2')}}"


faces = "\n".join([font("IBMPlexSans-var.woff2", "IBM Plex Sans", "300 700"),
                   font("IBMPlexMono-400.woff2", "IBM Plex Mono", "400"),
                   font("IBMPlexMono-600.woff2", "IBM Plex Mono", "600")])
slides = "\n".join((proj / "slides" / f"{sid}.html").read_text().strip() for sid in deck["order"])

css = """
* { box-sizing: border-box; margin: 0 }
html, body { height: 100%; background: #03050B; color: #F5F7FF; font-family: 'IBM Plex Sans', Arial, sans-serif }
#deck > section { width: 1920px; height: 1080px; overflow: hidden }
h1 { font-size: 96px; font-weight: 600; line-height: 1.1 } h2 { font-size: 64px; font-weight: 600; line-height: 1.15 }
h3 { font-size: 44px; font-weight: 600; line-height: 1.2 } p { font-size: 32px; line-height: 1.4 }
aside { display: none }
table { border-collapse: collapse; width: 100% }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #1E3157; vertical-align: top }
th { font-weight: 600; color: #AAB7D0 }
x-shape { display: block; flex: none }
x-shape[kind="arrow-right"] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%) }
@media screen {
  html, body { overflow: hidden }
  #stage { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center }
  #deck { position: relative; width: 1920px; height: 1080px; flex: none; transform-origin: center center }
  #deck > section { position: absolute; left: 0; top: 0 }
  #deck > section:not(.active) { display: none !important }
  #bar { position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%); display: flex; gap: 6px; align-items: center;
         padding: 6px 8px; border-radius: 10px; background: rgba(14, 23, 48, 0.88); border: 1px solid #1E3157;
         font: 14px 'IBM Plex Mono', monospace; color: #AAB7D0; opacity: 0.25; transition: opacity 0.2s; z-index: 5 }
  #bar:hover, body.notes #bar { opacity: 1 }
  #bar button { font: inherit; color: #F5F7FF; background: #10254A; border: 1px solid #1E3157; border-radius: 6px;
                padding: 4px 10px; cursor: pointer }
  #bar button:hover { background: #13305F }
  #count { min-width: 64px; text-align: center }
  #notes { position: fixed; left: 0; right: 0; bottom: 0; max-height: 34vh; overflow: auto; padding: 18px 24px 64px;
           background: rgba(7, 11, 22, 0.96); border-top: 2px solid #21C8FF; color: #DDE4F3;
           font: 17px/1.55 'IBM Plex Sans', Arial, sans-serif; display: none; z-index: 4 }
  body.notes #notes { display: block }
  #notes b { color: #21C8FF; font: 600 13px 'IBM Plex Mono', monospace; letter-spacing: 2px; display: block; margin-bottom: 6px }
}
@media print {
  @page { size: 1920px 1080px; margin: 0 }
  html, body { background: #070B16 }
  #deck > section { position: relative; break-after: page }
  #bar, #notes { display: none }
}
"""

js = """
const deck = document.getElementById('deck'), slides = [...deck.children], count = document.getElementById('count'),
      notes = document.getElementById('notestext');
let i = 0;
function fit() {
  const free = document.body.classList.contains('notes') ? 0.66 : 1;
  deck.style.transform = 'scale(' + Math.min(innerWidth / 1920, innerHeight * free / 1080) + ')';
  document.getElementById('stage').style.bottom = document.body.classList.contains('notes') ? '34vh' : '0';
}
function show(n) {
  i = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, k) => s.classList.toggle('active', k === i));
  count.textContent = (i + 1) + ' / ' + slides.length;
  const a = slides[i].querySelector('aside');
  notes.textContent = a ? a.textContent.trim() : 'No notes for this slide.';
  history.replaceState(null, '', '#' + (i + 1));
}
function toggleNotes() { document.body.classList.toggle('notes'); fit(); }
function fullscreen() { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(e.key)) { show(i + 1); e.preventDefault(); }
  else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) { show(i - 1); e.preventDefault(); }
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key === 'n' || e.key === 'N') toggleNotes();
  else if (e.key === 'f' || e.key === 'F') fullscreen();
});
document.getElementById('stage').addEventListener('click', e => show(e.clientX < innerWidth / 3 ? i - 1 : i + 1));
let x0 = null;
addEventListener('touchstart', e => { x0 = e.touches[0].clientX; }, { passive: true });
addEventListener('touchend', e => { if (x0 === null) return; const dx = e.changedTouches[0].clientX - x0;
  if (Math.abs(dx) > 40) show(dx < 0 ? i + 1 : i - 1); x0 = null; });
document.getElementById('prev').onclick = () => show(i - 1);
document.getElementById('next').onclick = () => show(i + 1);
document.getElementById('ntoggle').onclick = toggleNotes;
document.getElementById('fs').onclick = fullscreen;
addEventListener('resize', fit);
fit();
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="How we trained SynACK Decide v7: a 25-slide talk for non-ML audiences. Arrow keys to move, N for speaker notes, F for full screen.">
<title>{deck['title']}</title>
<style>
{faces}
{css}
</style>
</head>
<body>
<div id="stage"><div id="deck">
{slides}
</div></div>
<div id="notes"><b>SPEAKER NOTES</b><div id="notestext"></div></div>
<nav id="bar" aria-label="Slide controls">
<button id="prev" aria-label="Previous slide">&#8249;</button><span id="count">1 / 1</span><button id="next" aria-label="Next slide">&#8250;</button>
<button id="ntoggle">Notes (N)</button><button id="fs">Full screen (F)</button>
</nav>
<script>{js}</script>
</body>
</html>
"""
out.write_text(html)
print(f"{len(deck['order'])} slides -> {out} ({out.stat().st_size // 1024} KB)")
