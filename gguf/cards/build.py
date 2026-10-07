"""Build each repo's README.md as header-<repo>.md + body.md (the shared body).
  build.py OUT_DIR  -> OUT_DIR/README-main.md, README-mlx.md, README-gguf.md"""
import sys
from pathlib import Path
D = Path(__file__).parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else D)
body = (D / "body.md").read_text()
for name in ("main", "mlx", "gguf"):
    (out / f"README-{name}.md").write_text((D / f"header-{name}.md").read_text() + body)
    print(out / f"README-{name}.md")
