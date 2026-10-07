# Presentation: how we trained SynACK Decide v7

A 25-slide talk for people with no machine-learning background: what the model does, the one-token answer idea,
how training works from the ground up (tokens, cross-entropy, distillation, gradients, backpropagation, LoRA, the
training loop), the pipeline, the version history and the result (59.54 on the Decision Index).

| File | What |
|---|---|
| [SynACK-Decide-v7-how-we-trained-it.html](SynACK-Decide-v7-how-we-trained-it.html) | **The whole talk in one file.** Download it and open it in any browser; it works offline (fonts embedded) |
| [SynACK-Decide-v7-how-we-trained-it.pdf](SynACK-Decide-v7-how-we-trained-it.pdf) | The same slides as a PDF (GitHub shows the PDF in the browser) |
| [SPEAKER_NOTES.md](SPEAKER_NOTES.md) | A plain-language script for every slide |
| [source/](source/) | Slide sources (`deck.json` plus one HTML file per slide) |
| [tools/build_viewer.py](tools/build_viewer.py) | Rebuilds the one-file HTML from `source/` |

**Using the HTML file:** arrow keys, space or a click move between slides; **N** shows the speaker notes under the
slide; **F** is full screen; `#6` at the end of the address opens slide 6. Printing it (Save as PDF) gives one slide per
page. GitHub shows `.html` files as code, so use the download button on the file page (or "Raw", then save), then
open the saved file.

The deck explains the training in more depth than the README. For a written walk-through with the same worked
example, see [docs/HOW_TRAINING_WORKS.md](../docs/HOW_TRAINING_WORKS.md).

Example probabilities and teacher numbers on the slides are illustrative and labelled as such; every other number
comes from the project's logs and result files (see the main [README](../README.md)).
