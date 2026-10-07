#!/bin/sh
# Copy finished per-repo PR files to box every 2 minutes until the collector exits, then a final sync + DONE marker.
while pgrep -f collect_prs.py >/dev/null; do rsync -aq data/prs/ box:prcx/data/prs/; sleep 120; done
rsync -aq data/prs/ box:prcx/data/prs/ && ssh box 'touch ~/prcx/data/COLLECTION_DONE' </dev/null && echo "final sync + DONE marker"
