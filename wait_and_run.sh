#!/bin/bash
# Poll the GPUs; the moment one has enough FREE memory, launch run_level002.sh
# pinned to it (disconnect-proof via nohup), then exit.  Safe to run under nohup
# itself so it keeps waiting after you disconnect.
set -u
cd ~/PERSIST

NEED_MB=${NEED_MB:-40000}     # required free MiB (PERSIST XL headroom); override: NEED_MB=30000 bash wait_and_run.sh
POLL=${POLL:-60}             # seconds between checks

echo "$(date '+%F %T')  waiting for a GPU with >= ${NEED_MB} MiB free (poll ${POLL}s)..."
while true; do
  # pick the GPU with the most free memory
  best_idx=-1; best_free=-1
  while IFS=',' read -r idx free; do
    idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
    if [ "$free" -gt "$best_free" ]; then best_free=$free; best_idx=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  if [ "$best_free" -ge "$NEED_MB" ]; then
    echo "$(date '+%F %T')  GPU ${best_idx} has ${best_free} MiB free -> launching"
    export CUDA_VISIBLE_DEVICES="$best_idx"
    nohup bash run_level002.sh > level002.log 2>&1 &
    echo "$(date '+%F %T')  started run_level002.sh on GPU ${best_idx}, PID $!  (log: level002.log)"
    exit 0
  fi

  echo "$(date '+%F %T')  best is GPU ${best_idx} with ${best_free} MiB free (< ${NEED_MB}); waiting..."
  sleep "$POLL"
done
