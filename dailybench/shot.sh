#!/bin/sh
# Screenshot a page in headless Chrome, under a watchdog.
#
#   ./shot.sh <url> <out.png>
#
# Chrome hangs whenever it is not the only browser on the machine: the debug
# port, the profile singleton lock and a stopped page all block forever, and a
# task then burns its whole wall clock on the picture. This wrapper owns two
# things Chrome cannot be trusted with:
#
#   * a private ``--user-data-dir`` per call, so two shots (or two agents) never
#     share the singleton;
#   * a deadline. After ``SHOT_TIMEOUT`` seconds (25 by default) the whole
#     browser tree is killed with SIGKILL and the script exits 124, like
#     ``timeout(1)``, instead of waiting for a page that will not paint.
#
# ``CHROME_BIN`` overrides the binary (the container sets it to chromium; a
# test points it at a fake).
set -u

timeout=${SHOT_TIMEOUT:-25}
if [ "$#" -lt 2 ]; then
    echo "usage: $0 <url> <out.png>" >&2
    exit 64
fi
url=$1
out=$2
chrome=${CHROME_BIN:-chromium}

if ! command -v "$chrome" >/dev/null 2>&1; then
    echo "shot: браузер не найден: $chrome" >&2
    exit 127
fi

profile=$(mktemp -d "${TMPDIR:-/tmp}/shot.XXXXXX") || exit 1
cleanup() { rm -rf "$profile"; }
trap cleanup EXIT

# Depth-first: kill the descendants before their parent, or an orphan is
# reparented to init and the walk no longer reaches it.
kill_tree() {
    for child in $(ps -Ao pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'); do
        kill_tree "$child"
    done
    kill -9 "$1" 2>/dev/null
}

"$chrome" \
    --headless=new \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --hide-scrollbars \
    --window-size=1280,800 \
    --user-data-dir="$profile" \
    --no-first-run \
    --no-default-browser-check \
    --screenshot="$out" \
    "$url" &
pid=$!

deadline=$(( $(date +%s) + timeout ))
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        kill_tree "$pid"
        wait "$pid" 2>/dev/null
        echo "shot: таймаут ${timeout} с — браузер убит" >&2
        exit 124
    fi
    sleep 0.2
done
wait "$pid"
status=$?

if [ "$status" -ne 0 ]; then
    echo "shot: браузер завершился с кодом $status" >&2
    exit "$status"
fi
if [ ! -s "$out" ]; then
    echo "shot: кадр не записан: $out" >&2
    exit 1
fi
exit 0
