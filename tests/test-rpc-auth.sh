#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
server=$1
client=$2
multi=$3
multi_script=$4
# Keep temporary files inside the worktree/build working directory.
test_dir=$(mktemp -d "$PWD/rpc-auth.XXXXXX")
port=$((30000 + $$ % 10000))
endpoint="127.0.0.1:$port"
pid=
cleanup() {
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
    rm -rf "$test_dir"
}
trap cleanup EXIT
unset GGML_RPC_KEY_FILE GGML_RPC_NO_COMM
head -c 32 /dev/urandom > "$test_dir/key"
head -c 32 /dev/urandom > "$test_dir/wrong"
chmod 600 "$test_dir/key" "$test_dir/wrong"
start() {
    "$server" --device CPU --host 127.0.0.1 --port "$port" "$@" > "$test_dir/server.log" 2>&1 &
    pid=$!
    for _ in {1..200}; do
        kill -0 "$pid"
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then return; fi
        sleep 0.05
    done
    cat "$test_dir/server.log"
    exit 1
}
stop() {
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    pid=
}
fail_client() {
    local rc=0
    timeout 9 "$@" "$client" "$endpoint" > "$test_dir/client.log" 2>&1 || rc=$?
    if [[ "$rc" == 0 || "$rc" == 124 || "$rc" == 137 ]]; then
        cat "$test_dir/client.log"
        echo "expected prompt authentication failure, got $rc" >&2
        exit 1
    fi
}
start --key-file "$test_dir/key"
GGML_RPC_KEY_FILE="$test_dir/key" timeout 9 "$client" "$endpoint"
fail_client env GGML_RPC_KEY_FILE="$test_dir/wrong"
grep -q '127.0.0.1:.*RPC auth failed' "$test_dir/server.log"
fail_client env -u GGML_RPC_KEY_FILE
kill -0 "$pid"
GGML_RPC_KEY_FILE="$test_dir/key" timeout 9 "$client" "$endpoint"
stop
start
start_time=$SECONDS
fail_client env GGML_RPC_KEY_FILE="$test_dir/key"
(( SECONDS - start_time <= 7 ))
timeout 9 "$client" "$endpoint"
stop
# Invalid key configuration must fail before connecting or listening.
: > "$test_dir/empty"
printf 'too short' > "$test_dir/short"
head -c 4097 /dev/zero > "$test_dir/large"
cp "$test_dir/key" "$test_dir/public"
chmod 600 "$test_dir/empty" "$test_dir/short" "$test_dir/large"
chmod 644 "$test_dir/public"
for invalid in missing empty short large public; do
    fail_client env GGML_RPC_KEY_FILE="$test_dir/$invalid"
    rc=0
    GGML_RPC_KEY_FILE="$test_dir/$invalid" timeout 3 "$server" --device CPU --port "$port" > "$test_dir/invalid.log" 2>&1 || rc=$?
    [[ "$rc" != 124 && "$rc" != 137 ]]
    grep -q 'GGML_RPC_KEY_FILE' "$test_dir/invalid.log"
done
# Existing regression, including the pairwise comm connection, with a shared key.
GGML_RPC_KEY_FILE="$test_dir/key" TMPDIR="$test_dir" timeout 45 bash "$multi_script" "$server" "$multi"
echo 'RPC authentication end-to-end tests passed'
