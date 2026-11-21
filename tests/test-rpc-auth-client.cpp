#include "ggml-rpc.h"

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    size_t free = 0, total = 0;
    ggml_backend_rpc_get_device_memory(argv[1], 0, &free, &total);
    return total > 0 ? 0 : 1;
}
