#pragma once

#include "transport.h"
#include <vector>

void rpc_auth_sha256(const void * data, size_t size, uint8_t out[32]);
void rpc_auth_hmac(const uint8_t * key, size_t key_size, const void * data, size_t size, uint8_t out[32]);

struct rpc_auth_key {
    std::vector<uint8_t> bytes;
    rpc_auth_key() = default;
    rpc_auth_key(const rpc_auth_key &) = delete;
    rpc_auth_key & operator=(const rpc_auth_key &) = delete;
    ~rpc_auth_key();
    bool load();
};

bool rpc_authenticate(socket_t & sock, const rpc_auth_key & key, bool server);
