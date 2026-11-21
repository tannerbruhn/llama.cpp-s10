#include "auth.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static void wipe(void * ptr, size_t size) {
    volatile uint8_t * p = static_cast<volatile uint8_t *>(ptr);
    while (size--) { *p++ = 0; }
}

// SHA-256, implemented directly from FIPS 180-4 sections 4.1.2, 4.2.2 and 6.2.
static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
struct sha256 {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[64] = {};
    size_t used = 0;
    uint64_t length = 0;
    ~sha256() { wipe(this, sizeof(*this)); }
    void compress() {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t * p = block + 4*i;
            w[i] = uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t x = w[i-15], y = w[i-2];
            w[i] = w[i-16] + (rotr(x,7)^rotr(x,18)^(x>>3)) + w[i-7] + (rotr(y,17)^rotr(y,19)^(y>>10));
        }
        uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], v=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = v + (rotr(e,6)^rotr(e,11)^rotr(e,25)) + ((e&f)^(~e&g)) + k[i] + w[i];
            uint32_t t2 = (rotr(a,2)^rotr(a,13)^rotr(a,22)) + ((a&b)^(a&c)^(b&c));
            v=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=v;
        wipe(w, sizeof(w));
    }
    void update(const void * data, size_t size) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        length += size;
        while (size) {
            size_t n = std::min(size, 64-used);
            memcpy(block+used, p, n); used+=n; p+=n; size-=n;
            if (used == 64) { compress(); used = 0; }
        }
    }
    void finish(uint8_t out[32]) {
        uint64_t bits = length*8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (used != 56) { update(&pad, 1); }
        uint8_t tail[8];
        for (int i = 0; i < 8; ++i) { tail[7-i] = uint8_t(bits >> (8*i)); }
        update(tail, 8);
        for (int i = 0; i < 32; ++i) { out[i] = uint8_t(h[i/4] >> (24-8*(i%4))); }
    }
};

void rpc_auth_sha256(const void * data, size_t size, uint8_t out[32]) {
    sha256 s; s.update(data, size); s.finish(out);
}

void rpc_auth_hmac(const uint8_t * key, size_t key_size, const void * data, size_t size, uint8_t out[32]) {
    uint8_t pad[64] = {}, inner[32];
    if (key_size > 64) { rpc_auth_sha256(key, key_size, pad); }
    else { memcpy(pad, key, key_size); }
    for (auto & b : pad) { b ^= 0x36; }
    sha256 s; s.update(pad, 64); s.update(data, size); s.finish(inner);
    for (auto & b : pad) { b ^= 0x36 ^ 0x5c; }
    sha256 t; t.update(pad, 64); t.update(inner, 32); t.finish(out);
    wipe(pad, sizeof(pad)); wipe(inner, sizeof(inner));
}

rpc_auth_key::~rpc_auth_key() { wipe(bytes.data(), bytes.size()); }

bool rpc_auth_key::load() {
    wipe(bytes.data(), bytes.size()); bytes.clear();
    const char * path = std::getenv("GGML_RPC_KEY_FILE");
    if (!path) { return true; }
#ifdef _WIN32
    LOG_ERROR("GGML_RPC_KEY_FILE: RPC authentication is not supported on Windows\n");
    return false;
#else
    int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    struct stat st = {};
    uint8_t buf[4097] = {};
    size_t size = 0;
    bool ok = fd >= 0 && fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && !(st.st_mode & (S_IRGRP | S_IROTH));
    while (ok && size < sizeof(buf)) {
        ssize_t n = read(fd, buf+size, sizeof(buf)-size);
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0) { ok = false; break; }
        if (n == 0) { break; }
        size += size_t(n);
    }
    if (fd >= 0) { close(fd); }
    ok = ok && size >= 16 && size <= 4096;
    if (ok) { bytes.assign(buf, buf+size); }
    wipe(buf, sizeof(buf));
    if (!ok) { LOG_ERROR("GGML_RPC_KEY_FILE '%s': require a readable regular file of 16..4096 bytes, not group/world-readable\n", path); }
    return ok;
#endif
}

static bool random_nonce(uint8_t out[32]) {
#ifdef _WIN32
    (void) out;
    return false;
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return false; }
    size_t done = 0;
    while (done < 32) {
        ssize_t n = read(fd, out+done, 32-done);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { break; }
        done += size_t(n);
    }
    close(fd);
    return done == 32;
#endif
}

static void proof(const rpc_auth_key & key, bool server, const uint8_t * first, const uint8_t * second, uint8_t out[32]) {
    const char * label = server ? "ggml-rpc-auth-v1 server" : "ggml-rpc-auth-v1 client";
    uint8_t data[87]; // 23-byte label, two 32-byte nonces, no terminator
    memcpy(data, label, 23); memcpy(data+23, first, 32); memcpy(data+55, second, 32);
    rpc_auth_hmac(key.bytes.data(), key.bytes.size(), data, sizeof(data), out);
    wipe(data, sizeof(data));
}

static bool equal(const uint8_t a[32], const uint8_t b[32]) {
    volatile uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) { diff |= a[i] ^ b[i]; }
    return diff == 0;
}

bool rpc_authenticate(socket_t & sock, const rpc_auth_key & key, bool server) {
    if (key.bytes.empty()) { return true; }
    uint8_t greeting[36] = {'G','R','A','1'}, response[64] = {}, expected[32] = {};
    bool ok;
    if (server) {
        ok = random_nonce(greeting+4) && sock.auth_send(greeting, sizeof(greeting)) &&
             sock.auth_recv(response, sizeof(response));
        if (ok) {
            proof(key, false, greeting+4, response, expected);
            ok = equal(expected, response+32);
        }
        if (ok) {
            proof(key, true, response, greeting+4, expected);
            ok = sock.auth_send(expected, sizeof(expected));
        }
    } else {
        ok = sock.auth_recv(greeting, sizeof(greeting)) && memcmp(greeting, "GRA1", 4) == 0 && random_nonce(response);
        if (ok) {
            proof(key, false, greeting+4, response, response+32);
            ok = sock.auth_send(response, sizeof(response)) && sock.auth_recv(expected, sizeof(expected));
        }
        if (ok) {
            proof(key, true, response, greeting+4, response+32);
            ok = equal(expected, response+32);
        }
    }
    wipe(expected, sizeof(expected)); wipe(response, sizeof(response));
    if (!ok) { LOG_ERROR("%s: RPC auth failed\n", sock.peer_address().c_str()); }
    return ok;
}
