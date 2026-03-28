"""Sample CuTeDSL-style vector add source for integration tests.

This fixture is intentionally lightweight. The current integration-test goal is
API/interface coverage, not full end-to-end correctness of the DSL toolchain.
"""

import cutlass.cute as cute


def build_vector_add(length: int = 1 << 20, elements_per_thread: int = 4):
    threads = 256
    blocks = (length + threads * elements_per_thread - 1) // (threads * elements_per_thread)

    @cute.kernel
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n):
        block_idx = cute.block_idx.x
        thread_idx = cute.thread_idx.x
        base = (block_idx * threads + thread_idx) * elements_per_thread

        for lane in range(elements_per_thread):
            idx = base + lane
            if idx < n:
                out_ptr[idx] = x_ptr[idx] + y_ptr[idx]

    return {
        "kernel": vector_add_kernel,
        "length": length,
        "threads": threads,
        "blocks": blocks,
        "elements_per_thread": elements_per_thread,
    }


if __name__ == "__main__":
    spec = build_vector_add()
    print(spec)
