"""Two Gloo processes agree on the minimum via the world group's CPU group (no CUDA involved)."""
import multiprocessing as mp
import socket

import torch.distributed as dist

from sglang.multimodal_gen.runtime.models.dits import minimax_h3_adaln_cache as m


class _World:
    @property
    def cpu_group(self):
        return dist.group.WORLD


def _worker(rank, port, queue):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    m.world_group_is_initialized = lambda: True
    m.get_world_group = lambda: _World()
    queue.put((rank, m._all_ranks_min(15 - 5 * rank)))  # rank 0 -> 15, rank 1 -> 10
    dist.destroy_process_group()


def test_all_ranks_min_over_gloo():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, queue)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert sorted(queue.get() for _ in range(2)) == [(0, 10), (1, 10)]


def test_all_ranks_min_without_distributed():
    assert m._all_ranks_min(7) == 7
