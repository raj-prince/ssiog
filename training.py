#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A microbenchmark for training workloads.
"""

import argparse
import fsspec
import datetime
import queue
import random
import sys
import threading
import time
from typing import Iterable

import gcsfs
import torch.distributed as td
import torch.multiprocessing as mp
import pyarrow.fs as fs


# TODO(coryan) - the sample size and batch size should be randomly sampled


class Source(object):
    def __init__(self, name: str, filesystem: fs.FileSystem, objects: Iterable[str]):
        self.name = name
        self.filesystem = filesystem
        self.objects = list(objects)


def main():
    """Main entry point."""
    args = parse_args()

    print(
        f"# Starting process {args.group_member_id}/{args.group_size}",
        file=sys.stderr,
        flush=True,
    )
    td.init_process_group(
        "gloo",
        init_method=f"tcp://{args.group_coordinator_address}:{args.group_coordinator_port}",
        rank=args.group_member_id,
        world_size=args.group_size,
    )

    print("testing")

    sources = configure_object_sources(args)

    print(
        "epoch,step,duration_ns,batch_size,start,read_order"
        + ",filesystem_name"
        + ",arg_object_count_limit"
        + ",arg_epochs"
        + ",arg_sample_size"
        + ",arg_batch_size"
        + ",arg_read_order"
        + ",arg_background_queue_maxsize"
        + ",arg_background_threads"
        + ",arg_group_member_id"
        + ",arg_group_size"
        + ",labels"
    )

    for epoch in range(args.epochs):
        print(
            f"# Configure epoch {epoch}",
            file=sys.stderr,
            flush=True,
        )
        (reader, read_order, filesystem_name, filesystem, epoch_objects) = (
            configure_epoch(sources, args)
        )

        print(
            f"# Compute samples for epoch {epoch}, read_order={read_order}, epoch_objects.len={len(epoch_objects)}, fs_name={filesystem_name}",
            file=sys.stderr,
            flush=True,
        )
        samples = configure_samples(epoch_objects, filesystem, args)
        print(
            f"# Running epoch {epoch}, read_order={read_order}, epoch_objects.len={len(epoch_objects)}, fs_name={filesystem_name}, samples={len(samples)}",
            file=sys.stderr,
            flush=True,
        )

        annotations = ",".join(
            [
                f"{read_order}",
                f"{filesystem_name}",
                f"{args.object_count_limit}",
                f"{args.epochs}",
                f"{args.sample_size}",
                f"{args.batch_size}",
                f"{';'.join(args.read_order)}",
                f"{args.background_queue_maxsize}",
                f"{args.background_threads}",
                f"{args.group_member_id}",
                f"{args.group_size}",
                ";".join(args.labels),
            ]
        )
        for summary in Epoch(reader, epoch_objects, filesystem, samples, args):
            print(f"{epoch},{summary},{annotations}", flush=True)


def Epoch(
    reader: callable,
    epoch_objects: Iterable[str],
    filesystem: fs.PyFileSystem,
    samples: list,
    args: argparse.Namespace,
):
    q = queue.Queue(maxsize=args.background_queue_maxsize)
    for i in range(args.background_threads):
        threading.Thread(
            daemon=True,
            target=_background,
            args=(
                reader,
                q,
                epoch_objects,
                i,
                args.background_threads,
                filesystem,
                args.sample_size,
                samples,
            ),
        ).start()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(sep="T")
    step_start = time.monotonic_ns()
    step = 0
    running = args.background_threads
    batch_samples = 0
    remaining = len(samples)
    while running != 0 and step < args.steps:
        item = q.get()
        if isinstance(item, Done):
            q.task_done()
            running -= 1
            continue
        q.task_done()
        batch_samples += 1
        remaining -= args.batch_size
        if batch_samples < args.batch_size:
            continue
        duration_ns = time.monotonic_ns() - step_start
        yield f"{step},{duration_ns},{batch_samples},{start}"
        td.barrier()
        start = datetime.datetime.now(datetime.timezone.utc).isoformat(sep="T")
        step_start = time.monotonic_ns()
        step += 1
        batch_samples = 0
    for i in range(step, args.steps):
        print(f"# empty step {i}", file=sys.stderr)
        td.barrier()


class Done(object):
    pass


def _subset(samples: Iterable, index: int, count: int) -> list[str]:
    return [o for i, o in enumerate(samples) if i % count == index]


def _background(
    reader: callable,
    queue: queue.Queue,
    object_names: Iterable[str],
    thread_id: int,
    thread_count: int,
    filesystem: fs.FileSystem,
    sample_size: int,
    samples: list,
):
    for r in reader(
        object_names, thread_id, thread_count, filesystem, sample_size, samples
    ):
        queue.put(r)
    queue.put(Done())


def sequential_reader(
    object_names: Iterable[str],
    thread_id: int,
    thread_count: int,
    filesystem: fs.FileSystem,
    sample_size: int,
    samples: list,
):
    subset = _subset(object_names, td.get_rank(), td.get_world_size())
    subset = _subset(subset, thread_id, thread_count)
    for name in subset:
        # Only read as many samples as have been configured for this object.
        max_offset = sample_size * len([o for n, o in samples if n == name])
        with filesystem.open_input_stream(name) as f:
            offset = 0
            while offset < max_offset:
                chunk = f.read(sample_size)
                if not chunk:
                    break
                yield (offset, chunk)
                offset += len(chunk)


def file_random_reader(
    object_names: Iterable[str],
    thread_id: int,
    thread_count: int,
    filesystem: fs.FileSystem,
    sample_size: int,
    samples: list,
):
    subset = _subset(object_names, td.get_rank(), td.get_world_size())
    subset = _subset(subset, thread_id, thread_count)
    for name in subset:
        data = filesystem.open_input_file(name).readall()
        offsets = [o for n, o in samples if n == name]
        for offset in offsets:
            chunk = data[offset : min(len(data), offset + sample_size)]
            yield (offset, chunk)
        del offsets
        del data


def full_random_reader(
    object_names: Iterable[str],
    thread_id: int,
    thread_count: int,
    filesystem: fs.FileSystem,
    sample_size: int,
    samples: list,
):
    files = {n: filesystem.open_input_file(n) for n in object_names}
    subset = _subset(samples, td.get_rank(), td.get_world_size())
    subset = _subset(subset, thread_id, thread_count)
    for name, offset in subset:
        chunk = files[name].read_at(sample_size, offset)
        if not chunk:
            continue
        yield (offset, chunk)
    del samples
    del files


def configure_samples(
    object_names: Iterable[str], filesystem: fs.FileSystem, args: argparse.Namespace
):
    samples = []
    print(f"# Opening {len(object_names)} files", file=sys.stderr, flush=True)
    files = {n: filesystem.open_input_file(n) for n in object_names}
    total_samples = args.batch_size * args.steps
    print(f"# Computing {total_samples} samples", file=sys.stderr, flush=True)
    for name, f in files.items():
        samples.extend([(name, offset) for offset in range(0, f.size(), args.sample_size)])
    print(f"# Computing {total_samples} from {len(samples)} samples", file=sys.stderr, flush=True)
    samples = random.choices(samples, k=total_samples)

    td.broadcast_object_list(samples, src=0)
    td.barrier()
    return samples


def configure_epoch(sources: dict[str, Source], args: argparse.Namespace):
    prefix = [random.choice(args.prefix)]
    td.broadcast_object_list(prefix, src=0)
    td.barrier()

    p = prefix[0]
    name = sources[p].name
    filesystem = sources[p].filesystem
    epoch_objects = sources[p].objects.copy()
    random.shuffle(epoch_objects)
    if len(epoch_objects) > args.object_count_limit:
        epoch_objects = epoch_objects[0 : args.object_count_limit]

    td.broadcast_object_list(epoch_objects, src=0)
    td.barrier()

    read_order = [random.choice(args.read_order)]
    td.broadcast_object_list(read_order, src=0)
    td.barrier()

    if read_order[0] == "Sequential":
        reader = sequential_reader
    elif read_order[0] == "FileRandom":
        reader = file_random_reader
    elif read_order[0] == "FullRandom":
        reader = full_random_reader
    else:
        raise Exception(f"Unknown reading order {read_order[0]}")

    return (reader, read_order[0], name, filesystem, epoch_objects)


def configure_object_sources(args: argparse.Namespace) -> dict[str, Source]:
    sources = dict()
    for prefix in args.prefix:
        if prefix.startswith("gs://"):
            objects = fsspec.filesystem("gcs").ls(prefix.removeprefix("gs://"))
            sources[prefix] = Source("gcs", fs.GcsFileSystem(), objects)
        elif prefix.startswith("gcsfs://"):
            sources[prefix] = Source(
                "fsspec",
                fs.PyFileSystem(fs.FSSpecHandler(gcsfs.GCSFileSystem())),
                fsspec.filesystem("gcs").ls(prefix.removeprefix("gcsfs://")),
            )
        else:
            sources[prefix] = Source(
                "local", fs.LocalFileSystem(), fsspec.filesystem("local").ls(prefix)
            )
    return sources


def parse_args() -> argparse.Namespace:
    """Parse the arguments and invoke the necessary steps."""
    parser = argparse.ArgumentParser(description="Process Warp benchmark results.")
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="+",
        help=(
            "Use the files starting with the given prefix(es)."
            + " Use gs://... when using direct GCS access."
        ),
    )
    parser.add_argument(
        "--object-count-limit",
        type=int,
        help="Limit the number of objects.",
        default=1_000_000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs.",
        default=4,
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of steps.",
        default=2_000,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size in bytes.",
        default=1024,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size in number of samples.",
        default=1024,
    )
    parser.add_argument(
        "--read-order",
        type=str,
        nargs="+",
        help="Sampling order strategy (Sequential, FileRandom, FullRandom).",
        default=["Sequential", "FileRandom", "FullRandom"],
    )
    parser.add_argument(
        "--background-queue-maxsize",
        type=int,
        help="Maximum size for the threaded queue.",
        default=2048,
    )
    parser.add_argument(
        "--background-threads",
        type=int,
        help="Number of background threads.",
        default=16,
    )
    parser.add_argument(
        "--group-coordinator-address",
        type=str,
        help="The coordinator (rank==0) address.",
        default="localhost",
    )
    parser.add_argument(
        "--group-coordinator-port",
        type=str,
        help="The coordinator (rank==0) port.",
        default="4567",
    )
    parser.add_argument(
        "--group-member-id",
        type=int,
        help="The id within the group. Also known as the process rank.",
        default=0,
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="The process group size.",
        default=1,
    )
    parser.add_argument(
        "labels",
        type=str,
        nargs="*",
        help="Additional labels to distinguish this run.",
        default=[],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
