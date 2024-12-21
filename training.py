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
A Synthetic Scale IO Generator for training workloads.
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
import logging
import arguments
import util

import gcsfs
import torch.distributed as td
import pyarrow.fs as fs

import monitoring 

from opentelemetry import metrics
import metrics_logger

# Import the GCP resource detector

# TODO(coryan) - the sample size and batch size should be randomly sampled
# TODO (raj-prince) - write a development guide.
# TODO (raj-prince) - clear the logging path.
# TODO (raj-prince) - overall testing on scale.
# TODO (raj-prince) - See how to write unit test.

# Global for recording sample latency to export.
sample_lat = metrics.NoOpHistogram("no_op")

# Initialize the global metrics logger with no-op logger.
sample_lat_logger = metrics_logger.NoOpMetricsLogger()

# Initialize the global logger with basic INFO level log.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class Source(object):
    def __init__(self, name: str, filesystem: fs.FileSystem, objects: Iterable[str]):
        self.name = name
        self.filesystem = filesystem
        self.objects = list(objects)

def setup_metrics(args):
    # Initialize the OpenTelemetry MeterProvider
    meter = monitoring.initialize_monitoring_provider(exporter_type=args.exporter_type)

    # Create a histogram metric
    global sample_lat
    sample_lat = meter.create_histogram(
        name="ssiog.sample_lat",
        description="Sample latency histogram",
        unit="ms"
    )

    logger.info("Metrics exporter initialized.")

def setup_logger(args):
    global logger
    logger = logging.getLogger(args.label)

    # No propagation in the logger hierarchy.
    logger.propagate = False

    # Log level.
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)

    # Log destination, where to write?
    handler = logging.FileHandler(args.log_file) if args.log_file else logging.StreamHandler()

    # Beautify.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Logger initialized.")

def main():
    # Parse arguments
    args = arguments.parse_args()

    # Initialize the global application logger.
    logger.info("Setting up logger.")
    setup_logger(args)
    
    logger.debug(f"Running with args: {args}")

    # Initialize the OpenTelemetry MeterProvider
    if args.export_metrics:
        logger.info("Setting up otlp metrics exporter.")
        setup_metrics(args)

    # Initialize the metrics logger.
    if args.log_metrics:
        logger.info(f"Logging metrics to: {args.metrics_file}")
        global sample_lat_logger
        sample_lat_logger = metrics_logger.AsyncMetricsLogger(file_name=args.metrics_file)
        
    logger.info("Initial setup completed.\n")

    logger.info(f"Starting process: {args.group_member_id}/{args.group_size}")
    td.init_process_group(
        "gloo",
        init_method=f"tcp://{args.group_coordinator_address}:{args.group_coordinator_port}",
        rank=args.group_member_id,
        world_size=args.group_size,
    )
    logger.info(f"Process started successfully: {args.group_member_id}/{args.group_size}\n")
    
    logger.info(f"Logging important workload configurations.")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Sample size (bytes): {args.sample_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Read order: {args.read_order[0]}")
    logger.info(f"Background queue max size: {args.background_queue_maxsize}")
    logger.info(f"Background threads: {args.background_threads}")
    logger.info(f"Group member id: {args.group_member_id}")
    logger.info(f"Group size: {args.group_size}")
    logger.info(f"Label: {args.label}")
    logger.info(f"Data set path: {args.prefix}.\n")
    sources = configure_object_sources(args)
    
    for epoch in range(args.epochs):
        logger.info(f"******** Starting epoch: {epoch} ********.")
        logger.info(f"Configure epoch: {epoch}.")
        (reader, read_order, filesystem_name, filesystem, epoch_objects) = (configure_epoch(sources, args))
        logger.info(f"Configured, total objects: {len(epoch_objects)}")
        
        logger.info(f"Configuring samples.")
        samples = configure_samples(epoch_objects, filesystem, args)
        logger.info(f"Configured, total selected samples: {len(samples)}")

        logger.info(f"Running epoch: {epoch}")
        for summary in Epoch(reader, epoch_objects, filesystem, samples, args):
            logger.info(f"Epoch: {epoch}, {summary}")
            
        logger.info(f"Epoch {epoch} completed.\n")
        
        # Clear the kernel cache
        if args.clear_pagecache_after_epoch:
            util.clear_kernel_cache(logger)


    # Make sure remaining metrics in the queue buffere is flushed to metrics file. 
    sample_lat_logger.close()
    
    td.destroy_process_group()
    logger.info("Workload completed successfully!!!")
        

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
        yield f"Step: {step}, Duration (ms): {duration_ns/1000000}, Batch-sample: {batch_samples}"
        td.barrier()
        step_start = time.monotonic_ns()
        step += 1
        batch_samples = 0
    
    for i in range(step, args.steps):
        logger.info(f"Empty step {i}")
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
    for r in reader(object_names, thread_id, thread_count, filesystem, sample_size, samples):
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
        logger.debug(f"Reading {name} sequentially from {0} to {max_offset}.")
        with filesystem.open_input_stream(name) as f:
            offset = 0
            while offset < max_offset:
                start_time = time.monotonic_ns()
                chunk = f.read(sample_size)
                elapsed_time = time.monotonic_ns() - start_time
                sample_lat_logger.log_metric(elapsed_time / 1000000)
                sample_lat.record(elapsed_time / 1000000, {"reader": "sequential"})
                if not chunk:
                    break
                yield (name, offset, elapsed_time)
                offset += len(chunk)


# TODO (raj-prince): discuss what observability is required for file_random_reader pattern.
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
        logger.debug(f"Reading {name} at {offset} with size {sample_size}.")
        start_time = time.monotonic_ns()
        chunk = files[name].read_at(sample_size, offset)
        elapsed_time = time.monotonic_ns() - start_time
        sample_lat_logger.log_metric(elapsed_time / 1000000)
        sample_lat.record(elapsed_time / 1000000, {"reader": "full_random"})
        if not chunk:
            continue
        yield (offset, chunk)
    for name, f in files.items():
        f.close()
    del files
    del samples


def configure_samples(
    object_names: Iterable[str], filesystem: fs.FileSystem, args: argparse.Namespace
):
    samples = []
    logger.info(f"Opening {len(object_names)} files.")
    files = {n: filesystem.open_input_file(n) for n in object_names}
    
    req_samples = args.batch_size * args.steps
    logger.info(f"Collecting {req_samples} samples.")
    
    for name, f in files.items():
        samples.extend([(name, offset) for offset in range(0, f.size(), args.sample_size)])
    logger.info(f"Total samples: {len(samples)}")
    
    logger.info(f"Selecting {req_samples} samples from {len(samples)} randomly.")
    if req_samples > len(samples):
        logger.warning(f"Req sample ({req_samples}) > available ({len(samples)}), hence duplicated.")
    
    samples = random.choices(samples, k=req_samples)

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


if __name__ == "__main__":
    main()
