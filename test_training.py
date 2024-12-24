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


import unittest
from unittest.mock import patch
import argparse
from metrics_collector import analyze_metrics
import pandas as pd
from io import StringIO
import logging
import tempfile
import os
from training import main     

class TestTraining(unittest.TestCase):
    @patch('training.arguments.parse_args')
    @patch('training.configure_object_sources')
    @patch('training.configure_epoch')
    @patch('training.configure_samples')
    @patch('training.Epoch')
    @patch('training.setup_logger')
    @patch('training.setup_metrics_exporter')
    @patch('training.setup_metrics_logger')
    @patch('training.metrics_logger.AsyncMetricsLogger')
    @patch('training.monitoring.initialize_monitoring_provider')
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.destroy_process_group')
    @patch('training.util.clear_kernel_cache')
    @patch('training.sequential_reader')
    @patch('training.close_metrics_logger')
    def test_main_success(self, 
                          mock_close_metrics_logger,
                          mock_sequential_reader,
                          mock_clear_kernel_cache, 
                          mock_destroy_process_group, 
                          mock_init_process_group, 
                          mock_initialize_monitoring_provider, 
                          mock_AsyncMetricsLogger, 
                          mock_setup_metrics_exporter, 
                          mock_setup_metrics_logger, 
                          mock_setup_logger, 
                          mock_Epoch, 
                          mock_configure_samples, 
                          mock_configure_epoch, 
                          mock_configure_object_sources, 
                          mock_parse_args, 
                          ):

        # Mock the arguments
        mock_args = argparse.Namespace(
            prefix=["gs://test-bucket/"],
            epochs=1,
            steps=1,
            sample_size=1024,
            batch_size=1024,
            read_order=["Sequential"],
            background_queue_maxsize=2048,
            background_threads=16,
            group_coordinator_address="localhost",
            group_coordinator_port="4567",
            group_member_id=0,
            group_size=1,
            label="test-label",
            log_metrics=True,
            export_metrics=True,
            metrics_file="metrics.csv",
            clear_pagecache_after_epoch=True,
        )
        
        mock_parse_args.return_value = mock_args

        # Mock the necessary functions
        mock_configure_object_sources.return_value = {"gs://test-bucket/": "sequential_reader"}
        mock_configure_epoch.return_value = (lambda *args: [], "Sequential", "test_fs", "test_fs", ["test_object"])
        mock_configure_samples.return_value = [("test_object", 0)]
        mock_Epoch.return_value = [f"Step: 0, Duration (ms): 100, Batch-sample: 1024"]

        # Mock the logger and metrics exporter
        mock_logger = logging.getLogger("test-label")
        mock_logger.propagate = False
        mock_logger.setLevel(logging.INFO)
        mock_setup_logger.return_value = mock_logger
        mock_initialize_monitoring_provider.return_value = "test_meter"
        mock_setup_metrics_exporter.return_value = "test_meter"
        mock_setup_metrics_logger.return_value = "test_metrics_logger"
        mock_AsyncMetricsLogger.return_value.log_metric.return_value = None
        mock_AsyncMetricsLogger.return_value.close.return_value = None
        mock_sequential_reader.return_value = [("test_object", 0)]
        mock_close_metrics_logger.return_value = None
        mock_clear_kernel_cache.return_value = None

        # Call the main function
        main()

        # Assertions
        mock_parse_args.assert_called_once()
        mock_setup_logger.assert_called_once()
        mock_setup_metrics_exporter.assert_called_once()
        mock_setup_metrics_logger.assert_called_once()
        mock_configure_object_sources.assert_called_once()
        mock_configure_epoch.assert_called_once()
        mock_configure_samples.assert_called_once()
        mock_Epoch.assert_called_once()
        mock_init_process_group.assert_called_once()
        mock_clear_kernel_cache.assert_called_once()
        mock_close_metrics_logger.assert_called_once()

if __name__ == '__main__':
    loader = unittest.TestLoader()
    # Discover tests in the current directory and its subdirectories
    suite = loader.discover(".")  # "." specifies the starting directory

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    unittest.main()