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

# write a method to clean the kernel cache, also print the ram on the system using free 

import os
import time

def get_ram_info():
    """
    This method gets the RAM information of the system and returns it as a dictionary.
    """
    ram_info = {}
    with os.popen("free -mh") as f:
        lines = f.readlines()
        mem_info = lines[1].split()
        ram_info['total'] = mem_info[1]
        ram_info['used'] = mem_info[2]
        ram_info['free'] = mem_info[3]
    return ram_info


def clean_kernel_cache():
    """
    This method cleans the kernel cache.
    """
    print(f"RAM Info Before: {get_ram_info()}")
    os.system("sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'")
    time.sleep(1) # Wait for 1 second to allow the cache to be cleared.
    
    print(f"RAM Info After: {get_ram_info()}")