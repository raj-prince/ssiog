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

variable "project" {
  type = string
  default = "gcs-tess"
}

variable "location" {
  type = string
  default = "us-west1"
}

variable "cluster_name" {
  type = string
  default = "princer-ssiog"
}

variable "data_bucket_name" {
  type = string
  default = "princer-ssiog-data-bkt"
}

variable "metrics_bucket_name" {
  type = string
  default = "princer-ssiog-metrics-bkt"
}

variable "parallelism" {
  type = number
  default = "4"
}

variable "epochs" {
  type = number
  default = "2"
}

variable "labels" {
  default     = ["0_testing"]
  type        = list(string)
  description = "Annotations for the workload."
}

variable "image_name" {
  default = "v0.1.0"
  type = string
  description = "ssiog benchmark image name"
}
