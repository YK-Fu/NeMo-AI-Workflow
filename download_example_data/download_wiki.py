# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from nemo_curator.download import download_wikipedia
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    # Params
    dump_date = None
    output_directory = f"./wiki_data_{args.language}/"
    url_limit = args.url_limit

    # Set up Dask client
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Download and sample data
    wikipedia = download_wikipedia(
        output_directory, dump_date=dump_date, url_limit=url_limit, language=args.language
    )

    wikipedia.to_json(output_directory, write_to_filename=True)
    client.close()


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    parser.add_argument("--url-limit", type=int, default=1, help="Number of URLs to download")
    parser.add_argument("--language", type=str, default="zh-classical", help="Language to download")
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
