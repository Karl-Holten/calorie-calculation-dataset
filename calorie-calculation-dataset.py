# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
# TODO: Address all TODOs and remove all explanatory comments
"""This dataset consists of recipes OCRed and taken from Internet Archive Cookbooks."""


import csv
import json
import os

import datasets


_CITATION = """\\
@InProceedings{@misc{Karl Holten,
  author = {Holten, Karl},
  title = {Computing Nutrition Profiles of Print Recipes Processed using Optical Character Recognition},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/Karl-Holten/calorie-calculation-dataset}}
  }}
"""

_DESCRIPTION = """\
This dataset consists of recipes OCRed and taken from the Internet Archive Cookbook and Home Economics Collection, dating from the early 20th century.
NER tags are given to first instances of ingredients, quantities and units in each recipe for the purposes of calculating calorie information.
"""

_HOMEPAGE = "https://github.com/Karl-Holten/calorie-calculation-dataset/"

_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# TODO: Register URL with huggingface.
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL= "https://github.com/Karl-Holten/calorie-calculation-dataset/"
_URLS = {
    "seperateIngredients": _URL + "fullset.csv"
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class CalorieCalculationDataset(datasets.GeneratorBasedBuilder):
    """This dataset consists of recipes OCRed and taken from the Internet Archive Cookbook and Home Economics Collection, dating from the early 20th century.
NER tags are given to first instances of ingredients, quantities and units in each recipe for the purposes of calculating calorie information.
"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="seperateIngredients", version=VERSION, description="Data for recipes with ingredient lists at start of recipe")
    ]

    DEFAULT_CONFIG_NAME = "seperateIngredients"

    def _info(self):
        if self.config.name == "seperateIngredients":
            features = datasets.Features(
                {
                    "Word": datasets.Value("string"),
                    "Unit": datasets.Value("string"),
                    "Qty": datasets.Value("string"),
                    "Srv": datasets.Value("string"),
                    "Ing": datasets.Value("string")
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "seperateIngredients":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "Word": data["Word"],
                        "Unit": "" if split == "test" else data["Unit"],
                        "Qty": "" if split == "test" else data["Qty"],
                        "Srv": "" if split == "test" else data["Srv"],
                        "Ing": "" if split == "test" else data["Ing"]
                    }
