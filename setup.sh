#!/usr/bin/env bash

requirements=$1
python3.5 -m pip install -r $requirements
python3.5 -m nltk.downloader all
