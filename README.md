#Large Scale Network Project

Authors: D. Dündar, F. Laurent, N. Levent, G. Plays

## Introduction

Welcome to our brand new git project!

The goal of this project is to analyse a large scale citation network.
The graph will be preprocessed and new features will be computed for the nodes.
From these features we will define new metrics to compute centrality, closeness, betweeness and clustering.
The metrics should be defined in such a way that it is related to the probability that a paper cite another.

## Prerequisites

This project use __Python 3.6__
Please install the requirements using:
"pip install -r requirements.txt"


## Quick user guide

The data must be downloaded before being able to run any part of the project.
A [script](downloadData.sh) is provided for downloading the datasets.
Currently two datasets from the [aminer website](https://aminer.org/citation) are used.
We assume the datasets will be in a __.data__ directory at the root of the project. \
If it is not the case you will have to provide the path at the time of the preprocessing.
For non UNIX user, you can download manually the datasets using the links in the script.

__NB__: All hidden files (prepended with ".") won't be included in the git. Please DO NOT push any data.

