#!/usr/bin/env bash
echo 'deb http://downloads.skewed.de/apt/xenial xenial universe' | sudo tee -a  /etc/apt/sources.list
echo 'deb-src http://downloads.skewed.de/apt/xenial xenial universe' | sudo tee -a  /etc/apt/sources.list
sudo apt-get update
sudo apt-get install python3-graph-tool

