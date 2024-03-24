#!/bin/sh

git submodule update --init --recursive

mkdir build
cmake -S . -B build