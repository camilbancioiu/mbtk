#!/usr/bin/env bash

expshell="python3 main.py"

function adtrees() {
  echo "Building AD-tree, pruning 0"
  $expshell $1 --llt 0 adtree build

  echo "Building AD-tree, pruning 5"
  $expshell $1 --llt 5 adtree build

  echo "Building AD-tree, pruning 10"
  $expshell $1 --llt 10 adtree build
}

function run_unoptimized() {
  $expshell $1 exp unlock
  $expshell $1 --algrun-tag unoptimized --dont-preload-adtree exp run
}

function run_dcmi() {
  $expshell $1 exp unlock
  $expshell $1 --algrun-tag dcmi --dont-preload-adtree exp run
}

function run_adtree() {
  $expshell $1 exp unlock
  $expshell $1 --llt $2 --algrun-tag adtree-llt$2 exp run
}

function run_all_adtree() {
  run_adtree $1 0
  run_adtree $1 5
  run_adtree $1 10
}

function fullexperiment() {
  echo "=========================================="
  echo "Running full experiment for sample size $1"

  adtrees $1

  run_unoptimized $1

  run_dcmi $1

  run_adtree $1 0

  run_adtree $1 5

  run_adtree $1 10
}

echo "run_adtree 8e4 5"
run_adtree 8e4 5

echo "run_adtree 8e4 10"
run_adtree 8e4 10

$expshell --llt 5  2e3 adtree analyze
$expshell --llt 10 2e3 adtree analyze
$expshell --llt 5  8e3 adtree analyze
$expshell --llt 10 8e3 adtree analyze
$expshell --llt 5  2e4 adtree analyze
$expshell --llt 10 2e4 adtree analyze
$expshell --llt 5  4e4 adtree analyze
$expshell --llt 10 4e4 adtree analyze
$expshell --llt 5  8e4 adtree analyze
$expshell --llt 10 8e4 adtree analyze
