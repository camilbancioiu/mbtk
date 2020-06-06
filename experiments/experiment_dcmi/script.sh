#!/usr/bin/env bash

expshell="py main.py"

function build_adtrees() {
    local dataset_name=$1
    local sample_count=$2
    local tree_type=$3
    local experiment=$expshell $dataset_name $sample_count

    echo "Building AD-tree, pruning 0"
    $experiment --llt 0 adtree build --type=$tree_type

    echo "Building AD-tree, pruning 5"
    $experiment --llt 5 adtree build --type=$tree_type

    echo "Building AD-tree, pruning 10"
    $experiment --llt 10 adtree build --type=$tree_type
}

function run_unoptimized() {
    local dataset_name=$1
    local sample_count=$2
    local experiment=$expshell $dataset_name $sample_count
    $experiment exp unlock
    $experiment --algrun-tag=unoptimized exp run
}

function run_dcmi() {
    local dataset_name=$1
    local sample_count=$2
    local experiment=$expshell $dataset_name $sample_count
    $experiment exp unlock
    $experiment --algrun-tag=dcmi exp run
}

function run_adtree_static() {
    local dataset_name=$1
    local sample_count=$2
    local experiment=$expshell $dataset_name $sample_count
    $experiment exp unlock
    $experiment --llt=$2 --algrun-tag=adtree-static-llt$2 exp run
}

function fullexperiment() {
    local dataset_name=$1
    local sample_count=$2
    local experiment=$expshell $dataset_name $sample_count

    echo "=========================================="
    echo "Running full experiment for dataset $dataset_name and sample size $sample_count"

    $experiment exds build

    run_unoptimized $dataset_name $sample_count

    run_dcmi $dataset_name $sample_count

    build_adtrees $dataset_name $sample_count "static"
    run_adtree_static $dataset_name $sample_count 0
    run_adtree_static $dataset_name $sample_count 5
    run_adtree_static $dataset_name $sample_count 10

    build_adtrees $dataset_name $sample_count "dynamic"
}
