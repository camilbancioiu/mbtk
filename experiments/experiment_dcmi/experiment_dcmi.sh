#!/usr/bin/env bash

expshell="python3 main.py"

function build_adtrees() {
    local dataset_name=$1
    local sample_count=$2
    local tree_type=$3
    local experiment="$expshell $dataset_name $sample_count"

    echo "Building AD-tree, pruning 0"
    $experiment adtree build --tree-type=$tree_type --llt=0

    echo "Building AD-tree, pruning 5"
    $experiment adtree build --tree-type=$tree_type --llt=5

    echo "Building AD-tree, pruning 10"
    $experiment adtree build --tree-type=$tree_type --llt=10
}

function run_unoptimized() {
    local dataset_name=$1
    local sample_count=$2
    local experiment="$expshell $dataset_name $sample_count"
    $experiment exp unlock
    $experiment --algrun-tag=unoptimized exp run
}

function run_dcmi() {
    local dataset_name=$1
    local sample_count=$2
    local experiment="$expshell $dataset_name $sample_count"
    $experiment exp unlock
    $experiment --algrun-tag=dcmi exp run
}

function run_adtree() {
    local dataset_name=$1
    local sample_count=$2
    local tree_type=$3
    local llt=$4
    local experiment="$expshell $dataset_name $sample_count"
    $experiment exp unlock
    $experiment --algrun-tag="adtree-$tree_type-llt$llt" exp run
}

function fullexperiment() {
    local dataset_name=$1
    local sample_count=$2
    local experiment="$expshell $dataset_name $sample_count"

    echo "=========================================="
    echo "Running full experiment for dataset $dataset_name and sample size $sample_count"

    $experiment exds build

    run_unoptimized $dataset_name $sample_count

    run_dcmi $dataset_name $sample_count

    if [ "$dataset_name" == "alarm" ]; then
        build_adtrees $dataset_name $sample_count static
        run_adtree $dataset_name $sample_count static 0
        run_adtree $dataset_name $sample_count static 5
        run_adtree $dataset_name $sample_count static 10
    fi

    build_adtrees $dataset_name $sample_count dynamic
    run_adtree $dataset_name $sample_count dynamic 0
    run_adtree $dataset_name $sample_count dynamic 5
    run_adtree $dataset_name $sample_count dynamic 10
}

fullexperiment "$@"
