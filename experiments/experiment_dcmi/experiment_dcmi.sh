#!/usr/bin/env bash

expshell="python3 main.py"
dataset_name=$1
sample_count=$2
action=$3
experiment="$expshell $dataset_name $sample_count"


function build_adtrees() {
    local tree_type=$1

    echo "Building $tree_type AD-tree, pruning 0"
    $experiment adtree build --tree-type=$tree_type --llt=0

    echo "Building $tree_type AD-tree, pruning 5"
    $experiment adtree build --tree-type=$tree_type --llt=5

    echo "Building $tree_type AD-tree, pruning 10"
    $experiment adtree build --tree-type=$tree_type --llt=10
}



function run_unoptimized() {
    $experiment exp unlock
    $experiment --algrun-tag=unoptimized exp run
}



function run_dcmi() {
    $experiment exp unlock
    $experiment --algrun-tag=dcmi exp run
}



function run_adtree() {
    local tree_type=$1
    local llt=$2
    $experiment exp unlock
    $experiment --algrun-tag="adtree-$tree_type-llt$llt" exp run
}



if [ "$action" == "exds" ]; then
  $experiment exds build
fi

if [ "$action" == "prep" ]; then
    $experiment exds build
    build_adtrees dynamic
    if [ "$dataset_name" == "alarm" ]; then
        build_adtrees static
    fi
fi

if [ "$action" == "prep-static" ]; then
    $experiment exds build
    build_adtrees static
fi

if [ "$action" == "unoptimized" ]; then
    run_unoptimized
fi

if [ "$action" == "dcmi" ]; then
    run_dcmi
fi

if [ "$action" == "static-adtree" ]; then
    if [ "$dataset_name" == "alarm" ]; then
        run_adtree static 0
        run_adtree static 5
        run_adtree static 10
    else
        echo "Not running static-adtree on $dataset_name, only on 'alarm'."
    fi
fi

if [ "$action" == "dynamic-adtree" ]; then
    run_adtree dynamic 0
    run_adtree dynamic 5
    run_adtree dynamic 10
fi
