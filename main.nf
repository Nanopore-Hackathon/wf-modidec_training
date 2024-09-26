#!/usr/bin/env nextflow

import groovy.json.JsonBuilder
import nextflow.util.BlankSeparatedList
import java.time.LocalDateTime
nextflow.enable.dsl = 2

process TrainNeuralNetwork{
    label "modidec"
    publishDir (path: "${params.out_dir}/", mode: "copy")
    stageInMode "symlink"
    input:
        path train_path
        path valid_path
    output:
        path("trained_model"), emit: trained_model
    script:
    """
    mkdir -p trained_model
    python ${projectDir}/bin/train_neural_network.py -t ${train_path} -v ${valid_path} -m ./trained_model -c ${params.chunk_size} -b ${params.batch_size} -s ${params.single_data_size} -l ${params.max_seq_length} -k ${params.kmer_model} -y ${params.labels} -e ${params.epochs} -n ${params.model_name}  
    """
}

workflow {
    //input_dirs = Channel.fromPath("${params.model_path}", type: 'dir')
    TrainNeuralNetwork(file("${params.train_path}"),file("${params.valid_path}"))
}

