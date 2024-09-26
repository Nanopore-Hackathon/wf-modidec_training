#!/usr/bin/env nextflow

import groovy.json.JsonBuilder
import nextflow.util.BlankSeparatedList
import java.time.LocalDateTime
nextflow.enable.dsl = 2
nextflow.preview.recursion=true 

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
    python ${projectDir}/bin/analysis_neural_network.py -s ${params.start_index} -e ${params.end_index} -c ${params.chunk_size} -x ${params.max_seq_length} -r $reference_path -p ./pod5s -b $bam_path -m ./model -l $level_table_file 
    """
}

workflow {
    //input_dirs = Channel.fromPath("${params.model_path}", type: 'dir')
    TrainNeuralNetwork(file("${params.train_path}"),file("${params.valid_path}"))
}


