# Modidec - RNA modification detector and classifier
ModiDeC is a customizable neural network to identify RNA modifications from Oxford Nanopore Technology (ONT) based direct RNA sequencing data. ModiDeC combines LSTM and newly designed inception-res-net blocks for multi-modification-classification. ModiDec is composed of three Epi2ME integratable tools (data curation, network training and analysis), allowing researchers to train the multi-modification-classification model on synthesized RNA strands mimicking physiological relevant motifs and modification patterns on transcripts of interest to investigate modification ratios of transcripts derived from physiological data. During the data curation step of ModiDec, data derived from ONT based direct RNA sequencing experiments (RNA002 or RNA004) can be preprocessed to suit the succeeding model training step. During model training the network can be trained on the preprocessed data to optimally learn motif and modification patterns of the transcript of interest. The trained model can then be used in the analysis step of ModiDec to investigate modification ratios in physiological derived data.

Here the model training part is implemented. Please visit [wf-modidec_data-curation](https://github.com/Nanopore-Hackathon/wf-modidec-data_curation) and [wf-modidec_analysis](https://github.com/Nanopore-Hackathon/wf-modidec_analysis) to complete the toolset. 

## Requirements

Install dependencies on your system:
   -  Install [`Epi2Me Desktop`](https://labs.epi2me.io) (v5.1.14 or later)
   -  Install [`Miniconda`](https://conda.io/miniconda.html)
   -  Install [`Docker`](https://conda.io/miniconda.html)
   -  Install [`Nextflow`](https://www.nextflow.io/docs/latest/getstarted.html#installation) (`>=23.10.0`)
   -  Install samtools and minimap
   -  Make sure your [nvidia GPU drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/#ubuntu-installation) are installed and functional.

Import the workflow in Epi2Me:
   -  Open Epi2Me
   -  Navigate to Launch
   -  Click on import workflow -> Import from Github
   -  Paste https://github.com/Nanopore-Hackathon/wf-modidec_training into the Edit line
   -  Click on Download
   -  The repository should now appear on the workflow list in Launch
