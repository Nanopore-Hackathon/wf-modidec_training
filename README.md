# wf-modidec (Part2: Model training)

## Introduction
In this directory you find part 2 of the code from ModiDec. The scripts located in ./bin facilitate the model loading and training. You can find test data is the ./test folder of this repository.


### Functionality Overview
Below is a graphical overview of suggested routes through the pipeline depending on the desired output.

[image]

## Guideline

1. Get your system up and ready
    - Install [`Nextflow`](https://www.nextflow.io/docs/latest/getstarted.html#installation) (`>=23.10.0`)
    - Install [`Miniconda`](https://conda.io/miniconda.html)

    - Install [`Docker`](https://conda.io/miniconda.html)

    - Install [`Epi2Me Desktop`](https://labs.epi2me.io) (v5.1.14 or later)

    - Clone the Github repository (we recommend the GitHub Desktop client)


2. Translate the functions and logic into Nextflow processes and ultimately a Nextflow pipeline
    - Please check the repository wf-modidec_analysis to get some hints about the synthax and folder structure of an Epi2Me workflow
    - Extract the variables that will be needed to train the network (have a look in ./bin/Training_NN_GUI.py)
    - Define a set variable names that will occur in all the scripts (e.g. nextflow_schema.json, ./bin/Training_NN_GUI.py)
    - Remove GUI features from ./bin/Training_NN_GUI.py and implement a script that can be started via command line tool. We recommend implementing an argparser.
    - Implement a process in main.nf, that takes all necessary variables as input and starts the changed ./bin/Training_NN.py script. Check which files you want to write to the filesystem and define them in the output section. Define a PublishDir to write the models to your filesystem. Assign the label modidec to the process. 
    - Implement a workflow in main.nf which calls the defined process and assign the right input variables to it.
    - Define a test config.yaml file to test the code without running it in Epi2Me. Take the config.yaml of the wf-modidec_analysis repository as template. 
    - In nextflow.config assign a label to the Docker container "stegiopast/modidec:latest" to enable the modidec label feature. (check wf-modidec_analysis/nextflow.config) 
    - Test if the pipeline runs through, when starting it via console. (nextflow run main.nf -params-file config.yaml) Test data can be found in folder ./bin/data/

3. Convert the GUI into nextflow_schema.json format for Epi2Me integration
    - Define the datatype and description for each variable that needs to be taken as input for the nextflow pipeline.
    - make epi2melabs/workflows/modidec directory 
    - Copy the wf-modidec_training repositroy into epi2melabs/workflows/modidec
    - Open Epi2Me and check if the repository occurs in the workflow section  


5. Test and debug pipeline in Epi2Me using test data


## Credits & License

This code is provided by Dr. Nicolo Alagna and the Computational Systems Genetics Group of the University Medical Center of Mainz. © 2024 All rights reserved.

For the purpose of the Nanopore Hackathon, participants and collaborators of the event are granted a limited, non-exclusive, non-transferable license to use and modify the Applications for their intended purpose during the Time of the Event. Any further unauthorized reproduction, distribution, modification or publication of the Applications or their contents violates intellectual property rights and is strictly prohibited. For more please see the [Terms and Conditions](https://drive.google.com/file/d/18WN3YRoY9YvpYq6RCtwUQre-VAbN7jH6/view?usp=sharing) of the event.
