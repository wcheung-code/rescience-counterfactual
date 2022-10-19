
## Initialize all build arguments in docker build
ARG mode

## Reproduces all results in original research paper using provided R code:
FROM rocker/tidyverse:4.0.1 AS reproduction
RUN echo "this is the stage that sets VAR=reproduction"
ENV VAR="reproduction"

## Generates all data visualizations confirming successful replication:
FROM rocker/tidyverse:4.0.1 AS validation
RUN echo "this is the stage that sets VAR=validation"
ENV VAR="validation"

FROM python:3.8-slim-bullseye AS replication
RUN echo "this is the stage that sets VAR=replication"
ENV VAR="replication"

## Chooses docker build with respect to parameter `mode`
FROM ${mode} AS final
RUN echo "VAR is equal to ${VAR}"

## Copies all files in GitHub project and defaults working directory to /app
ADD . / app/
WORKDIR /app

## Removes all irrelevant src files with respect to provided mode parameter
RUN find ./mode -mindepth 1 ! -regex '^./mode/'$VAR'\(/.*\)?' -delete
## Removes all irrelevant data files with respect to provided mode parameter
#RUN find ./data -mindepth 1 ! -regex '^./data/'$VAR'\(/.*\)?' -delete

RUN mkdir -p data/post_processed data/precision_recall data/roc data/calibration;
RUN mkdir -p $VAR/post_processed $VAR/precision_recall $VAR/roc $VAR/calibration;

RUN if [ "$VAR" = "reproduction" ] ; then \
        sudo apt-get update; \
        sudo apt-get install -y python3-pip; \
        git clone https://github.com/mandycoston/counterfactual ./github/;\
        git clone https://github.com/mandycoston/equalized_odds_and_calibration/ ./github/equalized_odds_and_calibration/;\
        Rscript requirements.R; \
    elif [ "$VAR" = "validation" ] ; then \
        sudo apt-get update; \
        sudo apt-get install -y python3-pip; \
        sudo apt install -y texlive \
            texlive-latex-extra \ 
            texlive-fonts-recommended \
            dvipng \ 
            cm-super; \
        pip install latex; \
    elif [ "$VAR" = "replication" ] ; then \
        mkdir -p data/reweighing $VAR/reweighing; \
    else \
        echo do something else; \
    fi

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .