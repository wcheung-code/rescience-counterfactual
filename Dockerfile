
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

## Chooses docker build with respect to parameter `mode`
FROM ${mode} AS final
RUN echo "VAR is equal to ${VAR}"

## Copies all files in GitHub project and defaults working directory to /app
ADD . / app/
WORKDIR /app

## Removes all irrelevant src files with respect to provided mode parameter
RUN find ./src -mindepth 1 ! -regex '^./src/'$VAR'\(/.*\)?' -delete
## Removes all irrelevant data files with respect to provided mode parameter
RUN find ./data -mindepth 1 ! -regex '^./data/'$VAR'\(/.*\)?' -delete

RUN if [ "$VAR" = "reproduction" ] ; then \
        git clone https://github.com/mandycoston/counterfactual ./github/;\
        git clone https://github.com/mandycoston/equalized_odds_and_calibration/ ./github/equalized_odds_and_calibration/;\
        mkdir reproduction;\
    elif [ "$VAR" = "validation" ] ; then \
        mkdir validation; \
    else \
        echo do something else; \
    fi

RUN sudo apt-get update
RUN sudo apt-get install -y python3-pip 
RUN sudo apt install -y texlive \
    texlive-latex-extra \ 
    texlive-fonts-recommended \
    dvipng \ 
    cm-super

RUN pip install --upgrade pip
# pre install the packages during build
RUN Rscript requirements.R
RUN pip install latex
RUN pip install -r requirements.txt