ARG mode

## Reproduces all results in original research paper using provided R code:
FROM rocker/tidyverse:4.0.1 AS reproduction
RUN echo "this is the stage that sets VAR=reproduction"
ENV VAR="reproduction"

FROM rocker/tidyverse:4.0.1 AS validation
RUN echo "this is the stage that sets VAR=validation"
ENV VAR="validation"

FROM ${mode} AS final
RUN echo "VAR is equal to ${VAR}"

ADD . / app/
WORKDIR /app

RUN find ./src -mindepth 1 ! -regex '^./src/'$VAR'\(/.*\)?' -delete

RUN if [ "$VAR" = "reproduction" ] ; then \
        git clone https://github.com/mandycoston/counterfactual ./github/;\
        git clone https://github.com/mandycoston/equalized_odds_and_calibration/ ./github/equalized_odds_and_calibration/;\
        mkdir reproduction; mkdir data; mkdir data/synthetic;\
        chmod 777 reproduction;\
    else \
        echo do something else; \
    fi

RUN sudo apt-get update
RUN sudo apt-get install -y python3-pip

# pre install the packages during build
RUN Rscript requirements.R
RUN pip install -r requirements.txt