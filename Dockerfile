ARG mode

FROM rocker/tidyverse:4.0.1 AS replication
RUN echo "this is the stage that sets VAR=replication"
ENV VAR="replication"

FROM rocker/tidyverse:4.0.1 AS validation
RUN echo "this is the stage that sets VAR=validation"
ENV VAR="validation"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements.R /app/requirements.R
# pre install the packages during build
RUN Rscript requirements.R

FROM ${mode} AS final
RUN echo "VAR is equal to ${VAR}"