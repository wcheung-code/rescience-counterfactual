#!/bin/sh
sudo apt-get update;
sudo apt-get install -y python3-pip;
pip install --upgrade pip;
git clone https://github.com/mandycoston/counterfactual ./github/;
git clone https://github.com/mandycoston/equalized_odds_and_calibration/ ./github/equalized_odds_and_calibration/;
Rscript requirements.R;