# Lens_candidate

## Description
This is a small python package to analyze VLBI datasets to test the milli-lens hypothesis. It requires full VLBI datasets including uvfits file, fits image in total intensity, as well as Gaussian modelfit files readable by Difmap, which it utilized in the background and should be installed. The script then generates plots and other analysis tools flexibly for all epochs and frequencies studied.

## Installation
run git clone https://github.com/fpoetzl/Lens_candidate.git or save individual files on your computer. Add to your PYTHONPATH or to your local folder where you want to run the script.

## Prerequisites
astropy (tested with version 6.0.1)

numpy (tested with version 1.26.4)

matplotlib

Difmap (tested with version 2.5p (18 Nov 2022))


See the following resources for installing Difmap:

Original: http://brandeisastro.pbworks.com/w/page/14977088/Difmap%20Tutorial

Homebrew package by Kazu Akiyama: https://github.com/kazuakiyama/homebrew-difmap

## Usage
The provided jupyter notebook (example.ipynb) provides all relevant examples on how to use the script.

## Authors and acknowledgment
Felix Pötzl, 2025

## License
License tbd.

