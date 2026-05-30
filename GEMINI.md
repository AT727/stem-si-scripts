# Project: Wave Height Analysis Pipeline

## Overview
This project aims to automate wave height measurement from wave tank MP4 videos to replace manual tracking in Tracker OSP. The pipeline must handle high-precision fluid-solid interaction data, requiring <5% NRMSE and lowest possible RMSE.

## Building and Running
TODO: Document the commands to set up the environment, run the pipeline, and process videos.

## Development Conventions
- **Rigorous Test-Driven Development (TDD):** All features, bugfixes, and refactors MUST start with a failing test case that demonstrates the requirement or issue.
- **Precision First:** Maintain <5% NRMSE. Accuracy in methodology is paramount—validate against peer-reviewed literature.
- **Data Integrity:** Avoid fabrication, falsification, or plagiarism of data.
- **Accuracy Verification:** Use `benchmark.csv` (manually tracked data) as the reference for validation.
- **Constraint Management:** Be aware of visual artifacts (e.g., the 1100 mm sharpie line) that may interfere with filters/gradients.

# Files
@PhaseII_TestD_0003_c4_01.MP4 - Video used in benchmark testing
@benchmark.csv - Real peer reviewed data from manually tracking in Tracker OSP
Use benchmark.csv with maxHeight ~8.2 cm and minHeight = 0 cm as a reference for data accuracy

# Gotchas
- There is a sharpie line on the 1100 mm mark (corresponds ~ 8 cm b/c 0 is at the bottom of the screen) that may interfere with 
any filter or gradients