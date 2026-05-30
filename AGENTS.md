# AGENTS.md — Wave Height Analysis Pipeline

# Problem
I am interning at a research lab in civil engineering, fluid-solid interactions. Data is in MP4 and the team is using
tracker from Open Source Physics to manually measurement wave height. This process is extremely long. My goal is to automate wave height measurement from wave tank MP4s to process large amounts of videos. Videos will be ~128 fps between 0-10 minutes.

# Research First 
- Avoid fabrication, falsification, and plagarism of data
- Before implementing changes, do rigourous research on peer reviewed papers and the scientific literature in ensure most accurate methodology. 
- This pipeline requires PRECISE accuracy, <5% NRMSE and the lowest RMSE possible

# Files
@PhaseII_TestD_0003_c4_01.MP4 - Video used in benchmark testing
@benchmark.csv - Real peer reviewed data from manually tracking in Tracker OSP
Use benchmark.csv with maxHeight ~8.2 cm and minHeight = 0 cm as a reference for data accuracy

# Gotchas
- There is a sharpie line on the 1100 mm mark (corresponds ~ 8 cm b/c 0 is at the bottom of the screen) that may interfere with 
any filter or gradients
