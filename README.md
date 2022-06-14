# hello_aug

This repo contains some basic routines to fetch, process and plot ASDEX-Upgrade (AUG) data. It is designed to support data analysis and modeling as part of the EuroFusion WPTE RT07 (negative triangularity) activity, but may be also useful to others.

![AUG shot #40470, t=4s](aug_nt_overview_40470.png?raw=true "AUG shot #40470, t=4s")

All routines are in Python3+ and assume that the user already has access to the AUG data systems. Please contact C. Fuchs if you do not already have access.

To install all required packages:

    pip install aug_sfutils omfit_classes --user

Make sure to have the latest version of both of these packages! You can update them with pip using for example

    pip install -U aug_sfutils

If you use these routines, it would be appreciated if you acknowledge in the future where you originally got them from.

Please contact Francesco Sciortino (name.surname@ipp.mpg.de) for info, requests, suggestions or contributions.

