# ScanAnalysis

Sub-repository hosting modules for analyzing full scans.  Often times this requires the use of the other sub-repo, "ImageAnalysis" to analyze an individual image.  Here, the main distinquishing functionalities are

* Averaging across bins in a single scan before sending to an image analyzer (and other basic functionality across the scan)
* Analyzers that require data from multiple devices
* Automatically finding an analyzing scans that fit a given criteria.

Designed to work for any GEECS experiment, given the save data is in a similar format to HTU.
