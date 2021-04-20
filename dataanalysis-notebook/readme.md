# READ ME

## Analyze experimental data
You can analyze data with jupyter notebook very easily. Check out the [example notebook](200817_OAP2_ebeampointing.ipynb).

## Converting images to GIF
Check out an example in this [notebook](Gonvert2GIF.ipynb). You can convert images in a folder to one GIF. It will include a text with shotnumber and scan variable&parameter! The GIF will be saved in the folder where images are stored.
- Significant bits are taken care. Make labview=True for images saved by labview devices.
- Can lower the resolution by setting size to between 0 and 1(no resizing).
- Can change the contrast ratio by specifying gamma value (rescale= between 0 and 10) or taking a log of counts (rescale='log')
- For autoscale counts, set rescale='auto'
- To set rescale to make the 
- Can change frame per second (fps=)

![alt text](GIF_sample.gif)


For details about the files/codes, contact Fumika (fisono@lbl.gov, fumika21@gmail.com)
