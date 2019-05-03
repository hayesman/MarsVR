#!/usr/bin/env python

# Eric's tool for converting NASA rover images from binary data 
# to other image formats, such as JPEG, TIFF, or PNG, using info
# in the corresponding log files.

# Sample execution:
# nasa_img_converter.py -i input_image.IMG -o output_image.jpg

import argparse
import numpy as np
import math
from PIL import Image
import cv2


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input image file')
parser.add_argument('-o', '--output', type=str, help='output file')
args = parser.parse_args()

extension = args.input.split('.')[-1]
log = args.input.replace(extension, 'LBL')

# Get image metadata from the image's corresponding log file
try:
	f = open(log, "r")
	log_contents = f.read()
except IOError:
    print("Cannot open log: " + log)
metadata = {}
for line in log_contents.split('\n'):
	if '=' in line:
		# Parse fields from the log file
		metadata[line.split('=')[0].strip()] = line.split('=')[1].strip()

# Obtain values for relevant fields
width = int(metadata['LINE_SAMPLES'])
height = int(metadata['LINES'])
datatype = 'uint' + metadata['SAMPLE_BITS']

# Open image as binary data and load into NumPy array
f = open(args.input, mode='rb') 
data = np.fromfile(f, dtype=datatype)
header_size = data.size - (height * width)
data = data[header_size:]

# Determine bit depth (get bitmask from log file metadata)
bitmask = metadata['SAMPLE_BIT_MASK']
count = 0.0
for c in bitmask:
	if c == '1':
		count += 1.0
if count > 0:
	bitmask_int = math.pow(2, count)-1
else:
	bitmask_int = 4095.0

data.byteswap(inplace=True) # Swap from big to little endian

image_array = data.reshape(height, width) # Reshape data into 2D pixel array
image_float = image_array / bitmask_int # scale to float between 0.0 and 1.0
image_processed = np.power(image_float, (1/2.2)) # gamma curve adjustment
image_processed = image_processed * 255 # scale back to 0 to 255
image_8bit = image_processed.astype('uint8') # cast explicitly to uint8

# Save the processed binary data as an image
img = Image.fromarray(image_8bit)
img.save(args.output)

print "Output: %s" % args.output
