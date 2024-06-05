# Methodology

We have started with the SDK provided to multi-view foul recognition challange. First we have started with fixing some errors
from the provided code, cleaning it up and wrapping it up with Pytorch Lightning which allowed to us easily track experiments
using custom logger for Weights and Biases. Then we have started experimenting with the number of frames per second, start frame
and the end frame. Generally we found keeping the middle of the extracted clip in 75th frame to be the most efficient. We have
also tried testing with a higher amount of frames per second (24fps) but id did not provide us any better results. We have
also set up starting frame at 58 and ending frame at 92. We have been also experimenting with data augmentation such as
Gaussian Blur. We have also managed to improve results of augmentation by removing from original
implmentation random erasing but keeping all of the other transforms. Despite running on a large cluster with slurm we did not manage to make swin3d transformer as we have been experiencing
out of memory issues. Because of that we used mvitv2 which had close to the same results as swin3d.



