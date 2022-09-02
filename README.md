## Handriting Recongition Using A Convolutional Neural Network
Created a Convolutional Neural Network to classify my lowercase handwriting based off of manually-acquired images.

I utilized data augmentation to create more images programatically. Each image was split into 4 sub-images that were resized using a resampling technique(bilinear resampling). This allowed me to acquire more images without having to manually take more pictures. This moved the letter to different quadrants of the image to try to factor out letter positioning. 

I achieved a maximum accuracy of 82%. I felt like this was a good stopping point to avoid over-fitting the model. This was achieved using 3 CNN layers and 50 epochs. 

Libraries utilized = PyTorch, Pillow (PIL), Numpy, MatPlotLib 

<img src=Results\FinalResults.png>