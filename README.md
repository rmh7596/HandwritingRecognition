## Handriting Recongition Using A Convolutional Neural Network
Created a Convolutional Neural Network to classify my lowercase handwriting based off of manually-acquired images.

I utilized data augmentation to create more images programatically. Each image was split into 4 sub-images that were resized using a resampling technique. This allowed me to acquire more images without having to manually take more pictures. This moved the letter to different quadrants of the image to try to factor out letter position in the model. 

I achieved a maximum accuracy of 82%. I felt like this was a good stopping point to avoid over-fitting the model. This was achieved using 3 CNN layers and 50 epochs. 

![Results] (https://github.com/rmh7596/HandwritingRecognition/blob/main/Results/FinalResults.png)
