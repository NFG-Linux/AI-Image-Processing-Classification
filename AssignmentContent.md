                                                Part 1:
I had the AI system generate comments for each line of the base classifier and it lined up with what I thought the script was doing (all the imports, initializing the model, loading and formatting the image,then processing the image through the model and then outputting predictions/confidence scores, error handling, and image path)
1st execution of base classifier gave the below output:
1: lab_coat (0.13)
2: Windsor_tie (0.09)
3: academic_gown (0.06)

I incorporated the Grad-CAM functionality to the script and asked the AI tool to explain it. It is basically a program that peeks into the CNN (Convolutional Neural Network) and determiines which parts of the image mattered most when the CNN made its decision and then overlays a heatmap over the original picture to show the percieved areas of importance. basic flow: picks a late convolutional layer, measures how important each feature map is, then averages the gradients (1 weight per map), combines the feature maps using weights, and then outputs the overlay.
After I got the Grad-CAM functions to work, at first it wasn't generating a new file it was trying to just output via GUI; after I got it to save to a new file, I didn't like the heatmap being in the background(avatar-256-heatmap1.jpg), so I changed it(avatar-256-heatmap.jpg). It makes sense that it is mostly looking at my face, but strangely not my whole body outline. Doesn't make sense that it's looking at my face and what it's classifying as though.

                                                Part 2:
The AI tool chose to use "drop-in" occlusion tools that all use a rectangle; one function blanks it out, the 2nd blurs it, and the 3rd pixelates it.
Below are the occlusion results. Interesting predictions:
Top-3 Predictions for occlusion #1 (avatar-256-occ1.jpg):
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
Top-3 Predictions:
1: abaya (0.08)
2: window_shade (0.03)
3: dishwasher (0.03)

Top-3 Predictions for occlusion #2 (avatar-256-occ2.jpg):
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step
Top-3 Predictions:
1: red_wine (0.04)
2: spotlight (0.03)
3: punching_bag (0.03)

Top-3 Predictions for occlusion #3 (avatar-256-occ3.jpg):
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
Top-3 Predictions:
1: shower_curtain (0.10)
2: container_ship (0.04)
3: soap_dispenser (0.04)

Well I think the classifier struggled with the original picture, so it definitely had problems classifying the image with the occlusion. At least the original was based on something a person may wear, but the occluded predictions are all waaaaay off. I would say that the second occlusion (blurring) had the biggest impact because the confidence scores were the lowest for that one. they all were clearly no where close to classifying correctly though.

                                                Part 3:
The basic_filter.py script uses Pillow to load and resize an image, applies a Gaussian blur, and then uses Matplotlib to display and save the result. The comments match what I was expecting each line to do, but I've never used the plt. commands, so it was good to get clarification for those lines.
I asked the AI for 3 additional filters to implement (and I didn't name the examples from FSO) and it outputted to use sharpen, edge detection, and emboss
For my own filter, I chose to have the image split and then each side was swapped for the other side. It has the ability to split vertically or horizontally, and I call both in the script. (vertical/horizontal-split_image.png)

The classifiers behavior is interesting because it is focusing mostly on my face, but outputting weird predictions. It's predictions were even worse when the occlusions were applied over the hot area of the heat map. The AI tool is getting easier to work with, but I think the hardest part was getting the filters/affects to apply in the correct field (background/foreground) and then getting the opache levels to how I wanted. I also had an issue at first with the occlusions because it was only applying a tiny rectangle in the top left corner of the picture, not resizing the picture to the expected w x h and then using the heatmap to apply the occlusion. The other weirdness I didn't like with the AI was the comments. I copied in the comments it gave me, but I found that it changes when it puts comments in line and to the right or above the line it's referencing. Technically doesn't affect anything, but professionally I would expect formatting to be the same throughout. I can't wait to try this script on other images!