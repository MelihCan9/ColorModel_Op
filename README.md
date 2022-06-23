# ColorModel_Op
RGB to YIQ and vice-versa to apply histogram equalization method. And another method to highlight the red color on an image.

Used HSV color model to initialize a mask for red color, after initializing the mask, I used several bitwise method from OpenCV library and finally I used the final mask
into the origin image. 

And the second method does the RGB to YIQ and vice-versa. Unlike matlab, opencv does not have a built-in method for this process. So I had to use certain matrices and matrix
multiplication. After first transformation I apply the histogram equalization method for just 'Y' component of the YIQ image. Because the Y component is the intensity component
of the YIQ images.

You can look into the code and the comment lines for further detail.
