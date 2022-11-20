import imutils

def center_crop(img, dim, with_resize = True):
    img_shape = img.shape
    height, width = img_shape[0], img_shape[1]
    
    if with_resize:
        if width > height:
            img = imutils.resize(img, height=dim[1])
        else:
            img = imutils.resize(img, width=dim[0])

	# process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < width else width
    crop_height = dim[1] if dim[1] < height else height 

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)

    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	
    return crop_img