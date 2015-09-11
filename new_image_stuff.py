import scipy.ndimage as img
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patch

four_connection = np.array([[0,1,0],[1,1,1],[0,1,0]])

test_image = rand.rand(1200).reshape((30,40))
test_image[5:7,15:20]=0
test_image[18:23,1:5]=0
test_image[17, 3:7]=0
test_image[22:27, 8:13]=0
test_image[0,:] = 0
test_image[-1,:] = 0
test_image[:,0] = 0
test_image[:,-1] = 0

plt.matshow(test_image)
plt.savefig('/Users/cathyc/Dropbox/plots/plots_for_notes/test_array_finding_holes.pdf')
plt.show()

threshold = 1e-4
only_holes = np.where(test_image <=threshold, 1, 0)
plt.matshow(only_holes)
plt.show()

labeled_holes, n_holes = img.label(only_holes, four_connection)
plt.matshow(labeled_holes)
plt.savefig('/Users/cathyc/Dropbox/plots/plots_for_notes/test_array_labeled_holes.pdf')
plt.show()

just_outside = np.where(labeled_holes == 1, 0, 1)
unsorted_border_pix, border_pix = detect_border_pixels(just_outside)
plt.scatter(outside_border_pix.T[0], outside_border_pix.T[1])
plt.show()


outside_as_1s = np.where(labeled_holes == 1, 1, 0)
new_outside_border_pix = detect_border_pixels(outside_as_1s)
just_one_hole = np.where(labeled_holes == 2, 1, 0)
hole_border_pix = detect_border_pixels(just_one_hole)

outside_border_poly = poly.Polygon(border_pix)
outside_border_poly.area()

plot_polygon(outside_border_poly)

whole_mask = construct_mask_with_holes(test_image)
plot_polygon(whole_mask)

def construct_mask_with_holes(weight_array, threshold=1e-4):
    #Take an image and construct a polygon that contains only the
    #part of the image greater than threshold.  Allows for holes in
    #the image.
    #Put a border of 0s around the image so I know that the label of
    #[0,0] is the label of the outermost border
    weight_array=np.array(weight_array)
    image_dims = np.array(weight_array.shape)
    image_with_border = np.zeros(image_dims+2)
    image_with_border[1:-1, 1:-1] = weight_array
    #Start by masking to a binary image that has 1s where we're below
    #threshold and 0s above
    four_connection = np.array([[0,1,0],[1,1,1],[0,1,0]])
    below_threshold = np.where(image_with_border <=threshold, 1, 0)
    #Use scipy to group the below threshold areas into distinct connected
    #regions
    labeled_regions, n_regions = img.label(below_threshold, four_connection)
    label_of_outermost = labeled_regions[0, 0]
    #Now loop through the regions and get the border pixels for each
    #distinct region.  The outermost gets treated differently from the
    #holes because the border finder needs there to be 0s on the edges
    #of the image
    not_outside = np.where(labeled_regions == label_of_outermost, 0, 1)
    outermost_border = detect_border_pixels(not_outside)
    outermost_border = order_border_pixels(outermost_border)
    polygon_mask = poly.Polygon(outermost_border)
    #These are the labels of the holes inside the image
    holes = np.where(np.arange(n_regions) != label_of_outermost - 1)[0]
    holes += 1
    #Loop through the holes (if there are any)
    for label in holes:
        this_hole = np.where(labeled_regions == label, 1, 0)
        hole_border = detect_border_pixels(this_hole)
        hole_border = order_border_pixels(hole_border)
        hole = poly.Polygon(hole_border)
        polygon_mask = polygon_mask - hole
    return polygon_mask
    

        
    
def detect_border_pixels(input_array, threshold=1e-5):
    #LIFTED FROM THE MAST ARCHIVE FOOTPRINT FINDER
    # http://hla.stsci.edu/Footprintfinder/footprintfinder.py
    #Make sure we have a numpy array and get the dimensions of it
    input_array=np.array(input_array)
    dimensions = input_array.shape
    #Initialize the things we'll need
    borderpixels   = []
    data = np.zeros((3, dimensions[1]), dtype=np.float)
    intdata = np.zeros((3, dimensions[1]), dtype=np.int)
    thisarray = np.zeros(dimensions[1], dtype=np.int)
    #Loop through the rows
    for i in range(dimensions[0]):
        #Make sure the temporary data array we'll be using is empty
        data[:, :] = 0
        #Figure out what range of rows we'll be looking at
        x1 = max(i-1, 0)
        x2 = min(i+2, dimensions[0])
        #Pull out the relevant rows
        data[x1-(i-1):x2-(i+2)+3, :] = input_array[x1:x2, :]
        #Make any NaNs into 0s
        data[np.isnan(data)] = 0
        #Make intdata 1 where there's nonzero stuff and 0 elsewhere
        intdata = np.where(data < threshold, 0, 1)
        #Initialize the collapsed array that tells you whether or not
        #a pixel is a border pixel
        thisarray[:] = 0
        thisarray += intdata[0, :] #Add the row above
        thisarray += intdata[2, :] #Add the row below
        thisarray[1:] += intdata[1, :-1]  #add one to the right 
        thisarray[:-1] += intdata[1, 1:] #Add one to the left
        thisarray[:-1] += intdata[0, 1:] #Add the one above to the right
        thisarray[:-1] += intdata[2, 1:] #Add the one below to the right
        thisarray[1:]  += intdata[0, :-1] #Add the one above to the left
        thisarray[1:]  += intdata[2, :-1] #Add the one below to the left
        #Collapse down to just one border pixel instead of two for each
        #interface
        thisarray *= intdata[1, :]
        #Flag the pixels that aren't surrounded entirely by 0s or 1s
        index = np.where((thisarray!=0) & (thisarray!=8))[0]
        #Add in all the border pixels in this row
        for yind in index:
            borderpixels.append([i, yind])
    #Make the border pixels 
    borderpixels = np.array(borderpixels)
    return borderpixels


def order_border_pixels(borderpixels):
    #----------------
    #Order the pixels
    n_pixels = max(borderpixels.shape)
    available_pixels = np.ones(n_pixels, dtype=bool)
    ordered_pixels = np.zeros((n_pixels, 2))
    #Start with the very first pixel because the starting place
    #doesn't matter
    ordered_pixels[0] = borderpixels[0]
    available_pixels[0] = False
    #Loop through all the pixels
    for ipix in np.arange(n_pixels-1)+1:
        #Where are we?
        current_pix = ordered_pixels[ipix-1]
        #Figure out which pixels are around
        one_left = np.array([current_pix[0] - 1, current_pix[1]])
        one_up = np.array([current_pix[0], current_pix[1] + 1])
        one_right = np.array([current_pix[0] + 1, current_pix[1]])
        one_down = np.array([current_pix[0], current_pix[1] - 1])
        # print current_pix
        # print one_left, one_right, one_up, one_down
        #Find the next pixel
        next_pix = None
        for point in [one_down, one_right, one_up, one_left]:
            #Do we have the pixel we're looking at in our list of pixels?
            first_elem_matches = (borderpixels[available_pixels][:,0]==point[0])
            second_elem_matches = (borderpixels[available_pixels][:,1]==point[1])
            pix_mask = np.array(first_elem_matches & second_elem_matches)
            if pix_mask.any():
                #We do!  Keep it.  If we have multiples, this is going to get
                #overwritten, and that's what we want.  The points are ordered
                #in increasing preference.
                next_pix = point
        if next_pix is None:
            raise ValueError("ERROR: I don't have a next pixel to choose.")
        #Figure out which index the next pixel has in the order
        #of borderpixels, mask that one out as available and add to list
        ordered_pixels[ipix]=next_pix
        first_elem_matches = (borderpixels[:,0]==next_pix[0])
        second_elem_matches = (borderpixels[:,1]==next_pix[1])
        next_pix_mask = np.array(first_elem_matches & second_elem_matches)
        next_pix_ind = np.arange(n_pixels)[next_pix_mask]
        available_pixels[next_pix_ind]=False
    return ordered_pixels
    


def plot_polygon(polygon, save_to='/Users/cathyc/Desktop/testdisplay.pdf'):
    #takes a Polygon object and plots it
    #Figure out which is the main contour
    n_contours = len(polygon)
    is_hole = [polygon.isHole(i) for i in np.arange(n_contours)]
    is_hole = np.array(is_hole)
    is_main = np.invert(is_hole)
    main_index = np.arange(n_contours)[is_main][0]
    #Set up what we need for a patch for the main guy
    verts = polygon.contour(main_index)
    verts.append(verts[0])
    codes = [path.Path.MOVETO] + [path.Path.LINETO]*(len(verts)-2)
    codes += [path.Path.CLOSEPOLY]
    main_outline = path.Path(verts, codes)
    #Make the figure
    fig = plt.figure()
    fig.set_size_inches(5, 5*polygon.aspectRatio())
    ax = fig.add_subplot(111)
    #Make the axes invisible
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    #Plot the main thing
    xs = np.array(verts)[:,0]
    ys = np.array(verts)[:,1]
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    main_patch = patch.PathPatch(main_outline, facecolor='Blue', lw=0, alpha=.2, zorder=1)
    ax.add_patch(main_patch)
    ax.plot(xs, ys, lw=1, zorder=1, color='k')
    #Now let's see if we have any more contours to plot
    for i in np.arange(n_contours)[is_hole]:
        verts = polygon.contour(i)
        verts.append(verts[0])
        codes = [path.Path.MOVETO] + [path.Path.LINETO]*(len(verts)-2)
        codes += [path.Path.CLOSEPOLY]
        outline = path.Path(verts, codes)
        xs = np.array(verts)[:,0]
        ys = np.array(verts)[:,1]
        this_patch = patch.PathPatch(outline, facecolor='White',
                                     lw=0, zorder=5)
        #make sure it has an outline and add the patch
        ax.plot(xs, ys, lw=1, zorder=1, color='k')
        ax.add_patch(this_patch)
    plt.savefig(save_to, bbox_inches='tight')
    plt.close()
    
    


borderpixels=np.array(borderpixels)
plt.scatter(borderpixels.T[0], borderpixels.T[1])
plt.savefig('/Users/cathyc/Dropbox/plots/plots_for_notes/test_array_border_pixel_locations_8connection.pdf')

