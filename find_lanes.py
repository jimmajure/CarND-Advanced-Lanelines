'''
Created on Jan 18, 2017

@author: jim
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
from collections import deque


class Undistorter():
    '''
    A class that undistorts images based on a file
    containing the distortion matrices.
    '''

    def __init__(self, pickle_file):
        dist_pickle = pickle.load(open(pickle_file, 'rb'))
        
        self.mtx = dist_pickle['mtx']
        self.dist = dist_pickle['dist']
        self.image_shape = (1280,720)
        
    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
class PerspectiveTransformer():
    '''
    A class that calculates a perspective transform matrix and 
    inverse matrix. Methods are available to applie both the transform
    and inverse transforms.
    '''

    def __init__(self, src, dest, image_size = (1280,720)):
        self.M = cv2.getPerspectiveTransform(src, dest)
        self.Minv = cv2.getPerspectiveTransform(dest,src)
        self.image_size = image_size
        
    def transform(self, img):
        return cv2.warpPerspective(img, self.M, self.image_size, flags=cv2.INTER_LINEAR)
    
    def transformInverse(self, img):
        return cv2.warpPerspective(img, self.Minv, self.image_size, flags=cv2.INTER_LINEAR)

class Timer():
    def __init__(self):
        self.dt = datetime.now().microsecond
        self.dt_lap = self.dt
        pass
    
    def lap(self):
        dt_lap = datetime.now().microsecond
        result = dt_lap - self.dt_lap
        self.dt_lap = dt_lap
        return result
    
    
def scale(x, range=(0,255)):
    '''
    Scale single-band image data to the specified range.
    '''
    pct = lambda x: 1.*(x-np.min(x))/(np.max(x)-np.min(x))
    scale = lambda percent: range[0]+percent*(range[1]-range[0])
    return np.uint8(scale(pct(x)))
    

def imread(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

def imwrite(file, img):
    cv2.imwrite(file,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def gradient(img, orient='x', ksize=3):
    assert orient in ['x','y'], "orient must be 'x' or 'y', was {}".format(orient)
    if 'x' == orient:
        result = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        result = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.abs(result)

def gradient_magnitude(img, ksize=3):
    sobelx = gradient(img, orient='x', ksize=ksize)
    sobely = gradient(img, orient='y', ksize=ksize)
    
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    return abs_sobelxy
    
def gradient_direction(img, ksize=3):    
    sobelx = gradient(img, orient='x', ksize=ksize)
    sobely = gradient(img, orient='y', ksize=ksize)
    return np.arctan(sobely/sobelx)
    
def mask(img,mask_range=(0,255)):
    sxbinary = np.zeros_like(img)
    sxbinary[(img >= mask_range[0]) & (img <= mask_range[1])] = 1
    return sxbinary

def gradient_thresh(gray, orient='x', ksize=3, mask_range=(0,255)):
    return mask(np.uint8(scale(gradient(gray, orient, ksize))), mask_range)

def gradient_mag_thresh(gray, ksize=3, mask_range=(0,255)):
    return mask(np.uint8(scale(gradient_magnitude(gray, ksize))),mask_range)

def gradient_dir_thresh(gray, ksize=3, mask_range=(0,255)):
    return mask(gradient_direction(gray, ksize),mask_range)
    
def plot_images(imgs):
    '''
    A utility that plots a collection of images in a matrix. 
    '''
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(len(imgs)/cols)
    if len(imgs) % cols > 0:
        rows += 1

    indexer = lambda r,c: r*cols+c
    f, axes = plt.subplots(rows, cols, figsize=(20,10))
    for r in range(rows):
        for c in range(cols):
            idx = indexer(r,c)
            if idx < len(imgs):
                image = imgs[idx][1]
                title = imgs[idx][0]
                if rows == 1:
                    axis = axes[c]
                else:
                    axis = axes[r,c]
                axis.set_title(title)
                if len(image.shape)>2:
                    axis.imshow(image)
                else:
                    axis.imshow(image, cmap='gray')
    
def plot_images_2d(imgs):
    cols = len(imgs[0])
    rows = len(imgs)

    f, axes = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            image = imgs[r][c][1]
            title = imgs[r][c][0]
            if rows==1:
                axis = axes[c]
            else:
                axis = axes[r,c]
            axis.set_title(title)
            if not image == None:
                if len(image.shape)>2:
                    axis.imshow(image)
                else:
                    axis.imshow(image, cmap='gray')

def step_prob(c,w,prob):
    '''
    Initialize an array with 1.0 values in a specified window.
    '''
    assert c>=0
    assert c<=len(prob)
    assert w % 2 == 1
#     print("w={}".format(w))
    
    wby2 = int((w-1)/2)
    l = np.maximum(0,int(c - wby2))
    r = np.minimum(len(prob)-1,int(c + wby2))
    
    prob.fill(0.0)
    for idx in range(l,r):
        prob[idx] = 1.0
    
    return prob
    
def max_window(arr):
    '''
    for a given array, find the center of the widest point of 
    the array containing consecutive values of 1.
    
    Note: very slow implementation. Not sure if there's a better way.
    '''
    c = 0
    max_c = 0
    num_ones = 0
    yes = False
    for idx in range(len(arr)):
        if arr[idx]:
            num_ones += 1
            yes = True
        else:
            if yes:
                yes = False
                if num_ones > max_c:
                    max_c = num_ones
                    c = idx - int(num_ones/2)
            num_ones = 0
                    
    return c
    
def rgb2hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls, hls[:,:,0], hls[:,:,1], hls[:,:,2]
    
def rgb2hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv, hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
def rgb2lab(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return hls, hls[:,:,0], hls[:,:,1], hls[:,:,2]
    
def filter_image(rgb):
    '''
    Filter an image to identify lane pixels.
    Include the histogram-equalized value of the following:
    rgb r band
    hsv v band
    lab l band
    '''
#     gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#     grdmag = cv2.equalizeHist(scale(gradient(gray, orient='x', ksize=7)))
#     grdmask = mask(grdmag,(252,255))
    
    r = rgb[:,:,0]
    req = cv2.equalizeHist(r)
    rmask = mask(req, (252, 255))
    
    _,_,_,v = rgb2hsv(rgb)
    veq = cv2.equalizeHist(v)
    vmask = mask(veq, (252, 255))
    
    _,l,_,b = rgb2lab(rgb)
    leq = cv2.equalizeHist(l)
    lmask = mask(leq, (252, 255))
    beq = cv2.equalizeHist(b)
    bmask = mask(beq, (252, 255))
    
    comb = np.zeros_like(r)
    comb[(rmask==1)|(vmask==1)|(lmask==1)|(bmask==1)] = 1
    return comb

def find_lane(img, c, w_init = 201):
    '''
    Find a lane line centered on the x value c in a binary lane image.
    
    Iterate through the rows of the image from bottom to top. 
    
    If a lane line is identified for a row, find its center-of-mass and
    use that as the center of the window for the next line.
    
    If no lane line is identified for a row, keep the same center, but expand
    window size for the next row.
    '''
    prob_img = np.zeros_like(img, dtype=np.float32)
    w_max = 201
    w_min = 35
    w = w_init
    prob = np.array([0.]*1280)
    for ln in range(719,0,-1):
        prob = step_prob(c,w, prob)
        row = img[ln]
        new_prob = row*prob
        total_prob = np.sum(new_prob)
        if total_prob > 0:
            prob_img[ln] = new_prob
        
            # put the probablity in where the highest was...
            c = max_window(new_prob)
            w = w_min
        else:
            w = np.minimum(w_max, w+10)
    return prob_img

def find_lane_fit(img, fit):

    fitx = lambda y: fit[0]*y**2 + fit[1]*y + fit[2]

    band = np.zeros_like(img)
    ys = np.array(list(range(720)))
    xs = fitx(ys)
    lines = np.array([xs,ys]).T
    cv2.polylines(band,np.int32([lines]),0,(1),50)
    
    return (band * img)

class Lane(object):
    '''
    A lane object composed of 2 lane lines. The lane object is responsible
    for:
    
    Accepting a binary lane line image, identifying the lane lines, 
    fitting a polynomial line to them and doing sanity checks to ensure
    the lane lines are correct.
    
    It also generates an overlay image in the original perspective that can
    be combined with the undistorted image.
    '''
    def __init__(self, per_trans):
        # the size of the suppored image
        self.image_size = (1280,720)
        # y values used for generating lines
        self.ys = np.array(range(720))*1.
        # the perspective transform object
        self.per_trans = per_trans
        # list of recent fits for the left lane line
        self.fit_lefts = deque(maxlen=3)
        #list of recent fits for the right lane line
        self.fit_rights = deque(maxlen=3)
        # the current number of consecutive bad frames
        self.bad = 0
        # the maximum number of consecutive bad frames
        self.max_bad = 0
        # the total number of bad frames
        self.total_bad = 0
        # the current best guess of where the left lane line is
        self.left_start = None
        # the current best guess of where the right lane line is
        self.right_start = None
        # m/pixel in the y axis
        self.ym_per_pix = 30/720 
        # m/pixel in the x axis
        self.xm_per_pix = 3.7/700 
        
        self.position = 0
        self.diffs = None
        self.curvatures = None
        
    def fit_lane(self, img):
        '''
        Fit a polynomial to the x and y points that represent
        lane-line pixels in the provided image.
        '''
        fit,fitm = None,None
        
        # generate the indices for  cells with a value of 1...
        rind,cind = np.indices(img.shape)
        xs = cind[img[rind,cind]==1]
        ys = rind[img[rind,cind]==1]

        if len(xs) > 0:
            fit = np.polyfit(ys, xs, 2)
            fitm = np.polyfit(ys*self.ym_per_pix, xs*self.xm_per_pix, 2)
            
        return fit,fitm
    
    def curvature(self, val, fit):
        return ((1 + (2*fit[0]*val + fit[1])**2)**1.5) /np.absolute(2*fit[0])
    
    def add_fit(self, left_fit, left_fitm, right_fit, right_fitm):
        '''
        Add the provided polynomial fits to the Lane, sanity checking to make
        sure that the curvature of the two lanes is within tolerance and that
        the lines are roughly parallel.
        
        If the lines appear to be incorrect, increment the bad counter. If there
        are too many consecutive bad frames, set the starting points to None.
        '''
        left_fitx, right_fitx = self.__genxs(left_fit, right_fit)
        
        diff = right_fitx - left_fitx
        
        diff = diff[[self.image_size[1]-1,int(self.image_size[1]/2),0]]
        
        # calculate the change in width from the beginning of the image to its midpoint
        diff01 = np.abs(diff[0]-diff[1])
        
        #save the diffs for some data analysis...
        if self.diffs != None:
            self.diffs = np.concatenate((self.diffs, diff))
        else:
            self.diffs = diff

        # calculate the curvatures
        self.left_c = self.curvature(self.ys[-1], left_fitm)
        self.right_c = self.curvature(self.ys[-1], right_fitm)
        
        # save the curvatures for some data analysis...
        if self.curvatures != None:
            self.curvatures = np.concatenate((self.curvatures,np.array([self.left_c, self.right_c])))
        else:
            self.curvatures = np.array([self.left_c, self.right_c])

        minc = np.abs(np.amin([self.left_c, self.right_c]))
        maxc = np.abs(np.amax([self.left_c, self.right_c]))

        result = True
        if maxc < minc*10: #and diff01 < 120:
            self.fit_lefts.appendleft(left_fit)
            self.fit_rights.appendleft(right_fit)
            
            seconds_per_hour = 60.0*60.0
            frames_per_second = 30.0
            km_per_hour = 80.0 # assume 80 km/h
            m_per_second = km_per_hour * 1000 / seconds_per_hour
            m_per_frame = m_per_second / frames_per_second
            
            y = self.image_size[1] - m_per_frame/self.ym_per_pix
            self.left_start = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
            self.right_start = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            
            lane_center = self.left_start + (self.right_start - self.left_start)/2
            self.position = ((self.image_size[0]/2)-lane_center) * self.xm_per_pix
            
        else:
            result = False
        
        return result
    
    def get_current_fit(self):
        if len(self.fit_lefts) > 0 and len(self.fit_rights)>0:
            lf = self.fit_lefts[-1]
            rf = self.fit_rights[-1]
            return lf, rf
        else:
            return None, None
            
    def get_average_fit(self):
        if len(self.fit_lefts) > 0 and len(self.fit_rights)>0:
            lf = np.mean(np.array(self.fit_lefts), axis=0)
            rf = np.mean(np.array(self.fit_rights), axis=0)
            return lf, rf
        else:
            return None, None
            
    
    def process_lanes(self, lanes):
        '''
        Accept a classified lane image, pull out the lanes, 
        fit polynomials and add the fits to the Lane.
        '''
        success = True
        left, right = None,None
        # get the place to start
        lf,rf = self.get_current_fit()
        if self.bad < 2 and lf != None:
            left = find_lane_fit(lanes, lf)
            right = find_lane_fit(lanes, rf)
        else:
            ls,rs = get_start_point(lanes)
            if ls and rs:
                # find the left lane line
                left = find_lane(lanes, ls)
                #find the right lane line
                right = find_lane(lanes, rs)
                
            
        # we have good starting points, so start small
        if None != left:
            left_fit,left_fitm = self.fit_lane(left)
                
                #find the right lane line
            right_fit,right_fitm = self.fit_lane(right)
                
                # add the current fits
            success = right_fit != None and left_fit != None and \
                self.add_fit(left_fit, left_fitm, right_fit, right_fitm)
        else:
            success = False

        if success:
            # reset the consecutive bad counter
            self.bad = 0
        else:
            self.bad += 1
            self.total_bad += 1
            self.max_bad = np.amax((self.max_bad, self.bad))

        print("\ncurvature:      {: >10.3f}|{: >10.3f}".format(self.left_c, self.right_c))
        print("bad/max/total: {}/{}/{}".format(self.bad,self.max_bad, self.total_bad))
        
        return success

        
    def __genxs(self, left_fit, right_fit, units='pixel'):
        if units == 'pixel':
            mult = 1.
        else:
            mult = self.ym_per_pix
        left_fitx = left_fit[0]*(self.ys*mult)**2 + left_fit[1]*(self.ys*mult) + left_fit[2]
        right_fitx = right_fit[0]*(self.ys*mult)**2 + right_fit[1]*(self.ys*mult) + right_fit[2]
        return left_fitx, right_fitx
    
    def generate_overlay(self):
        # Create an image to draw the lines on
        warp_zero = np.zeros((self.image_size[1],self.image_size[0])).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        lf,rf = self.get_average_fit()
        left_fitx, right_fitx = self.__genxs(lf, rf)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ys])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
        

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.per_trans.transformInverse(color_warp)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(newwarp,"curvature: {: >10.3f}|{: >10.3f}".format(self.left_c, self.right_c),(500,620), font, 0.5,(255,255,255),2)
        cv2.putText(newwarp,"lane position: {: >10.3f}".format(self.position),(500,650), font, 0.5,(255,255,255),2)

        return newwarp
    
    
def get_start_point(img):
    '''
    Create histogram values to estimate the likely x position of each of the two
    lane lines.
    '''
    sums = np.sum(img[400:], axis=0)
    m = int(1280/2)
    left_start = np.argmax(sums[:m])
    right_start = m+np.argmax(sums[m:])
    return left_start, right_start

def process_image(img):
    global counter
    # undistort the image
    img_und = undistorter.undistort(img)
    # do a perspective transform
    img_pt = per_trans.transform(img_und)
    # identify possible lane-line pixels
    lanes = filter_image(img_pt)
    if not lane.process_lanes(lanes):
#         imwrite("test_images/bad{}.jpg".format(lane.total_bad), img)
        pass
    
    counter += 1
    imwrite("test_images/frame{}.jpg".format(counter), img)
    overlay_img = lane.generate_overlay()
    result = cv2.addWeighted(img_und, 1, overlay_img, 0.3, 0)
    return result

if __name__ == '__main__':
    counter = 0
    undistorter = Undistorter('camera_cal/distortion_pickle.p')
    src = np.float32(
        [[100,720],
         [585,450],
         [695,450],
         [1180,720]]
        )
    dest = np.float32(
        [[200,720],
         [200,0],
         [1080,0],
         [1080,720]]
        )

    per_trans = PerspectiveTransformer(src, dest, (1280,720))
    lane = Lane(per_trans)
    
    from moviepy.editor import VideoFileClip
 
    yellow_output = 'project_out.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)
    
    pickle.dump( lane.diffs, open( "diffs.p", "wb" ) )
    pickle.dump(lane.curvatures, open("curvatures.p","wb"))
