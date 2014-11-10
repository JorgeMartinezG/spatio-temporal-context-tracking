import cv
import cv2
import math
import numpy as np
import os
import sys
from scitools import numpyutils  as sn

def get_context(im, pos, sz, window):
    # Get and process context region.
    xs = pos[1] + np.array(range(sz[1])) + 1 - (sz[1]/2)
    ys = pos[0] + np.array(range(sz[0])) + 1 - (sz[0]/2)

    # Check for out-of-bounds coordinates, and set them to the values
    # at the borders.
    if xs[0] < 0:
        xs = xs - xs[0]

    if ys[0] < 0:
        ys = ys - ys[0]

    if ys[-1] > im.shape[0]:
        ys = ys - ys[-1] + im.shape[0]

    if xs[-1] > im.shape[1]:
        xs = xs - xs[-1] + im.shape[1]

    slicey = slice(ys[0] ,ys[-1] + 1)
    slicex = slice(xs[0], xs[-1] + 1)

    # Extract image in context region.
    out = im[slicey, slicex].astype('d')
    out = out - np.mean(out)
    out = window * out

    return out


if __name__ == '__main__':

    dataset_folder = 'Datasets'
    video_name = sys.argv[1]
    frames_list = os.listdir(os.path.join(dataset_folder, video_name, 'img'))
    frames_list.sort()

    # Get initial rectangle [x, y, width, height].
    initstate = [161, 65, 75, 95]

    # Center target.
    pos = np.array([initstate[1] + initstate[3]/2.,
                    initstate[0] + initstate[2]/2.])

    # Initial target size.
    target_sz = np.array([initstate[3], initstate[2]])
    
    # Parameters according to the paper.
    padding = 1                               # Extra area.
    rho = 0.075                                 # Learning parameter rho.
    sz = target_sz * (1 + padding)             # Context region size.


    # Parameters of scale update - scale ratio, lambda, average frames.
    scale, lambada, num = 1, 0.25, 5
    # Pre-computed confidence map.
    alapha = 2.25

    rs, cs = sn.ndgrid(np.array(range(sz[0])) + 1 - sz[0]/2.,
                       np.array(range(sz[1])) + 1  - sz[1]/2.)

    dist = rs**2. + cs**2.

    conf = np.exp(-0.5 * np.sqrt(dist) / alapha)

    # normalization
    conf =conf / conf.sum(axis=0).sum()

    # frequency
    conff = np.fft.fft2(conf)

    # store pre-computed weight window
    hamming_window = np.outer(np.hamming(sz[0]),
                              np.hanning(sz[1]).T)

    # initial sigma for the weight function
    sigma = np.mean(target_sz)


    # use Hamming window to reduce frequency effect of image boundary
    window = hamming_window * np.exp(-0.5 * dist / sigma**2)

    # a normalized window
    window = window / window.sum(axis=0).sum()

    maxconf = []

    # Loop reading frames
    for f, frame in enumerate(frames_list):
        sigma = sigma * scale
        window = hamming_window * np.exp(-0.5 * dist / sigma**2)
        window = window / window.sum(axis=0).sum()
        # Image read.
        img = cv2.imread(os.path.join(dataset_folder, video_name,
                                      'img', frame))
        if img.shape[2] == 3:
            im = cv2.cvtColor(img, cv.CV_BGR2GRAY)

        context_prior = get_context(im, pos, sz, window)

        # Frame loop should be here...
        if f > 0:
            #calculate response of the confidence map at all locations
            confmap = np.fft.ifft2(Hstcf * np.fft.fft2(context_prior)).real
            # target location is at the maximum response
            [col, ], [row, ] = np.where(confmap == confmap.max())
            pos = pos - sz / 2. + [col, row]

            context_prior = get_context(im, pos, sz, window)
            
            conftmp = np.fft.ifft2(Hstcf * np.fft.fft2(context_prior)).real
            maxconf.append(conftmp.max())

            # update scale by Eq.(15)
            if f % (num + 5) == 0:
                scale_curr = 0

                for kk in range(num):
                    scale_curr = scale_curr + np.sqrt(maxconf[f - kk - 1] / maxconf[f - kk - 2])

                # update scale
                scale = (1 - lambada) * scale + lambada * (scale_curr / num )
            
        # Update the spatial context model h^{sc} in Eq.(9)
        context_prior = get_context(im, pos, sz, window)
        hscf = conff / np.fft.fft2(context_prior)

        if f == 0:
            # First frame, initialize the spatio-temporal context model.
            Hstcf = hscf
        else:
            # Update the spatio-temporal context model H^{stc} by Eq. (12)
            Hstcf = (1 - rho) * Hstcf + rho * hscf

        # Visualization.
        target_sz = target_sz * scale


        rect_position = np.hstack([pos[[1, 0]] - target_sz[[1, 0]]/2.,
                                  target_sz[[1, 0]]])

        print rect_position
        print pos

        import ipdb;ipdb.set_trace()

        init_point = tuple(rect_position[[0, 1]].astype(int))
        end_point = tuple([int(rect_position[0]) + int(rect_position[2]),
                          int(rect_position[1]) + int(rect_position[3])])
        cv2.rectangle(img, init_point, end_point, (0, 255, 255), 2)
        cv2.imshow('image', img)
        key = cv2.waitKey(30)
    
        if key & 0xFF == ord('q'):
            break

    print "Loop finished!"
