
import random
from os import listdir
from os.path import isfile, join
import os, shutil

from math import floor
from scipy.ndimage.interpolation import zoom, rotate

import imageio
import face_recognition
import cv2


import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt

## Face extraction

## set up the device
device = torch.device('cuda')

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length


class FaceFinder(Video):
    def __init__(self, path, load_first_face = True):
        super().__init__(path)
        self.faces = {}
        self.coordinates = {}  # stores the face (locations center, rotation, length)
        self.last_frame = self.get(0)
        self.frame_shape = self.last_frame.shape[:2]
        self.last_location = (0, 200, 200, 0)
        if (load_first_face):
            face_positions = face_recognition.face_locations(self.last_frame, number_of_times_to_upsample=2)
            if len(face_positions) > 0:
                self.last_location = face_positions[0]
    
    def load_coordinates(self, filename):
        np_coords = np.load(filename)
        self.coordinates = np_coords.item()
    
    def expand_location_zone(self, loc, margin = 0.2):
        ''' Adds a margin around a frame slice '''
        offset = round(margin * (loc[2] - loc[0]))
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)
    
    @staticmethod
    def upsample_location(reduced_location, upsampled_origin, factor):
        ''' Adapt a location to an upsampled image slice '''
        y0, x1, y1, x0 = reduced_location
        Y0 = round(upsampled_origin[0] + y0 * factor)
        X1 = round(upsampled_origin[1] + x1 * factor)
        Y1 = round(upsampled_origin[0] + y1 * factor)
        X0 = round(upsampled_origin[1] + x0 * factor)
        return (Y0, X1, Y1, X0)

    @staticmethod
    def pop_largest_location(location_list):
        max_location = location_list[0]
        max_size = 0
        if len(location_list) > 1:
            for location in location_list:
                size = location[2] - location[0]
                if size > max_size:
                    max_size = size
                    max_location = location
        return max_location
    
    @staticmethod
    def L2(A, B):
        return np.sqrt(np.sum(np.square(A - B)))
    
    def find_coordinates(self, landmark, K = 2.2):
        '''
        We either choose K * distance(eyes, mouth),
        or, if the head is tilted, K * distance(eye 1, eye 2)
        /!\ landmarks coordinates are in (x,y) not (y,x)
        '''
        E1 = np.mean(landmark['left_eye'], axis=0)
        E2 = np.mean(landmark['right_eye'], axis=0)
        E = (E1 + E2) / 2
        N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
        B1 = np.mean(landmark['top_lip'], axis=0)
        B2 = np.mean(landmark['bottom_lip'], axis=0)
        B = (B1 + B2) / 2

        C = N
        l1 = self.L2(E1, E2)
        l2 = self.L2(B, E)
        l = max(l1, l2) * K
        if (B[1] == E[1]):
            if (B[0] > E[0]):
                rot = 90
            else:
                rot = -90
        else:
            rot = np.arctan((B[0] - E[0]) / (B[1] - E[1])) / np.pi * 180
        
        return ((floor(C[1]), floor(C[0])), floor(l), rot)
    
    
    def find_faces(self, resize = 0.5, stop = 0, skipstep = 0, no_face_acceleration_threshold = 3, cut_left = 0, cut_right = -1, use_frameset = False, frameset = []):
        '''
        The core function to extract faces from frames
        using previous frame location and downsampling to accelerate the loop.
        '''
        not_found = 0
        no_face = 0
        no_face_acc = 0
        
        # to only deal with a subset of a video, for instance I-frames only
        if (use_frameset):
            finder_frameset = frameset
        else:
            if (stop != 0):
                finder_frameset = range(0, min(self.length, stop), skipstep + 1)
            else:
                finder_frameset = range(0, self.length, skipstep + 1)
        
        # Quick face finder loop
        for i in finder_frameset:
            # Get frame
            frame = self.get(i)
            if (cut_left != 0 or cut_right != -1):
                frame[:, :cut_left] = 0
                frame[:, cut_right:] = 0            
            
            # Find face in the previously found zone
            potential_location = self.expand_location_zone(self.last_location)
            potential_face_patch = frame[potential_location[0]:potential_location[2], potential_location[3]:potential_location[1]]
            potential_face_patch_origin = (potential_location[0], potential_location[3])
    
            reduced_potential_face_patch = zoom(potential_face_patch, (resize, resize, 1))
            reduced_face_locations = face_recognition.face_locations(reduced_potential_face_patch, model = 'cnn')
            
            if len(reduced_face_locations) > 0:
                no_face_acc = 0  # reset the no_face_acceleration mode accumulator

                reduced_face_location = self.pop_largest_location(reduced_face_locations)
                face_location = self.upsample_location(reduced_face_location,
                                                    potential_face_patch_origin,
                                                    1 / resize)
                self.faces[i] = face_location
                self.last_location = face_location
                
                # extract face rotation, length and center from landmarks
                landmarks = face_recognition.face_landmarks(frame, [face_location])
                if len(landmarks) > 0:
                    # we assume that there is one and only one landmark group
                    self.coordinates[i] = self.find_coordinates(landmarks[0])
            else:
                not_found += 1

                if no_face_acc < no_face_acceleration_threshold:
                    # Look for face in full frame
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample = 2)
                else:
                    # Avoid spending to much time on a long scene without faces
                    reduced_frame = zoom(frame, (resize, resize, 1))
                    face_locations = face_recognition.face_locations(reduced_frame)
                    
                if len(face_locations) > 0:
                    print('Face extraction warning : ', i, '- found face in full frame', face_locations)
                    no_face_acc = 0  # reset the no_face_acceleration mode accumulator
                    
                    face_location = self.pop_largest_location(face_locations)
                    
                    # if was found on a reduced frame, upsample location
                    if no_face_acc > no_face_acceleration_threshold:
                        face_location = self.upsample_location(face_location, (0, 0), 1 / resize)
                    
                    self.faces[i] = face_location
                    self.last_location = face_location
                    
                    # extract face rotation, length and center from landmarks
                    landmarks = face_recognition.face_landmarks(frame, [face_location])
                    if len(landmarks) > 0:
                        self.coordinates[i] = self.find_coordinates(landmarks[0])
                else:
                    print('Face extraction warning : ',i, '- no face')
                    no_face_acc += 1
                    no_face += 1

        print('Face extraction report of', 'not_found :', not_found)
        print('Face extraction report of', 'no_face :', no_face)
        return 0
    
    def get_face(self, i):
        ''' Basic unused face extraction without alignment '''
        frame = self.get(i)
        if i in self.faces:
            loc = self.faces[i]
            patch = frame[loc[0]:loc[2], loc[3]:loc[1]]
            return patch
        return frame
    
    @staticmethod
    def get_image_slice(img, y0, y1, x0, x1):
        '''Get values outside the domain of an image'''
        m, n = img.shape[:2]
        padding = max(-y0, y1-m, -x0, x1-n, 0)
        padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return padded_img[(padding + y0):(padding + y1),
                        (padding + x0):(padding + x1)]
    
    def get_aligned_face(self, i, l_factor = 1.3):
        '''
        The second core function that converts the data from self.coordinates into an face image.
        '''
        frame = self.get(i)
        if i in self.coordinates:
            c, l, r = self.coordinates[i]
            l = int(l) * l_factor # fine-tuning the face zoom we really want
            dl_ = floor(np.sqrt(2) * l / 2) # largest zone even when rotated
            patch = self.get_image_slice(frame,
                                    floor(c[0] - dl_),
                                    floor(c[0] + dl_),
                                    floor(c[1] - dl_),
                                    floor(c[1] + dl_))
            rotated_patch = rotate(patch, -r, reshape=False)
            # note : dl_ is the center of the patch of length 2dl_
            return self.get_image_slice(rotated_patch,
                                    floor(dl_-l//2),
                                    floor(dl_+l//2),
                                    floor(dl_-l//2),
                                    floor(dl_+l//2))
        return frame


class VideoPredict:
    '''
    Gets a video as Input and
    returns the Prediction label as output
    '''

    def __init__(self,face_finder,model,target_size=256):
        self.finder = face_finder
        self.target_size = target_size
        self.head = 0 
        self.length = int(face_finder.length)
        self.model = model
        self.transform =  transforms.Compose([transforms.ToTensor()])
        


    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))

    def frames(self):
        self.total = len(self.finder.coordinates)
        while self.head<self.length:
            if self.head in self.finder.coordinates:
                path = 'temp_images/images/' + str(self.head)+'.jpg'
                patch = self.resize_patch(self.finder.get_aligned_face(self.head))## get the image
                imageio.imwrite(path,patch)
                print('Image Saved')
            self.head +=1
        return self.predict()

    def predict(self):
        ## Make data Loaders
        data = datasets.ImageFolder('temp_images',transform=self.transform)
        score = 0
        loader = torch.utils.data.DataLoader(data,batch_size=self.total)
        for data,labels in loader:
            data,labels = data.cuda(),labels.cuda()
            outputs = self.model(data)
            _,pred = torch.max(outputs,1)
        score = pred.cpu().numpy()
        label = sum(score)/self.total
        print(score,label,self.total)
        print('Decimal Score of this Video: ',label)
        self.flush()
        return int(label>0.4)
        
    def flush(self):
        folder = 'temp_images/images/'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print('Files Deleted')

def detect_face(gray,img,label):
    face_clf = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    face_img = img.copy()
    face_rects = face_clf.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    font = cv2.FONT_HERSHEY_DUPLEX
    for (x,y,w,h) in face_rects:
        if label==0:
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(face_img,'FAKE',(x+6,y-6),font,1.0,(255,255,255),1)
        else:
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(face_img,'REAL',(x+6,y-6),font,1.0,(255,255,255),1)

    return face_img


def fake_detect(path,label,name):
    '''
    Returns Bounding Box Detected Video as result
    '''
    print('Saving Video')
    input_movie = cv2.VideoCapture(path)
    frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height =int(input_movie.get( cv2.CAP_PROP_FRAME_HEIGHT))


    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('detect_'+str(name[:-4])+'.avi', fourcc, 1, (frame_height, frame_width))


    print('Starting Video')
    while True:
        ret,frame = input_movie.read() ## read the file
        if not ret:
            break 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detect_face(gray,frame,label)

        cv2.imshow('Video',frame)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    input_movie.release()
    cv2.destroyAllWindows()



def compute_accuracy(model, dirname,box=False, frame_subsample_count = 30):
    '''
    Extraction + Prediction over a video
    '''
    model.eval()
    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    predictions = {}
    
    for vid in filenames:
        print('Dealing with video ', vid)
        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep = skipstep)
        
        print('Predicting ', vid)
        predict = VideoPredict(face_finder,model)
        score = predict.frames()
        predictions[vid[:-4]] = score
        if box:
            path = join(dirname,vid)
            fake_detect(path,predictions[vid[:-4]],vid)

        print('----------------------------------------------------------------')
    return predictions







