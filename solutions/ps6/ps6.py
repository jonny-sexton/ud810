from ps6_functions import *
import cv2 as cv
import numpy as np
import time

from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *


def ps6_1_a():
    vid = "solutions/ps6/input/pres_debate.avi"

    n = 100

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = gray[156:314,320:428]
    cv.imwrite("solutions/ps6/output/ps6-1-a-1.png", face)
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(face, search_space, num_particles = 100, state_dims=2,
                             control_std=10, sim_std=20, alpha=0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:
            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame

        # store frames 28, 84, 144
        if count in ([28, 84, 144]):
            cv2.imwrite("solutions/ps6/output/ps6-1-a-" + str(save_count) + ".png",frame)
            save_count += 1


    cap.release()
    cv.destroyAllWindows()

def ps6_1_b():
    vid = "solutions/ps6/input/pres_debate.avi"

    n = 100

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # face = gray[106:364,270:478] # bigger
    face = gray[186:284,350:398] # smaller
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(face, search_space, num_particles = 100, state_dims=2,
                             control_std=10, sim_std=20, alpha=0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:
            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame


    cap.release()
    cv.destroyAllWindows()

    """
    TEXT RESPONSE:
    - Smaller window only concentrates on face and misses out on other features such as hair, etc.
    - Larger window takes more context in, e.g. hair, however also takes in other potentially (un)wanted features such as the background
    """

def ps6_1_c():
    vid = "solutions/ps6/input/pres_debate.avi"

    n = 100

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = gray[156:314,320:428]
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(face, search_space, num_particles = 100, state_dims=2,
                             control_std=10, sim_std=50, alpha=0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:
            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame


    cap.release()
    cv.destroyAllWindows()

    """
    TEXT RESPONSE:
    - Small theta_mse (sim_std) values lead to a "locked on" behaviour, after initialization the model stays put and does not jump to other potential candidates.
    - Large theta_mse (sim_std) values lead to a "noisy behaviour", the particles are more spread out and the estimated state jumps around the image.
    """

def ps6_1_d():
    vid = "solutions/ps6/input/pres_debate.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = gray[156:314,320:428]
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(face, search_space, num_particles = 200, state_dims=2,
                             control_std=10, sim_std=20, alpha=0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:
            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame


    cap.release()
    cv.destroyAllWindows()

    """
    TEXT RESPONSE:
    - A smaller number of particles (~10) leads to a fast but inaccurate behaviour, as there are not enough samples to find the correct candidate.
    - A larger number of particles (~1000) leads to a much more accurate result, however the performance (fps) is vastly reduced. 
    - An optimal setting of ~200 particles was found running on a MacBook Pro M1 2021
    """

def ps6_1_e():
    vid = "solutions/ps6/input/noisy_debate.avi"

    n = 100

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = gray[156:314,320:428]
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(face, search_space, num_particles = 200, state_dims=2,
                             control_std=10, sim_std=20, alpha=0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:
            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame
        
        # store frames 28, 84, 144
        if count in ([14, 32, 46]):
            cv2.imwrite("solutions/ps6/output/ps6-1-e-" + str(save_count) + ".png",frame)
            save_count += 1


    cap.release()
    cv.destroyAllWindows()

def ps6_2_a():
    vid = "solutions/ps6/input/pres_debate.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hand = gray[369:483,542:602]
    cv.imwrite("solutions/ps6/output/ps6-2-a-1.png", hand)
    search_space = np.array(gray.shape)
    w, h = hand.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(hand, search_space, num_particles = 700, state_dims=2,
                             control_std=10, sim_std=5, alpha=0.3)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:

            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:hand.shape[0], :hand.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame

        # # store frames 28, 84, 144
        if count in ([15, 50, 140]):
            cv2.imwrite("solutions/ps6/output/ps6-2-a-" + str(save_count) + ".png",frame)
            save_count += 1

    cap.release()
    cv.destroyAllWindows()

def ps6_2_b():
    vid = "solutions/ps6/input/noisy_debate.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hand = gray[369:483,542:602]
    cv.imwrite("solutions/ps6/output/ps6-2-b-1.png", hand)
    search_space = np.array(gray.shape)
    w, h = hand.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PF(hand, search_space, num_particles = 700, state_dims=2,
                             control_std=10, sim_std=5, alpha=0.5)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:

            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:hand.shape[0], :hand.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame

        # store frames 28, 84, 144
        if count in ([15, 50, 140]):
            cv2.imwrite("solutions/ps6/output/ps6-2-b-" + str(save_count) + ".png",frame)
            save_count += 1

    """
    TEXT RESPONSE:
    - Increased alpha so that the model reacted faster to the noise
    """

def ps6_3_a():
    vid = "solutions/ps6/input/pres_debate.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = gray[156:314,320:428]
    cv.imwrite("solutions/ps6/output/ps6-3-a-1.png", face)
    search_space = np.array(gray.shape)
    w, h = face.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PFMSLPF(face, search_space, num_particles = 1000, state_dims=2,
                             control_std=10, sim_std=10, alpha=0.0)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:

            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:face.shape[0], :face.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame

        # # store frames 28, 84, 144
        if count in ([28, 84, 144]):
            cv2.imwrite("solutions/ps6/output/ps6-3-a-" + str(save_count) + ".png",frame)
            save_count += 1

def ps6_3_b():
    vid = "solutions/ps6/input/pres_debate.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hand = gray[369:483,542:602]
    cv.imwrite("solutions/ps6/output/ps6-3-b-1.png", hand)
    search_space = np.array(gray.shape)
    w, h = hand.shape
    # cv.rectangle(gray,(320,155),(428,314),(0,0,255),3)
    # cv.imshow('face', face); cv.waitKey(0)

    tracker = Video_Tracker_PFMSLPF(hand, search_space, num_particles = 1000, state_dims=2,
                             control_std=10, sim_std=10, alpha=0.01)

    count = 1  # a frame has already been retrieved
    save_count=2  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        # track model in frame
        tracker.update(gray)

        tracker.visualize_filter(frame)
        # cv2.imshow("", frame);cv2.waitKey(0)
        if len(tracker.model.shape) < 3:

            color_model = cv2.cvtColor(tracker.model, cv2.COLOR_GRAY2BGR)

        frame[:hand.shape[0], :hand.shape[1]] = color_model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow("", frame)  # Display the resulting frame

        # # store frames 28, 84, 144
        if count in ([28, 84, 144]):
            cv2.imwrite("solutions/ps6/output/ps6-3-b-" + str(save_count) + ".png",frame)
            save_count += 1

def ps6_4_a():
    pass

def ps6_4_b():
    pass


ps6_list = OrderedDict([('1a', ps6_1_a), ('1b', ps6_1_b), ('1c', ps6_1_c), ('1d', ps6_1_d), ('1e', ps6_1_e), ('2a', ps6_2_a), ('2b', ps6_2_b), ('3a', ps6_3_a), ('3b', ps6_3_b), ('4a', ps6_4_a), ('4b', ps6_4_b)])

if __name__=="__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] in ps6_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps6_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,1b,1c,1d,1e,2a,2b,3a,3b,4a,4b} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps6_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps6_list[idx]()