import cv2
import datetime
import argparse

from handtracking.utils import detector_utils as detector_utils
from pyvjoy.vjoydevice import *

detection_graph, sess = detector_utils.load_inference_graph()

MAX_VJOY = 32767
MAX_STEERING_SLOPE = 1000

j = VJoyDevice(1)

def get_centroid_of_box(box):
    return box[0]+(box[2]-box[0])/2, box[1]+(box[3]-box[1])/2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    gas_percent = 0
    brake_percent = 0
    steering_slope = 0

    brake_hand = [0,0,0,0]
    gas_hand = [1,1,1,1]
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # find boxes with high enough score
        valid_boxes = [boxes[i] for i in range(len(boxes)) if scores[i] > args.score_thresh]

        # update controller values
        if len(valid_boxes) >= 2:
            for i in range(2):
                left, top, right, bottom = (valid_boxes[i][1] * im_width, valid_boxes[i][0] * im_height,
                                            valid_boxes[i][3] * im_width, valid_boxes[i][2] * im_height)

                centroid = get_centroid_of_box((left, top, right, bottom))

                if centroid[0] < im_width/2: # left
                    # brake
                    brake_percent = (bottom-top) / im_height
                    brake_hand = [left, top, right, bottom]
                else: # right
                    #gas
                    gas_percent = (bottom-top) / im_height
                    gas_hand = [left, top, right, bottom]

                cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (77, 255, 9), 3, 1)

        steering_slope = (get_centroid_of_box(brake_hand)[1] - get_centroid_of_box(gas_hand)[1]) / (get_centroid_of_box(brake_hand)[0] - get_centroid_of_box(gas_hand)[0])

        print("Gas Percent:", gas_percent, "Brake Percent:", brake_percent, "Steering Slope:", steering_slope)

        # update controller values
        j.data.wThrottle = int(gas_percent * MAX_VJOY)
        j.data.wWheel = int((steering_slope/MAX_STEERING_SLOPE) * MAX_VJOY)

        j.update()

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                      cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))