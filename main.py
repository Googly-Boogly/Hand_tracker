import cv2
from handtrackingmodule import handTracker
import pyautogui


def get_screen_size():
    return pyautogui.size()


def main():
    """
    this function gets your camera and runs some deep learning to figure out the 21 points of your hand
    the points go as follows.
    0: wirst
    1-4 Thumb (from base to tip so 1 is the base of the thumb and 4 is the tip of the thumb)
    5-8 index finger
    9-12 middle finger
    13-16 ring finger
    17-20 little finger (winter is coming)
    :param: NA
    :return:  NA
    """
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    screen_width, screen_height = get_screen_size()
    print(f"{screen_width} <- SW:  :{screen_height} SH")
    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            if lmList[20][1] < 30:
                print('Left')
            elif lmList[4][1] > 600:
                print('Right')

        cv2.imshow("Video",image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()