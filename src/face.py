from src.settings import *


def cv_rect2dlib_rect(cv_rect):
    (x, y, w, h) = cv_rect.astype(dtype=np.long)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    return rect


class Face:
    def __init__(self, detect_mode=DETECTOR_MODE):
        detector_dir = DETECTOR_DIR
        self.detect_mode = detect_mode

        # init the dlib's face detector
        if detect_mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        elif detect_mode == 'haar':
            # init the opencv's cascade haarcascade
            front_detector_path = os.path.join(detector_dir, "haarcascade_frontalface_alt2.xml")
            profile_detector_path = os.path.join(detector_dir, "haarcascade_profileface.xml")
            if not os.path.isfile(front_detector_path) or not os.path.isfile(profile_detector_path):
                sys.stderr.write("no exist detector.\n")
                sys.exit(1)
            self.front_detector = cv2.CascadeClassifier(front_detector_path)
            self.profile_detector = cv2.CascadeClassifier(profile_detector_path)

        else:
            sys.stderr.write("no defined detector mode.\n")
            sys.exit(1)

        # init the dlib's face shape predictor
        shape_predictor_path = os.path.join(detector_dir, "shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(shape_predictor_path):
            sys.stderr.write("no exist shape_predictor.\n")
            sys.exit(1)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect_faces(self, frame):
        if self.detect_mode == 'dlib':
            rects = self.detector(frame, 0)
            return rects
        elif self.detect_mode == 'haar':
            front_rects = self.front_detector.detectMultiScale(frame, scaleFactor=1.6, minNeighbors=1)
            cv_rects = list(front_rects)
            # profile_rects = self.profile_detector.detectMultiScale(frame, scaleFactor=1.6, minNeighbors=1)
            # cv_rects += list(profile_rects)

            # convert cv_rect to dlib_rect
            rects = []
            for cv_rect in cv_rects:
                rect = cv_rect2dlib_rect(cv_rect)
                rects.append(rect)
            return rects

    def get_landmarks(self, frame, rect):
        return np.array([[p.x, p.y] for p in self.shape_predictor(frame, rect).parts()])

    def get_cen_pt(self, landmarks, group):
        cen_pt = np.array([0, 0])
        for i in group:
            cen_pt += np.array(landmarks[i])
        return cen_pt/len(group)

    def get_pts(self, landmarks):
        return [
            self.get_cen_pt(landmarks, RIGHT_EYE_POINTS),
            self.get_cen_pt(landmarks, MOUTH_POINTS),
            self.get_cen_pt(landmarks, LEFT_EYE_POINTS)
        ]


if __name__ == '__main__':
    coin = cv2.imread("../data/Euro Coin2 three.png")
    # cv2.imshow("coin", coin)
    # cv2.waitKey(0)
    print(coin.shape[:2])

    img = cv2.imread("../data/index.jpeg")
    rect = Face().detect_faces(img)[0]
    landmarks = Face().get_landmarks(img, rect)
    print(Face().get_left_pt(landmarks))
    print(img.shape[:2])
