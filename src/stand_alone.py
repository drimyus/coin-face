from src.face import Face
from src.coin import Coin
from src.settings import *

coin = Coin()
face = Face()


def proc(coin_img_path, face_video_path):
    l_scale = LEFT_EYE_SCALE
    r_scale = RIGHT_EYE_SCALE
    m_scale = MOUSE_SCALE

    coin_img = cv2.imread(coin_img_path, cv2.IMREAD_UNCHANGED)
    [coin_l_pt, coin_m_pt, coin_r_pt], [coin_l_mask, coin_m_mask, coin_r_mask] = coin.get_hole_pts(img=coin_img)

    coin_rgb = cv2.cvtColor(coin_img.copy(), cv2.COLOR_BGRA2BGR)
    coin_mask = cv2.cvtColor(coin.get_mask(coin_img), cv2.COLOR_GRAY2BGR) / 255.0
    coin_l_mask = cv2.cvtColor(coin_l_mask, cv2.COLOR_GRAY2BGR) / 255.0
    coin_m_mask = cv2.cvtColor(coin_m_mask, cv2.COLOR_GRAY2BGR) / 255.0
    coin_r_mask = cv2.cvtColor(coin_r_mask, cv2.COLOR_GRAY2BGR) / 255.0

    print([coin_l_pt, coin_m_pt, coin_r_pt])

    res_h, res_w = coin_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    saver = cv2.VideoWriter('result.avi', fourcc, 25.0, (res_w, res_h))

    cap = cv2.VideoCapture(face_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        rects = face.detect_faces(img)
        if len(rects) == 0:
            continue

        rect = rects[0]

        landmarks = face.get_landmarks(img, rect)
        face_l_pt, face_m_pt, face_r_pt = face.get_pts(landmarks)

        M_l = np.float32([
            [l_scale, 0, coin_l_pt[0] - face_l_pt[0] * l_scale],
            [0, l_scale, coin_l_pt[1] - face_l_pt[1] * l_scale]
        ])

        M_m = np.float32([
            [m_scale, 0, coin_m_pt[0] - face_m_pt[0] * m_scale],
            [0, m_scale, coin_m_pt[1] - face_m_pt[1] * m_scale]
        ])

        M_r = np.float32([
            [r_scale, 0, coin_r_pt[0] - face_r_pt[0] * r_scale],
            [0, r_scale, coin_r_pt[1] - face_r_pt[1] * r_scale]
        ])

        warp_l = cv2.warpAffine(img, M_l, (res_w, res_h))
        warp_m = cv2.warpAffine(img, M_m, (res_w, res_h))
        warp_r = cv2.warpAffine(img, M_r, (res_w, res_h))

        res = coin_rgb * coin_mask + coin_l_mask * warp_l + coin_m_mask * warp_m + coin_r_mask * warp_r
        res = res.astype(np.uint8)

        cv2.imshow('res', cv2.resize(res, (1000, 700)))
        saver.write(res)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    saver.release()


if __name__ == '__main__':
    proc("../data/Euro Coin2 three.png", "../data/THE FINAL MANUP 20171080P.mp4")
