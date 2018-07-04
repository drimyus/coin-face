from src.settings import *


class Coin:
    def __init__(self):
        pass

    def get_mask(self, img):
        if len(img.shape) == 2 or img.shape[-1] != 4:
            sys.stderr.write(" not find alpha channel ...")
            sys.exit(0)

        alpha = img[:, :, 3]
        kernel = np.ones((11, 11), dtype=int)
        alpha = cv2.dilate(alpha, kernel=kernel)
        return alpha

    def get_hole_pts(self, img):
        mask = self.get_mask(img)
        binary = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)[1]

        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cen_pts = []
        individual_masks = []
        for c in range(len(contours)):
            # [next, previous, first_child, parent] = hierarchy[0][c]
            (x, y, w, h) = cv2.boundingRect(contours[c])
            if x < w or not 0.5 < w / h < 2.0:
                continue
            cen_pt = np.array([.0, .0])
            for pt in contours[c]:
                cen_pt += pt[0] / len(contours[c])

            _individual_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            _individual_mask[y:y+h, x:x+w] = (255 - mask)[y:y+h, x:x+w]

            if len(cen_pts) == 0:
                cen_pts.append(cen_pt)
                individual_masks.append(_individual_mask)
            else:
                i = 0
                while i < len(cen_pts):
                    if cen_pt[0] > cen_pts[i][0]:
                        i += 1
                        continue
                    else:
                        break
                cen_pts.insert(i, cen_pt)
                individual_masks.insert(i, _individual_mask)

            # cv2.drawContours(img, contours, c, (0, 0, 255), 2)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return cen_pts, individual_masks


if __name__ == '__main__':
    img = cv2.imread("../data/Euro Coin2 three.png", cv2.IMREAD_UNCHANGED)
    Coin().get_hole_pts(img)