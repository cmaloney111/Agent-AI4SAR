import numpy as np
from scipy.spatial import cKDTree

def randsmpl(p):
    return np.random.choice(len(p), p=p)

def run_replicate(initial_point, find_point, map_data, T, p_behavior, alpha, LL):
    # Define possible motions in body coords
    right = np.array([1, 0])
    left = np.array([-1, 0])
    front = np.array([0, 1])
    back = np.array([0, -1])
    stay = np.array([0, 0])
    frontRight = np.array([1, 1])
    frontLeft = np.array([-1, 1])
    backRight = np.array([1, -1])
    backLeft = np.array([-1, -1])

    possibleMotions = np.array([frontLeft, front, frontRight, left, stay, right, backLeft, back, backRight])

    # Initial positions
    X = np.zeros(T + 2)
    Y = np.zeros(T + 2)
    X[0], Y[0] = initial_point
    X[1] = X[0] + (np.floor(3 * np.random.rand()) - 1)
    Y[1] = Y[0] + (np.floor(3 * np.random.rand()) - 1)

    x = np.full(T + 2, np.nan)
    y = np.full(T + 2, np.nan)
    x[0], y[0] = X[0], Y[0]
    x[1], y[1] = X[1], Y[1]

    u = np.full(T + 1, np.nan)
    v = np.full(T + 1, np.nan)
    u[0], v[0] = x[1] - x[0], y[1] - y[0]

    vx = np.full(T + 2, np.nan)
    vy = np.full(T + 2, np.nan)
    xs = np.full(T + 2, np.nan)
    ys = np.full(T + 2, np.nan)

    behavior = np.full(T + 2, np.nan)

    flag = 0

    for ii in range(1, T + 1):
        if flag == 1:
            break

        # Choose behavior
        behavior[ii] = randsmpl(p_behavior)

        flag_sp = 0

        if behavior[ii] == 1:
            # Random traveling
            px_rw = np.ones(9) / 9
            inds = randsmpl(px_rw)
            motions = possibleMotions[inds]

            u, v = motions
            if x[ii] - x[ii - 1] == 0 and y[ii] - y[ii - 1] == 0:
                theta = 2 * np.pi * np.random.rand()
            else:
                theta = -np.arctan2(x[ii] - x[ii - 1], y[ii] - y[ii - 1])

            M = np.array([[np.cos(theta), -np.sin(theta), x[ii]],
                          [np.sin(theta), np.cos(theta), y[ii]],
                          [0, 0, 1]])

            temp = np.dot(M, np.array([u, v, 1]))
            x_rw, y_rw = round(temp[0]), round(temp[1])

            XY_temp = np.array([x_rw, y_rw])

        elif behavior[ii] == 2:
            # Route traveling
            px_rt = np.array([3, 3, 3, 0, 0, 0, 0, 0, 0]) / 9
            if x[ii] - x[ii - 1] == 0 and y[ii] - y[ii - 1] == 0:
                theta = 2 * np.pi * np.random.rand()
            else:
                theta = -np.arctan2(x[ii] - x[ii - 1], y[ii] - y[ii - 1])

            M = np.array([[np.cos(theta), -np.sin(theta), x[ii]],
                          [np.sin(theta), np.cos(theta), y[ii]],
                          [0, 0, 1]])

            BW_temp = np.full(9, np.nan)
            for jj in range(9):
                aux = possibleMotions[jj]
                aux1 = np.round(np.dot(M, np.array([aux[0], aux[1], 1]))).astype(int)[:-1]

                if aux1[1] > LL[1] or aux1[1] < LL[3] or aux1[0] > LL[0] or aux1[0] < LL[3]:
                    flag = 1
                    break
                
                BW_temp[jj] = map_data['BWLF'][aux1[1], aux1[0]]

            if flag == 1:
                break

            px_lin = px_rt * BW_temp
            px_lin /= np.sum(px_lin)
            px_lin[np.isnan(px_lin)] = 1 / 9

            inds_lin = randsmpl(px_lin)
            motions_lin = possibleMotions[inds_lin]

            u, v = motions_lin
            temp_lin = np.dot(M, np.array([u, v, 1]))
            x_rt, y_rt = round(temp_lin[0]), round(temp_lin[1])

            XY_temp = np.array([x_rt, y_rt])

        elif behavior[ii] == 3:
            # Direction traveling
            px_dt = np.array([0, 9, 0, 0, 0, 0, 0, 0, 0]) / 9
            inds = randsmpl(px_dt)
            motions = possibleMotions[inds]

            u, v = motions
            if x[ii] - x[ii - 1] == 0 and y[ii] - y[ii - 1] == 0:
                theta = 2 * np.pi * np.random.rand()
            else:
                theta = -np.arctan2(x[ii] - x[ii - 1], y[ii] - y[ii - 1])

            M = np.array([[np.cos(theta), -np.sin(theta), x[ii]],
                          [np.sin(theta), np.cos(theta), y[ii]],
                          [0, 0, 1]])

            temp = np.dot(M, np.array([u, v, 1]))
            x_dt, y_dt = round(temp[0]), round(temp[1])

            XY_temp = np.array([x_dt, y_dt])

        elif behavior[ii] == 4:
            # Staying put
            XY_temp = np.array([x[ii], y[ii]])
            flag_sp = 1

        elif behavior[ii] == 5:
            # View enhancing
            if x[ii] - x[ii - 1] == 0 and y[ii] - y[ii - 1] == 0:
                theta = 2 * np.pi * np.random.rand()
            else:
                theta = -np.arctan2(x[ii] - x[ii - 1], y[ii] - y[ii - 1])

            M = np.array([[np.cos(theta), -np.sin(theta), x[ii]],
                          [np.sin(theta), np.cos(theta), y[ii]],
                          [0, 0, 1]])

            int_temp = np.zeros(9)
            for kk in range(9):
                auxv = possibleMotions[kk]
                auxv1 = np.round(np.dot(M, np.array([auxv[0], auxv[1], 1]))).astype(int)[:-1]

                if auxv1[1] > LL[1] or auxv1[1] < LL[3] or auxv1[0] > LL[0] or auxv1[0] < LL[3]:
                    flag = 1
                    break

                int_temp[kk] = map_data['sZelev'][auxv1[1], auxv1[0]]

            if flag == 1:
                break

            int_temp -= int_temp[4]
            aux_max = np.max(int_temp)
            inds_ve = np.where(int_temp == aux_max)[0]
            np.random.shuffle(inds_ve)

            motions_ve = possibleMotions[inds_ve[0]]
            u, v = motions_ve
            temp_ve = np.dot(M, np.array([u, v, 1]))
            x_ve, y_ve = round(temp_ve[0]), round(temp_ve[1])

            XY_temp = np.array([x_ve, y_ve])

        else:
            # Backtracking
            if behavior[ii - 1] != 6:
                XY_temp = np.array([x[ii - 1], y[ii - 1]])
            else:
                BT_steps = 1
                while behavior[ii - BT_steps - 1] == 6:
                    BT_steps += 1
                ind_bt = max(ii - 2 * (BT_steps) - 1, 1)
                XY_temp = np.array([x[ind_bt], y[ind_bt]])

        # Update position
        vx[ii] = XY_temp[0] - x[ii]
        vy[ii] = XY_temp[1] - y[ii]
        xs[ii + 1] = round((2 - alpha) * x[ii] + (alpha - 1) * x[ii - 1] + alpha * vx[ii])
        ys[ii + 1] = round((2 - alpha) * y[ii] + (alpha - 1) * y[ii - 1] + alpha * vy[ii])

        XY_temp_s = np.array([xs[ii + 1], ys[ii + 1]])
        x[ii + 1], y[ii + 1] = XY_temp_s

        if not (LL[2] < xs[ii + 1] < LL[0]) or not (LL[3] < ys[ii + 1] < LL[1]):
            flag = 1
            break

        kdtree = cKDTree([(find_point[0], find_point[1])])
        temp_dist, temp_inds = kdtree.query([xs[ii + 1], ys[ii + 1]])

        if temp_dist <= 5:
            flag = 1
            break

        if flag_sp == 1:
            behavior[ii + 1] = 4
        else:
            behavior[ii + 1] = behavior[ii]

    endXY = XY_temp
    success = (np.linalg.norm(np.array([endXY[0], endXY[1]]) - np.array(find_point)) <= 5)
    return success, endXY, xs, ys


if __name__ == "__main__":
    from skimage.io import imread
    import matplotlib.pyplot as plt

    # Load image and depth map
    image = imread('image.tif', as_gray=True)
    depth_map = imread('image_d.tif')

    # Assume the image and depth map are the same size
    img_height, img_width = image.shape[:2]

    # Define initial and find points (e.g., random points in the image)
    initial_point = (np.random.randint(0, img_width), np.random.randint(0, img_height))
    find_point = (np.random.randint(0, img_width), np.random.randint(0, img_height))

    # Prepare map_data dictionary
    map_data = {
        'BWLF': image,      # The image or binary map for line following
        'sZelev': depth_map # Depth map (elevation data)
    }

    # Define boundaries (LL)
    LL = [img_width, img_height, 0, 0]  # Boundaries [max_x, max_y, min_x, min_y]

    # Define T (number of time steps), p_behavior, alpha for the simulation
    T = 100
    p_behavior = [0.5, 0, 0, 0, 0.5, 0]  # Probabilities for each behavior
        
    # 1 = RW
    # 2 = Route
    # 3 = direction
    # 4 = stay put
    # 5 = view en
    # 6 = backtrack
    alpha = 0.55  

    # Run the simulation
    success, endXY, xs, ys = run_replicate(initial_point, find_point, map_data, T, p_behavior, alpha, LL)

    # Output the result
    print("Simulation Success:", success)
    print("End Position:", endXY)

    # Plot the initial point, end point, and find point on the image
    plt.imshow(image, cmap='gray')
    plt.scatter(initial_point[0], initial_point[1], color='red', label='Initial Point')
    plt.scatter(find_point[0], find_point[1], color='blue', label='Find Point')
    plt.scatter(endXY[0], endXY[1], color='green', label='End Position')
    plt.plot(xs, ys, color='aqua', label='Path taken')
    plt.legend()
    plt.show()
