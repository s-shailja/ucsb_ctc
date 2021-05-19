import os
import numpy as np
import pandas as pd
import networkx as nx
import time as timing
import SimpleITK as sitk
import concurrent.futures

from networkx.algorithms import similarity


def compute_unique_vals(array: np.ndarray, return_counts: bool = False,
                        return_sorted: bool = True) -> (np.ndarray, np.ndarray):
    '''
    helper-func to calc unique vals and their frequency in a np.ndarray
    using the highly optimized pd.unique()
    significantly faster than np.unique() for smaller numbers of unique vals
    '''
    unique_vals = pd.unique(array.flatten('K'))

    if return_sorted:
        unique_vals = np.sort(unique_vals)

    if return_counts:
        occurrences = np.zeros(len(unique_vals), dtype=np.uint16)

        for idx, val in enumerate(unique_vals):
            occurrences[idx] = np.count_nonzero(array == val)

        return unique_vals, occurrences
    else:
        return unique_vals


def load_img_from_tiff(path2img: str) -> np.ndarray:
    """
    helper-func to parallelize loading imgs from tiff-files
    """
    img = sitk.ReadImage(path2img)
    img_array = sitk.GetArrayFromImage(img)

    return img_array


def save_img_as_tiff(img_array: np.ndarray, filename: str, save_dir: str):
    """
    helper-func to parallelize saving imgs as tiff-files
    """
    img = sitk.GetImageFromArray(img_array.astype("uint16"))
    sitk.WriteImage(img, os.path.join(save_dir, filename))


def cell_center_fast(seg_img: np.ndarray, labels: np.ndarray) -> dict:
    """
    faster version of cell_center()
    speed gained by reusing previously calculated labels
    """
    results = {}
    for label in labels:
        if label != 0:
            all_points_z, all_points_x, all_points_y = np.where(seg_img == label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label] = [avg_z, avg_x, avg_y]

    return results


def compute_cell_location_fast(seg_img: np.ndarray, all_labels: np.ndarray) \
                               -> nx.Graph:
    """
    faster version of compute_cell_location()
    speed gained by reusing previously calculated labels and
    by using cell_center_fast()
    """
    g = nx.Graph()
    centers = cell_center_fast(seg_img, all_labels)

    # Compute vertices
    for i in all_labels:
        if i != 0:
            g.add_node(i)

    # Compute edges
    for i in all_labels:
        if i != 0:
            for j in all_labels:
                if j != 0:
                    if i != j:
                        pos1 = centers[i]
                        pos2 = centers[j]
                        distance = np.sqrt((pos1[0] - pos2[0])**2 +
                                           (pos1[1] - pos2[1])**2 +
                                           (pos1[2] - pos2[2])**2)

                        g.add_edge(i, j, weight=distance)
    return g


def tracklet_fast(g1: nx.Graph, g2: nx.Graph, seg_img1: np.ndarray, seg_img2: np.ndarray,
                  maxtrackid: int, time: int, linelist: list, tracksavedir: str,
                  labels_img1: np.ndarray, labels_img2: np.ndarray) -> (int, list):
    """
    faster version of tracklet()
    speed gained by parallelizing IO and
    by parallelizing some computations
    """
    f1 = {}
    f2 = {}
    dict_associate = {}
    new_seg_img2 = np.zeros(seg_img2.shape)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        thread1 = executor.submit(cell_center_fast, seg_img1, labels_img1)
        thread2 = executor.submit(cell_center_fast, seg_img2, labels_img2)
        thread3 = executor.submit(g1.degree, weight='weight')
        thread4 = executor.submit(g2.degree, weight='weight')

        cellcenter1 = thread1.result()
        cellcenter2 = thread2.result()
        loc1 = thread3.result()
        loc2 = thread4.result()

    for ele1 in loc1:
        cell = ele1[0]
        f1[cell] = [cellcenter1[cell], ele1[1]]

    for ele2 in loc2:
        cell = ele2[0]
        f2[cell] = [cellcenter2[cell], ele2[1]]

    for cell in f2.keys():
        tmp_center = f2[cell][0]
        min_distance = seg_img2.shape[0]**2 + seg_img2.shape[1]**2 + \
                       seg_img2.shape[2]**2

        for ref_cell in f1.keys():
            ref_tmp_center = f1[ref_cell][0]
            distance = (tmp_center[0] - ref_tmp_center[0])**2 + \
                       (tmp_center[1] - ref_tmp_center[1])**2 + \
                       (tmp_center[2] - ref_tmp_center[2])**2

            if distance < min_distance:
                dict_associate[cell] = ref_cell
                min_distance = distance

    inverse_dict_ass = {}

    for cell in dict_associate:
        if dict_associate[cell] in inverse_dict_ass:
            inverse_dict_ass[dict_associate[cell]].append(cell)
        else:
            inverse_dict_ass[dict_associate[cell]] = [cell]

    maxtrackid = max(maxtrackid, max(inverse_dict_ass.keys()))

    for cell in inverse_dict_ass.keys():
        if len(inverse_dict_ass[cell]) > 1:
            for cellin2 in inverse_dict_ass[cell]:
                maxtrackid = maxtrackid + 1
                new_seg_img2[seg_img2 == cellin2] = maxtrackid
                string = '{} {} {} {}'.format(maxtrackid, time+1, time+1, cell)
                linelist.append(string)
        else:
            cellin2 = inverse_dict_ass[cell][0]
            new_seg_img2[seg_img2 == cellin2] = cell
            i = 0

            for line in linelist:
                i = i + 1
                if i == cell:
                    list_tmp = line.split()
                    new_string = '{} {} {} {}'.format(list_tmp[0], list_tmp[1],
                                                      time+1, list_tmp[3])
                    linelist[i-1] = new_string

    filename1 = 'mask' + '%0*d' % (3, time) + '.tif'
    filename2 = 'mask' + '%0*d' % (3, time+1) + '.tif'

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        thread1 = executor.submit(save_img_as_tiff, seg_img1, filename1, tracksavedir)
        thread2 = executor.submit(save_img_as_tiff, new_seg_img2, filename2, tracksavedir)

    return maxtrackid, linelist


def track_main_fast(seg_fold: str, track_fold: str):
    """
    faster version of track_main()
    speed gained by parallelizing IO,
    by reusing unique_vals and
    by parallelizing some computations
    """
    folder1 = track_fold
    folder2 = seg_fold
    times = len(os.listdir(folder2))
    maxtrackid = 0
    linelist = []
    total_start_time = timing.time()

    for time in range(times-1):
        print('linking frame {} to previous tracked frames'.format(time+1))
        start_time = timing.time()
        threshold = 100

        if time == 0:
            file1 = 'mask000.tif'
            img1 = sitk.ReadImage(os.path.join(folder2, file1))
            img1 = sitk.GetArrayFromImage(img1)
            img1_label, img1_counts = compute_unique_vals(img1, return_counts=True)

            for l in range(len(img1_label)):
                if img1_counts[l] < threshold:
                    img1[img1 == img1_label[l]] = 0

            labels = compute_unique_vals(img1)
            start_label = 0

            for label in labels:
                img1[img1 == label] = start_label
                start_label = start_label + 1

            img1 = sitk.GetImageFromArray(img1)
            sitk.WriteImage(img1, os.path.join(folder1, file1))

        file1 = 'mask' + '%0*d' % (3, time) + '.tif'
        file2 = 'mask' + '%0*d' % (3, time+1) + '.tif'
        path2file1 = os.path.join(folder1, file1)
        path2file2 = os.path.join(folder2, file2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thread1 = executor.submit(load_img_from_tiff, path2file1)
            thread2 = executor.submit(load_img_from_tiff, path2file2)

            img1 = thread1.result()
            img2 = thread2.result()

        if len(compute_unique_vals(img2)) < 2:
            img2 = img1
            img2_img = sitk.GetImageFromArray(img2)
            sitk.WriteImage(img2_img, os.path.join(folder2, file2))

        img2_label_counts = np.array(compute_unique_vals(img2, return_counts=True)).T
        i = 0

        adjusted_labels_img2 = False

        for label in img2_label_counts[:, 0]:
            if img2_label_counts[i, 1] < threshold:
                img2[img2 == label] = 0
                adjusted_labels_img2 = True
            i = i + 1

        if adjusted_labels_img2:
            labels_img2 = compute_unique_vals(img2)
        else:
            labels_img2 = img2_label_counts[:, 0]

        labels_img1 = compute_unique_vals(img1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thread1 = executor.submit(compute_cell_location_fast, img1, labels_img1)
            thread2 = executor.submit(compute_cell_location_fast, img2, labels_img2)

            g1 = thread1.result()
            g2 = thread2.result()

        if time == 0:
            for cell in compute_unique_vals(img1):
                if cell != 0:
                    string = '{} {} {} {}'.format(cell, time, time, 0)
                    linelist.append(string)
                maxtrackid = max(cell, maxtrackid)

        maxtrackid, linelist = tracklet_fast(g1, g2, img1, img2, maxtrackid,
                                             time, linelist, folder1, labels_img1, labels_img2)

        print('--------%s seconds-----------' % (timing.time() - start_time))

    filetxt = open(os.path.join(folder1, 'res_track.txt'), 'w')

    for line in linelist:
        filetxt.write(line)
        filetxt.write("\n")

    filetxt.close()
    print('whole time sequnce running time %s' % (timing.time() - total_start_time))
