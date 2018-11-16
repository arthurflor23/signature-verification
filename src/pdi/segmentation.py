import numpy as np
import pdi.image as image

def otsu(arr):
    hist = image.histogram(arr)
    total = len(arr) * len(arr[0])

    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0

    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0,256):
        sumT += (i * hist[i])

    for i in range(0,256):
        weightB += hist[i]
        weightF = total - weightB

        if (weightF <= 0):
            break
        if (weightB <= 0):
            weightB = 1

        sumB += (i * hist[i])
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = (weightB * weightF)
        varBetween *= (meanB-meanF) * (meanB-meanF)

        if (varBetween > current_max):
            current_max = varBetween
            threshold = i

    arr[arr <= threshold] = 0
    arr[arr > threshold] = 1
    return arr