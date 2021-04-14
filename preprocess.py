import math

# 分帧处理函数
def enframe(wavData, frame_size, overlap):
'''Enframe the wave data.

Args:
    wavData: sequence of the sound wave data
        e.g. 
    frame_size: 

Return:
    frameData: sequence

'''
    coeff = 0.97#预加重系数
    wlen = len(wavData)
    step = frame_size - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frame_size, frameNum))
#汉明窗
    hamwin = np.hamming(frame_size)

    for i in range(frameNum):
        singleFrame = wavData[np.arange(i * step, min(i * step+frame_size,wlen))]
        singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff*singleFrame[1:])#预加重
        frameData[:len(singleFrame),i] = singleFrame
        frameData[:,i] = hamwin * frameData[:,i]#加窗
    return frameData
