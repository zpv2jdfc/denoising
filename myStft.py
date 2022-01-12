import librosa
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import scipy.signal as signal
import pywt
from PyEMD import EMD,Visualisation
st = read(r'C:\Users\79301\Documents\GitHub\denoising\data\3.miniseed')
samples_rate = st.traces[0].meta.sampling_rate  # 采样率
data = st.traces[0].data  # 数据
data = data[76000:80000]
# n = 0
# for i in range(len(data)):
#     n = n + data[i]
# n = n / len(data)
# data = data - n
x = np.arange(4000)
# data = 2000*np.sin(x/2)
# data = data + 3000*np.sin(x/4+2.3)
# data = data + 4000*np.sin(x/8+2.3)
# st.spectrogram()
def shift(aa):
    for i in range(20):
        tmp = 0+0j
        tmp1 =0+0j
        tmp2 =0+0j
        for j in range(50):
            # tmp = tmp + aa[i][j]
            if(aa[i][j].real>0):
                tmp1+=aa[i][j]
            else:
                tmp2+=aa[i][j]
        tmp1 = tmp1/50
        tmp2 = tmp2/50
        for j in range(len(aa[i])):
            if(aa[i][j].real>0):
                aa[i][j] = aa[i][j] - tmp1
            else:
                aa[i][j] = aa[i][j] - tmp2
    return aa
def specdram(data,a=40,b=40,c=30):
    f, t, zxx = signal.stft(data,fs=a,nperseg=b,noverlap=c) #f:采样频率数组；t:段时间数组；Zxx:STFT结果
    plt.subplot(211)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.subplot(212)
    plt.pcolormesh(t, f, np.angle(zxx))
    plt.colorbar()
    plt.title('STFT Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()
def stft_scipy():

    signal.check_NOLA(signal.hann(128),128, 115)
    f, t, zxx = signal.stft(data,fs=40,nperseg=40,noverlap=30) #f:采样频率数组；t:段时间数组；Zxx:STFT结果
    plt.figure(figsize=(12, 8))
    zxx = shift(zxx)
    #振幅谱
    plt.subplot(411)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    # plt.show()
    #相位谱
    plt.subplot(412)
    plt.pcolormesh(t, f, np.angle(zxx))
    plt.colorbar()
    plt.title('STFT Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    #原始信号
    plt.subplot(413)
    plt.plot(range(len(data)),data)
    # 逆变换
    t, x = signal.istft(zxx, fs=40,nperseg=40,noverlap=30)
    plt.subplot(414)
    plt.plot(range(len(t)),x)
    plt.show()

    # if picname is not None:
    #     plt.savefig('..\\picture\\' + str(picname) + '.jpg')       #保存图像
def stft_librosa():
    noisy_mag, noisy_phase = librosa.magphase(librosa.stft(data+0.0, n_fft=16, hop_length=14, win_length=16))
    enhanced = librosa.istft(noisy_mag * noisy_phase, hop_length=14, win_length=16, length=len(data))
def wavelets():
    ca,cd = pywt.dwt(data,'db2')
    plt.figure(figsize=(12, 8))
    plt.subplot(711)
    plt.plot(range(len(data)),data)
    plt.subplot(712)
    plt.plot(range(len(ca)), ca)
    plt.subplot(713)
    plt.plot(range(len(cd)), cd)
    ca1,cd1 = pywt.dwt(ca, 'db2')
    plt.subplot(714)
    plt.plot(range(len(ca1)), ca1)
    plt.subplot(715)
    plt.plot(range(len(cd1)), cd1)
    res1 = pywt.idwt(ca1,np.zeros(len(cd1)),'db2')
    plt.subplot(716)
    plt.plot(range(len(res1)), res1)
    # res2 = pywt.idwt(res1, np.zeros(len(cd)), 'db2')
    # plt.subplot(717)
    # plt.plot(range(len(res2)), res2)
    plt.show()
def testEMD(imfs, res):
    plt.pcolormesh(range(len(imfs[0])), range(len(imfs)), imfs)
    plt.colorbar()
    plt.title('STFT Magnitudeqqq')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()
    newImf = imfs
    for i in range(len(imfs)):
        f, t, zxx = signal.stft(imfs[i], fs=40, nperseg=40, noverlap=30)  # f:采样频率数组；t:段时间数组；Zxx:STFT结果
        _,newImf[i] = signal.istft(shift(zxx), fs=40, nperseg=40, noverlap=30)
    for i in range(len(newImf)):
        # specdram(newImf[i])
        res = res + newImf[i]
    specdram(newImf[0],40,40,30)
    specdram(newImf[1],40,128,96)
    specdram(newImf[1]+newImf[2]+newImf[3],40,256,128)
    specdram(newImf[4] + newImf[2] + newImf[3],40,256,192)
    specdram(newImf[5] + newImf[6] + newImf[7],40,128,96)
    #原始信号
    plt.subplot(311)
    plt.plot(range(len(data)),data)
    #只经过stft
    plt.subplot(312)
    f, t, zxx_stft = signal.stft(data, fs=40, nperseg=40, noverlap=30)
    _,ttt = signal.istft(shift(zxx_stft), fs=40, nperseg=40, noverlap=30)
    plt.plot(range(len(ttt)), ttt)
    #emd + stft
    plt.subplot(313)
    plt.plot(range(len(res)),res)
    err1 = 0
    err2 = 0
    err3=0
    n1 = 0
    n2 = 0
    for i in range(500):
        err1 = err1 + abs(res[i] - data[i])
    for i in range(500):
        err2 = err2 + abs(ttt[i] - data[i])
    for i in range(500):
        err3 = err3 + abs(ttt[i] - res[i])
    for i in range(500):
        n1 = n1 + abs(res[i])
    for i in range(500):
        n2 = n2 + abs(ttt[i])
    plt.show()
def emd():
    emd = EMD()
    emd.emd(data)
    imfs, res = emd.get_imfs_and_residue()
    testEMD(imfs, res)
    # In general:
    # components = EEMD()(S)
    # imfs, res = components[:-1], components[-1]
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=range(len(data)), include_residue=True)
    vis.plot_instant_freq(range(len(data)), imfs=imfs)
    vis.show()
def cwt_pywt():
    sampling_rate = 40
    wavename = 'cgau8'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(range(len(data)), data)
    plt.xlabel("time")
    plt.title("frequency")
    plt.subplot(212)
    plt.contourf(range(len(data)), frequencies[220:256], abs(cwtmatr[220:256]))
    plt.ylabel("frequency")
    plt.xlabel("time")
    plt.subplots_adjust(hspace=0.4)
    pywt.wavedec(data, 'db1', level=2)
    plt.show()
def emd_stft():
    emd = EMD()
    emd.emd(data)
    imfs, res = emd.get_imfs_and_residue()
    #高频部分
    f, t, zxx = signal.stft(imfs[0], fs=40, nperseg=32, noverlap=24)
    plt.subplot(411)
    plt.pcolormesh(t, f, np.abs(zxx))
    f, t, zxx = signal.stft(imfs[0], fs=40, nperseg=128, noverlap=96)
    plt.subplot(412)
    plt.pcolormesh(t, f, np.abs(zxx))
    #低频部分
    plt.subplot(413)
    f, t, zxx = signal.stft(imfs[2]+imfs[3]+imfs[4]+imfs[5]+imfs[6]+imfs[7], fs=40, nperseg=40, noverlap=30)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.subplot(414)
    f, t, zxx = signal.stft(imfs[2]+imfs[3]+imfs[4]+imfs[5]+imfs[6]+imfs[7], fs=40, nperseg=128, noverlap=96)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.show()
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=range(len(data)), include_residue=True)
    vis.plot_instant_freq(range(len(data)), imfs=imfs)
    vis.show()
#小波频谱
def wavelet_spec():
    ca, cd = pywt.dwt(data, 'db2')
    caa1,cad1 = pywt.dwt(ca, 'db2')
    cda1,cdd1 = pywt.dwt(cd, 'db2')
    zxx = [0,0,0,0,0,0,0,0]
    zxx[0], zxx[1] = pywt.dwt(caa1, 'db2')
    zxx[2], zxx[3] = pywt.dwt(cad1, 'db2')

    zxx[4], zxx[5] = pywt.dwt(cda1, 'db2')
    zxx[6], zxx[7] = pywt.dwt(cdd1, 'db2')

    plt.pcolormesh(range(len(zxx[0])),range(len(zxx)), zxx)

    plt.show()
def test():
    specdram(data)
    emd()