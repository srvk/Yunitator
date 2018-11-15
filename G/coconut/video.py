#!/opt/python27/bin/python

from janus import *

# -----------------------------

#kkk=FMatrix(2,3)

#janus.Tcl_Eval(interp, "FMatrix f 5 6")
#print janus.Tcl_GetStringResult(interp)

janus.itfGetObject.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
janus.itfGetObject.restype  = ctypes.POINTER(FMatrix)
#f_o = janus.itfGetObject ("f", "FMatrix")
#print f_o.contents.m

if False:
    kkk=DMatrix(4,3)
    kkk.array[0][0]=2
    
    lll=DMatrix(2,3)
    lll.matPA[0][0]=4
    
    mmm=DMatrix(4,2)
    mmm.mulot(kkk,lll)
    
    A=DMatrix(5,4)
    A.array[1][3]=3
    print A.array
    print A.array[1][3]
    
    b=numpy.array(([0,1,2,3],[6,7,8,9],[6,7,9,0],[0,1,2,4],[6,7,9,0]), numpy.float64)
    B=DMatrix(b)
    print B.array
    print B.array[1][3]
    
    c=janus.dmatrixCreate (5, 4)
    C=DMatrix(c)
    C.array[1][3]=4
    print C.array
    print C.array[1][3]
    print C.array.shape
    
    d=janus.fmatrixCreate(4,11)
    D=FMatrix(d)
    D.array[1][3]=4
    print D.array
    print D.array[1][3]

from scipy.io import wavfile
#fs, sig = wavfile.read('/people/fmetze1/RT_Alaska/adc/siptrans.wav')

janus.Tcl_Eval(interp, "FeatureSet fs")
#janus.Tcl_Eval(interp, "fs readADC ADC CHICKEN_RUN_DISC_D.m4v")
#janus.Tcl_Eval(interp, "fs spectrum FFT ADC 16msec")
#janus.Tcl_Eval(interp, "fs:")
#print janus.Tcl_GetStringResult(interp)
#ADC FFT 
janus.itfGetObject.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
janus.itfGetObject.restype  = ctypes.POINTER(FMatrix)
#f_o = janus.itfGetObject ("fs:FFT.data", "FMatrix")
#F_O = FMatrix(f_o)
#numpy.savetxt('F_O',F_O.array)

janus.Tcl_Eval(interp, "fs readADC ADCE chicken_4secs/chicken_eng_ch3_4s.wav")
janus.Tcl_Eval(interp, "fs readADC ADCD chicken_4secs/chicken_ger_ch3_4s.wav")

janus.Tcl_Eval(interp, "fs cut adcE ADCE 0s end")
janus.Tcl_Eval(interp, "fs cut adcD ADCD 0s end")

#janus.Tcl_Eval(interp, "fs writeADC adcE eng.wav -hm WAV")
#janus.Tcl_Eval(interp, "fs writeADC adcD ger.wav -hm WAV")

janus.Tcl_Eval(interp, "fs spectrum E adcE 16msec")
janus.Tcl_Eval(interp, "fs spectrum D adcD 16msec")

janus.Tcl_Eval(interp, "set melN 30")
janus.Tcl_Eval(interp, "set points [fs:E configure -coeffN]")
janus.Tcl_Eval(interp, "set rate   [expr 1000 * [fs:E configure -samplingRate]]")
janus.Tcl_Eval(interp, "[FBMatrix matrixMEL] mel -N $melN -p $points -rate $rate")
janus.Tcl_Eval(interp, "fs   filterbank       EMEL  E matrixMEL")
janus.Tcl_Eval(interp, "fs   log              ElMEL EMEL 1.0 1.0")

janus.Tcl_Eval(interp, "fs   filterbank       DMEL  D matrixMEL")
janus.Tcl_Eval(interp, "fs   log              DlMEL DMEL 1.0 1.0")

e = janus.itfGetObject ("fs:ElMEL.data", "FMatrix")
E = FMatrix(e)
d = janus.itfGetObject ("fs:DlMEL.data", "FMatrix")
D = FMatrix(d)

t  = E.array**2
ts = t.sum(axis=1)
ts = ts**0.5
#e_sum  = E.array.sum(axis=1)
E_norm = E.array / ts[:, numpy.newaxis]

t  = D.array**2
ts = t.sum(axis=1)
ts = ts**0.5
#d_sum  = D.array.sum(axis=1)
D_norm = D.array / ts[:, numpy.newaxis]

S=numpy.einsum('...i,...i',D_norm,E_norm)
S[numpy.isnan(S)] = 1

def smooth (x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:  
        w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

SS=smooth(S,100)
SSS=numpy.diff(numpy.where(SS<0.99,1,0))
numpy.savetxt('4s',SS)

ups  =numpy.where(SSS== 1)[0]
downs=numpy.where(SSS==-1)[0]
if len(ups) < len(downs):
    downs=numpy.delete(downs,0)

for s in zip(ups,downs):
    if s[1]-s[0] < 100:
        continue
    print .01*s[0],.01*s[1],.01*s[0]
    
    janus.Tcl_Eval(interp, "fs cut tmp adcE %fs %fs" % (.01*s[0],.01*s[1]))
    janus.Tcl_Eval(interp, "fs writeADC tmp tmp-%.2f.wav -hm WAV" % (.01*s[0]))
