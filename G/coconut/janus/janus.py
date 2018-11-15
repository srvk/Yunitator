from ctypes import *
import ctypes
import numpy

#cdll.LoadLibrary("/home/fmetze/janus/src/Linux.x86_64-gcc44-ltcl8.5-DLL-NX/libjanus.so")
cdll.LoadLibrary("/data/ASR1/tools/janus/lib/libjanus.so")
janus=CDLL("libjanus.so")

#from janus_ctypes import *
#import janus_ctypes
#janus=CDLL("libjanus.so")

# http://stackoverflow.com/questions/14576270/ctypes-function-returning-pointer-to-a-structure
class Tcl_Interp(ctypes.Structure):
      _fields_ = [
          ("result",    ctypes.c_char_p), 
          ("freeProc",  ctypes.c_char_p),
          ("errorLine", ctypes.c_int)
      ]


class DMatrix (Structure):
      """A Janus DMatrix."""
      _fields_ = [
            ("matPA", POINTER(POINTER(ctypes.c_double))),
            ("n",     ctypes.c_int),
            ("m",     ctypes.c_int),
            ("count", ctypes.c_double)
      ]

      def __init__ (self, *args, **kwargs):
            f8ptr = POINTER(c_double)

            # allow creation from numpy array, fmatrix, and dimensions
            if (len(args) == 2 and type(args[0]) is int and type(args[1]) is int):
                  self.m = args[0]
                  self.n = args[1]
                  self.count = 0.0
                  self.array = numpy.zeros((self.m, self.n), numpy.float64)

                  data  = (f8ptr*len(self.array))(*[row.ctypes.data_as(f8ptr) for row in self.array])
                  self.matPA = cast (data, POINTER(POINTER(c_double)))
                  
            elif (len(args) == 1 and type(args[0]) is numpy.ndarray and args[0].ndim is 2):
                  self.m = args[0].shape[0]
                  self.n = args[0].shape[1]
                  self.count = 0.0
                  self.array = args[0]
                  
                  data  = (f8ptr*len(self.array))(*[row.ctypes.data_as(f8ptr) for row in self.array])
                  self.matPA = cast (data, POINTER(POINTER(c_double)))
                  
            elif (len(args) == 1):
                  #and type(args[0]) is POINTER(DMatrix)):
                  self.m = args[0].contents.m
                  self.n = args[0].contents.n
                  self.count = 0.0

                  self.array = numpy.ctypeslib.as_array (args[0].contents.matPA[0], (self.m, self.n))
                  
            else:
                  raise Exception

      def mulot (self, a, b):
            janus.dmatrixMulot.argtypes = [POINTER(DMatrix), POINTER(DMatrix), POINTER(DMatrix)]
            janus.dmatrixMulot.restype  =  POINTER(DMatrix)
            return janus.dmatrixMulot (self, a, b)

      def det (self):
            janus.dmatrixDet.argtypes = [POINTER(DMatrix)]
            janus.dmatrixDet.restype  =  ctypes.c_double
            return janus.dmatrixDet (self)

janus.dmatrixCreate.argtypes = [c_int, c_int]
janus.dmatrixCreate.restype  = POINTER(DMatrix)


class FMatrix (Structure):
    """A Janus FMatrix."""
    _fields_ = [
        ("matPA", POINTER(POINTER(ctypes.c_float))),
        ("n",     ctypes.c_int),
        ("m",     ctypes.c_int),
        ("count", ctypes.c_double)
    ]

    def __init__ (self, *args, **kwargs):
        f4ptr = POINTER(c_float)

        # allow creation from numpy array, fmatrix, and dimensions
        if (len(args) == 2 and type(args[0]) is int and type(args[1]) is int):
            self.m = args[0]
            self.n = args[1]
            self.count = 0.0
            self.array = numpy.zeros((self.m, self.n), numpy.float32)

            data  = (f4ptr*len(self.array))(*[row.ctypes.data_as(f4ptr) for row in self.array])
            self.matPA = cast (data, POINTER(POINTER(c_float)))

        elif (len(args) == 1 and type(args[0]) is numpy.ndarray and args[0].ndim is 2):
            self.m = args[0].shape[0]
            self.n = args[0].shape[1]
            self.count = 0.0
            self.array = args[0]

            data  = (f4ptr*len(self.array))(*[row.ctypes.data_as(f4ptr) for row in self.array])
            self.matPA = cast (data, POINTER(POINTER(c_float)))

        elif (len(args) == 1):
            #and type(args[0]) is POINTER(FMatrix)):

            self.m = args[0].contents.m
            self.n = args[0].contents.n
            self.count = 0.0

            self.array = numpy.ctypeslib.as_array (args[0].contents.matPA[0], (self.m, self.n))

        else:
            raise Exception

    def mulot (self, a, b):
        janus.fmatrixMulot.argtypes = [ctypes.POINTER(FMatrix), ctypes.POINTER(FMatrix), ctypes.POINTER(FMatrix)]
        janus.fmatrixMulot.restype  =  ctypes.POINTER(FMatrix)
        self = janus.fmatrixMulot (self, a, b)

    def bsave (self, name):
        janus.fmatrixBSave.argtypes = [ctypes.POINTER(FMatrix), ctypes.c_char_p]
        janus.fmatrixBSave.restype  =  ctypes.POINTER(FMatrix)
        janus.fmatrixBSave (self, name)

FMatrix_t = POINTER(FMatrix)
janus.fmatrixCreate.argtypes = [c_int, c_int]
janus.fmatrixCreate.restype  = FMatrix_t


# -----------------------------------------------

interp = janus.Tcl_CreateInterp
interp.restype = ctypes.POINTER(Tcl_Interp)

interp = janus.Tcl_CreateInterp()

ok = janus.Tcl_Init (interp)

#  Itf_Init (interp)
ok = janus.DllItf_Init (interp)
tcl = POINTER(Tcl_Interp)
itf = tcl.in_dll(janus, "itf")
print itf, interp

ok = janus.Janus_Init (interp)
# It is ok that this returns '1'

janus.Tcl_Eval.argtypes = [ctypes.POINTER(Tcl_Interp), c_char_p]
janus.Tcl_GetStringResult.argtypes = [ctypes.POINTER(Tcl_Interp)]
janus.Tcl_GetStringResult.restype  = c_char_p
