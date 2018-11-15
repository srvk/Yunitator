# __init__.py for Coconut (pythonesque audio processing)

import os
import atexit
import socket
import datetime
import resource
import __main__ as main


# welcome/ goodbye message
if hasattr(main, '__file__'):
    print '[[[ Starting', __name__, 'on', socket.gethostname().partition('.')[0], os.getpid(), 'at', datetime.datetime.now().replace(microsecond=0).isoformat(' '), ']]]'

    def atexit_coconut (dt):
        u = resource.getrusage(resource.RUSAGE_SELF).ru_utime + \
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        s = resource.getrusage(resource.RUSAGE_SELF).ru_stime + \
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        t = datetime.datetime.now()-dt
        if t.microseconds < 500000:
            T = t-datetime.timedelta(microseconds=t.microseconds)
        else:
            T = t+datetime.timedelta(microseconds=1000000-t.microseconds)
        print '[[[ Elapsed time is', T, 'with user= {:.1f}%'.format(u/t.total_seconds()*100), 'and system= {:.1f}%'.format(s/t.total_seconds()*100), ']]]'

    atexit.register (atexit_coconut, dt=datetime.datetime.now())


# make sure any 'parmap' directory that we link to gets recognized as a package, 
# even though only the original package is checked in,
# which in 1.2.1 does not contain __init__
import os.path
fname=os.path.join(os.path.dirname(__file__), 'parmap/__init__.py')
if not os.path.exists(fname):
    open(fname, 'a').close()


# import 'coconut' packages
from coconut.parmap    import parmap
from coconut.opensmile import opensmile
from coconut.utils     import utils
