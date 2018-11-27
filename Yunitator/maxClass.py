import sys, os, os.path
import operator

INPUT_FILE = sys.argv[1]        # RTTM containing segments & labels

classlist=[]
durations={}

# grow List all class names
f = open(INPUT_FILE, 'r')
for line in f:
    a = line.split()
    theclass=a[7]  # 8th column of RTTM is class label
    duration=float(a[4])
    if not theclass in classlist:
        classlist.append(theclass)
        durations[theclass]=float(0)
    durations[theclass]+=float(duration)

#print classlist

#print durations
sorted_durations = sorted(durations.items(), key=operator.itemgetter(1))
#print sorted_durations
if len(sorted_durations) == 0:
    print "SIL"
else:
    print sorted_durations[len(sorted_durations)-1][0]
