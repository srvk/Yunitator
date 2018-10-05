# Yunitator
Diarization of child / adult speech using a Pytorch classifier

Run script is `runYunitator.sh` which takes a .wav file as input
and produces in the temp/working folder `/vagrant/Yunitemp/` a RTTM
file with matching base name. (this assumes installed in a Vagrant virtual machine)  
For example, if you have an audio file test.wav:
```
% vagrant ssh # ssh into running VM
vagrant@vagrant-ubuntu-trusty-64:~$ cd Yunitator 
vagrant@vagrant-ubuntu-trusty-64:~/Yunitator$ ./runYunitator.sh /vagrant/test.wav 
Extracting features for test.wav ...
(MSG) [2] in SMILExtract : openSMILE starting!
(MSG) [2] in SMILExtract : config file is: /vagrant/MED_2s_100ms_htk.conf
(MSG) [2] in cComponentManager : successfully registered 95 component types.
(MSG) [2] in cComponentManager : successfully finished createInstances
                                 (19 component instances were finalised, 1 data memories were finalised)
(MSG) [2] in cComponentManager : starting single thread processing loop
(MSG) [2] in cComponentManager : Processing finished! System ran for 1460 ticks.
DONE!

vagrant@vagrant-ubuntu-trusty-64:~/Yunitator$ cat /vagrant/Yunitemp/test.rttm
SPEAKER	test	1	0.0	0.2	<NA>	<NA>	SIL	<NA>	<NA>
SPEAKER	test	1	1.9	1.6	<NA>	<NA>	SIL	<NA>	<NA>
SPEAKER	test	1	0.2	1.7	<NA>	<NA>	CHI	<NA>	<NA>
```

## More Details
Yunitator produces these 3 classes (plus Silence):
```
CHI
FEM
MAL
SIL
```

## YuniSegs

This is a way of running Yunitator but you give it a RTTM file containing already-computed SAD segments. It runs Yunitator repeatedly iterating over all segments, and outputs the majority class for each segment in an RTTM file with extension `.yuniSegs.rttm`, with the same segments as the input RTTM (unless they were 0 duration)
It accepts 2 parameters, a WAV file, and an RTTM file, e.g:
```
~/Yunitator/runYuniSegs.sh /vagrant/test2.wav /vagrant/test2.rttm
```
It now accepts a 3rd parameter, the string "SkipSIL", which causes it to output a SIL segment for any segment marked as such in the input SAD RTTM. This could speed processing, and eliminate long segments marked as silence that might otherwise bog down Yunitator, and should really not be processed.

