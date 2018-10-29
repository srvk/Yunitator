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

This is a way of running Yunitator but you give it a RTTM file containing already-computed SAD segments. It runs Yunitator repeatedly iterating over all segments, and outputs the majority class for each segment in an RTTM file, name based on the input WAV filename, with extension `.yuniSegs.rttm`. IT has the same segments as the input RTTM (unless they were 0 duration).
YuniSegs accepts 2 parameters, a WAV file, and an RTTM file, e.g:
```
~/Yunitator/runYuniSegs.sh /vagrant/test2.wav /vagrant/test2.rttm
```
It now accepts a 3rd parameter, the string "SkipSIL", which causes it to output a SIL segment for any segment marked as such in the input SAD RTTM. This could speed processing, and eliminate long segments marked as silence that might otherwise bog down Yunitator, and should really not be processed.

## Even More Details
General intro

This system will classify slices of the audio recording into one of 17 noiseme classes:

    background
    speech
    speech non English
    mumble
    singing alone
    music + singing
    music alone
    human sounds
    cheer
    crowd sounds
    animal sounds
    engine
    noise_ongoing
    noise_pulse
    noise_tone
    noise_nature
    white_noise
    radio

To learn more, read the source file Wang, Y., Neves, L., & Metze, F. (2016, March). Audio-based multimedia event detection using deep recurrent neural networks. In Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on (pp. 2742-2746). IEEE. pdf
Instructions for direct use

You can analyze just one file as follows. Imagine that <$MYFILE> is the name of the file you want to analyze, which you've put inside the data/ folder in the current working directory.
```
$ vagrant ssh -c "OpenSAT/runOpenSAT.sh data/<$MYFILE>"
```
You can also analyze a group of files as follows:
```
$ vagrant ssh -c "OpenSAT/runDiarNoisemes.sh data/"
```
This will analyze all .wav's inside the "data" folder.

Created annotations will be stored inside the same "data" folder.
Some more technical details

For more fine grained control, you can log into the VM and from a command line, and play around from inside the "Diarization with noisemes" directory, called "OpenSAT":
```
$ vagrant ssh
$ cd OpenSAT
```
The main script is runOpenSAT.sh and takes one argument: an audio file in .wav format. Upon successful completion, output will be in the folder (relative to ~/OpenSAT) SSSF/data/hyp/<input audiofile basename>/confidence.pkl.gz

The system will grind first creating features for all the .wav files it found, then will place those features in a subfolder feature. Then it will load a model, and process all the features generated, producing output in a subfolder hyp/ two files per input: <inputfile>.confidence.mat and <inputfile>.confidence.pkl.gz - a confidence matrix in Matlab v5 mat-file format, and a Python compressed data 'pickle' file. Now, as well, in the hyp/ folder, <inputfile>.rttm with labels found from a config file noisemeclasses.txt

-More details on output format-

The 18 classes are as follows:
```
0	background	
1	speech	
2	speech_ne	
3	mumble	
4	singing	
5	music_sing
6	music
7	human	
8	cheer	
9	crowd	
10	animal
11	engine
12	noise_ongoing
13	noise_pulse
14	noise_tone
15	noise_nature
16	white_noise
17	radio
```
The frame length is 0.1s. The system also uses a 2-second window, so the i-th frame starts at (0.1 * i - 2) seconds and finishes at (0.1 * i) seconds. That's why 60 seconds become 620 frames. 'speech_ne' means non-English speech

-Sample RTTM output snippet-
```
SPEAKER family  1       4.2     0.4     noise_ongoing <NA>    <NA>    0.37730383873
SPEAKER family  1       4.6     1.2     background    <NA>    <NA>    0.327808111906
SPEAKER family  1       5.8     1.1     speech        <NA>    <NA>    0.430758684874
SPEAKER family  1       6.9     1.2     background    <NA>    <NA>    0.401730179787
SPEAKER family  1       8.1     0.7     speech        <NA>    <NA>    0.407463937998
SPEAKER family  1       8.8     1.1     background    <NA>    <NA>    0.37258502841
SPEAKER family  1       9.9     1.7     noise_ongoing <NA>    <NA>    0.315185159445 
```
The script `runClasses.sh` works like `runDiarNoisemes.sh`, but produces the more detailed results as seen above.

