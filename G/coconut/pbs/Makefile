# ---------------------------------------------------------------
#
#   Makefile for PBS DAG creation
#
#     ('DAG' as in Condor)
#     Florian Metze, fmetze@cs.cmu.edu
#
# ---------------------------------------------------------------
#
#   This Makefile can be used to run a workflow on the rocks cluster
#
#   Define targets and dependencies as in any makefile, see the examples below
#   The example targets show how different types of jobs can be defined 
#   (using the calls to 'pbs-*') and how job requirements can be specified
#
#   Usage: "make [ clean | new ] target+ [ submit ]"
#
#     * 'clean'   and 'new' clean up the current folder
#     * 'submit'  submits a job to the queue
#                 it is equivalent to doing './pbs.sh'
#     * 'jobwait' waits for a job that is already in the queue to finish
#     * 'kill'    kills all jobs that were submitted from this directory
#
#   The output of "make" typically is a file './pbs.sh' which contains
#     the commands that need to be submitted for this "build" to complete
#   'make submit' submits them, and records the Job IDs in './pbs/log'
#   'make kill' sources './pbs/log' and kills any IDs found therein
#     (use with care...)
#   The PBS submit script is in './pbs/target.sh'; once it has completed, 
#   'touch ./pbs/target' will be executed 
#     (and another call to 'make target' will simply return)
#   'make clean' removes any info except which targets have been built
#   'make new' wipes everything (including log-files)
#   Log files are always written into './Log', merging stdout and stderr,
#     usign the name of the target (and sometimes the Job ID) as the name
#
# ---------------------------------------------------------------

# ----------------------------
# ---- individual defines ----
# ----------------------------

# Write an MPI job to the queue
define pbs-mpi
@echo "" >> $(DAGFILE)
$(foreach j,1,$(foreach i,0,$(call pbs-core,$(1),$(2),$(3),mpirun --hostfile $$PBS_NODEFILE --output-filename $(LOGDIR)/$$PBS_JOBNAME $(4),1)))
endef

# Write a "single" or "array" job to the queue
define pbs-single
@echo "" >> $(DAGFILE)
$(foreach j,1,$(foreach i,0,$(call pbs-core,$(1),$(2),$(3),$(4),1)))
endef
#
define pbs-array
@echo "" >> $(DAGFILE)
$(foreach j,$(3),$(foreach i,$(call encode0,$(3)),$(call pbs-core,$(1),$(2),$(4),$(5),1)))
endef

# Write a client/ server "map/ reduce" job to the queue
define pbs-mapreduce
@echo "" >> $(DAGFILE)
$(foreach j,$(3),$(foreach i,$(call encode0,$(3)),$(call pbs-core,$(1),$(2)_c,$(4),$(5) -client 1,1)))
@echo "# (server node)" >> $(DAGFILE)
$(foreach j,1,$(foreach i,0,$(call pbs-core,$(2)_c,$(2),$(4),$(5) -server 1,1)))
endef

# Write multiple map-reduce iterations to the queue
define pbs-iterreduce
$(foreach i,$(3),$(call pbs-mapreduce,$(call patsubst,$(2)0,$(1),$(2)$(call decode,$(call decrement,$(call encode,$(i))))),$(2)$(i),$(4),$(5),$(6) -iter $(i)))
@echo "# (virtual node)" >> $(DAGFILE)
$(foreach j,1,$(foreach i,0,$(call pbs-core,$(2)$(word $(words $(3)),$(3)),$(2),walltime=00:00:05,date,1)))
endef

# Does the actual writing ...
#   Called with dependency,node-name,resource-list,command,number-of-jobs
define pbs-core
$(call pbs-script,$(2),$(4))
@echo '# rm -f $(PBSDIR)/$(2).sh'                 >> $(PBSDIR)/log
@echo "# JOB $(2)"                                >> $(DAGFILE)
@echo "export  C_NOJOBS=$(j) C_PROCESS=$(i)"      >> $(DAGFILE)
@echo "jid=\`qsub -S /bin/bash -v C_NOJOBS,C_PROCESS -j oe -o $(LOGDIR) -d . \\" >> $(DAGFILE)
@[ -z "$(QUEUE)" ] || echo "  -q $(QUEUE) \\"     >> $(DAGFILE)
@echo "  -N $(2) \\"                              >> $(DAGFILE)
@[ -z "$(strip $(filter-out $(VPATH)%,$(1)))" ] || echo "  -W depend=afterok"$(foreach i,$(strip $(filter-out $(VPATH)%,$(1))),'$${jid_$(i)}')" \\" | sed 's/ \$$/$$/g' >> $(DAGFILE)
@[ -z "$(3)" ]  || echo "  -l $(3) \\"            >> $(DAGFILE)
@echo "  $(PBSDIR)/$(2).sh\` "                    >> $(DAGFILE)
@echo 'jid_$(2)+=":$${jid}"'                      >> $(DAGFILE)
@echo 'echo qdel $${jid}        >> $(PBSDIR)/log' >> $(DAGFILE)
@echo 'echo \# LOG $${jid_$(2)} >> $(PBSDIR)/log' >> $(DAGFILE)
@echo "echo \# $(2) === $(4)    >> $(PBSDIR)/log" >> $(DAGFILE)

endef
# the blank line seems to be required

# ... and create a script to run
#     the 'touch' command shows that the command has completed successfully
#     (not reliable for array/ map-reduce jobs)
define pbs-script
@echo '#!/bin/sh'                   > $(PBSDIR)/$(1).sh
@echo '$(strip $(2))'              >> $(PBSDIR)/$(1).sh
@echo 'errorCode=$$?'              >> $(PBSDIR)/$(1).sh
@echo 'if [[ $$errorCode -ne 0 ]]' >> $(PBSDIR)/$(1).sh
@echo 'then'                       >> $(PBSDIR)/$(1).sh
@echo 'exit $$errorCode'           >> $(PBSDIR)/$(1).sh
@echo "fi"                         >> $(PBSDIR)/$(1).sh
@echo "touch $(PBSDIR)/$(1)"       >> $(PBSDIR)/$(1).sh
@chmod 755 $(PBSDIR)/$(1).sh
endef


# -----------------------------
# ---- integer arithmetics ----
# -----------------------------

input_int := $(shell seq 1 999)
input_i0  := 0 $(input_int)
decode     = $(words $1)
encode     = $(wordlist 1,$1,$(input_int))
encode0    = $(wordlist 1,$1,$(input_i0))
increment  = $1 x
decrement  = $(wordlist 2,$(words $1),$1)


# ----------------------------
# ---- target definitions ----
# ----------------------------

DAGFILE    = ./pbs.sh
PBSDIR     = ./pbs
LOGDIR     = ./Log
VPATH      = $(PBSDIR)
# NB: this use of VPATH may be technically wrong (and may cause the long runtime)

#QUEUE      = testing

# default target
all: run

# submit something to PBS
submit:
	@echo "*** Submitting to PBS ***"
	@cat $(DAGFILE)
	@echo "*** Result ***"
	@$(DAGFILE)

# clean, but keep record of targets that have been 'built' ok
#        i.e. './pbs/labels' exists means 'labels' has built ok 
clean:
	[ -d $(PBSDIR) ] || mkdir -p $(PBSDIR) $(LOGDIR)
	rm -f $(DAGFILE) $(PBSDIR)/*.sh $(PBSDIR)/log
	@echo "#!/bin/bash" > $(DAGFILE)
	@echo "# Auto-generated DAG $(DAGFILE)" >> $(DAGFILE)
	@chmod 755 $(DAGFILE)

# 'cleaner', also remove built targets (i.e. dependencies are correct)
new: clean
	rm -f $(PBSDIR)/* $(LOGDIR)/*

# kill the currently running jobs
kill:
	bash $(PBSDIR)/log


# ---------------------------------
# ----  interesting targets    ----
# ---------------------------------

run: clean lspf submit

lspf:
	$(call pbs-single,$^,$@,vmem=120Gb -l nodes=1:ppn=20 -l walltime=35:00:00,\
	PYTHONPATH=.:/home/iyu/sfep/asr_noisemes/base/lib/python2.7/site-packages \
	./extractLSPFeatures.py med13-14delta ./out \
	/home/iyu/sfep/asr_noisemes/noisemes/large_scale_feature_extraction/mergebaseANDmfcc_manual.ftr)

lspfpca:
	$(call pbs-single,$^,$@,vmem=120Gb -l nodes=1:ppn=20 -l walltime=35:00:00,\
	PYTHONPATH=.:/home/iyu/sfep/asr_noisemes/base/lib/python2.7/site-packages \
	./extractLSPFeatures.py med13-14delta ./out-pca \
	/home/iyu/sfep/LargeScaleAudioPoolingFeatures/config/FeatureSelection/keeper5500 \
	--meanvar /home/iyu/sfep/LargeScaleAudioPoolingFeatures/config/Normalization/mv.dat \
	--eigVal  /home/iyu/sfep/LargeScaleAudioPoolingFeatures/config/PCA/latent.csv \
	--eigVec  /home/iyu/sfep/LargeScaleAudioPoolingFeatures/config/PCA/pcomps.csv \
	--comps   100 300)

qnsamples: lda
	$(call pbs-mapreduce,$^,$@,10,file=250gb,\
	./janus qnsamples-v1.tcl)

split: cluster
	$(call pbs-single,$^,$@,"",\
	./janus split-v0.tcl)

cluster: trainPT
	$(call pbs-mpi,$^,$@,nodes=3:ppn=4 -l walltime=12:00:00,\
	./janus cluster-v0.tcl)

test: viterbi
	$(call pbs-array,$^,$@,20,"",\
	./janus test-v0.tcl )

viterbi: train_ofs
	$(call pbs-mpi,$^,$@,nodes=5:ppn=4 -l walltime=06:00:00,\
	./janus train-v0.tcl -labelPath "" -lda Weights/4i.GALL.bmat -cbsParam Weights/4i.ALL.cbs.gz -dssParam Weights/4i.ALL.dss.gz -begin 1 -end 1)

train: kmeans
	$(call pbs-iterreduce,$^,$@,1 2 3 4,20,"",\
	./janus train-v0.tcl)

kmeans: samples
	$(call pbs-mapreduce,$^,$@,50,"",\
	./janus kmeans-v0.tcl)

trainPT: makePT train_mas
	$(call pbs-mpi,$^,$@,nodes=5:ppn=4,\
	./janus train-v0.tcl -dssDesc desc/distribSet.PT.gz -dssParam "" -dstDesc desc/distribTree.PT.gz -ptreeDesc desc/ptreeSet.PT.gz)

makePT: lda train_mas
	$(call pbs-mpi,$^,$@,nodes=3:ppn=4,\
	./janus makePT-v0.tcl)

train_ofs: init_ofs
	$(call pbs-mpi,$^,$@,nodes=5:ppn=4 -l walltime=06:00:00,\
	./janus train-ofs-v0.tcl)

init_ofs: train_mas
	$(call pbs-single,$^,$@,"",\
	./janus init-ofs-v0.tcl)

train_mas: samples
	$(call pbs-mpi,$^,$@,nodes=6:ppn=4 -l walltime=06:00:00,\
	./janus train-mas-v0.tcl)

samples: lda
	$(call pbs-mpi,$^,$@,nodes=4:ppn=4 -l walltime=06:00:00,\
	./janus samples-v0.tcl)

lda:
	$(call pbs-mpi,$^,$@,nodes=4:ppn=4 -l walltime=06:00:00,\
	./janus lda-v0.tcl)

labels: 
	$(call pbs-mapreduce,$^,$@,50,"",\
	./janus labels-v4.tcl)

jobwait:
	$(call pbs-single,$^,$@,walltime=00:00:05 -W depend=afterok:817935,\
	date)
