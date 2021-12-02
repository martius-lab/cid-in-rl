SHELL:=$(shell which bash)

.PHONY: all training scoring

PROJ_ROOT := $(shell realpath ../..)
DATA_DIR := ${PROJ_ROOT}/data
ONEDSLIDE_DATA_DIR := ${DATA_DIR}/onedslide
PNP_DATA_DIR := ${DATA_DIR}/fetchpickandplace
EXP_DIR := ${DATA_DIR}/experiments/1-influence
MODEL_DIR := ${EXP_DIR}/models
EVAL_SETTINGS_PATH := ${PROJ_ROOT}/experiments/1-influence

all: training scoring

seeds := 0 1 2 3 4

########### Training ###########

TRAIN_ONEDSLIDE_RANDOM_1K := OneDSlide-Random-1k
TRAIN_ONEDSLIDE_DDPG_1K := OneDSlide-DDPG-1k
TRAIN_PNP_RANDOM_5K := FetchPickAndPlace-Random-5k
TRAIN_PNP_DDPG_5K := FetchPickAndPlace-DDPG-5k
TRAIN_PNP_DDPG_NOISY_5K := FetchPickAndPlace-DDPG-Noisy-5k

TRAIN_ONEDSLIDE_RANDOM_1K_DIR := ${MODEL_DIR}/${TRAIN_ONEDSLIDE_RANDOM_1K}
TRAIN_ONEDSLIDE_DDPG_1K_DIR := ${MODEL_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}
TRAIN_PNP_RANDOM_5K_DIR := ${MODEL_DIR}/${TRAIN_PNP_RANDOM_5K}
TRAIN_PNP_DDPG_5K_DIR := ${MODEL_DIR}/${TRAIN_PNP_DDPG_5K}
TRAIN_PNP_DDPG_NOISY_5K_DIR := ${MODEL_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}

noises = all_0.05 all_0.10 all_0.20
TRAIN_PNP_DDPG_NOISY_5K_DIRS := $(addprefix ${TRAIN_PNP_DDPG_NOISY_5K_DIR}/,${noises})

train_dirs := ${TRAIN_ONEDSLIDE_RANDOM_1K_DIR} ${TRAIN_PNP_DDPG_5K_DIR} ${TRAIN_PNP_DDPG_NOISY_5K_DIRS}

training: $(train_dirs)

$(train_dirs): % : $(addprefix %/CMI/,$(seeds)) $(addprefix %/Transformer/,$(seeds))

########### Training: CMI ###########

${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/CMI/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${ONEDSLIDE_DATA_DIR}/train_memory_1k_random_agent.npy \
		--val-memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_random_agent.npy \
		${EVAL_SETTINGS_PATH}/onedslide_model_gaussian.gin

${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/CMI/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${ONEDSLIDE_DATA_DIR}/train_memory_1k_her_agent.npy \
		--val-memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_her_agent.npy \
		${EVAL_SETTINGS_PATH}/onedslide_model_gaussian.gin

${TRAIN_PNP_RANDOM_5K_DIR}/CMI/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${PNP_DATA_DIR}/memory_5k_random_agent_v2.npy \
		--val-memory-path ${PNP_DATA_DIR}/val_memory_2k_random_agent_v2.npy \
		${EVAL_SETTINGS_PATH}/pickandplace_model_gaussian.gin

${TRAIN_PNP_DDPG_5K_DIR}/CMI/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${PNP_DATA_DIR}/memory_5k_her_agent_v2.npy \
		--val-memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v2.npy \
		${EVAL_SETTINGS_PATH}/pickandplace_model_gaussian.gin

noise_train_dirs_cmi := $(foreach NOISE,$(noises), $(foreach SEED,$(seeds),${TRAIN_PNP_DDPG_NOISY_5K_DIR}/${NOISE}/CMI/${SEED}))
${noise_train_dirs_cmi}:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${PNP_DATA_DIR}/memory_5k_her_agent_v3.npy \
		--val-memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v3.npy \
		${EVAL_SETTINGS_PATH}/pickandplace_model_gaussian_noisy.gin \
		"NOISE_LEVEL=$(shell echo $@ | sed -n 's/.*\(0\.[0-9][0-9]\).*/\1/p')" \
		"fetch_with_object_state_noise_fn.only_object_noise=$(shell if [ `echo $@ | grep -c obj` -eq 1 ]; then echo True; else echo False; fi)"

########### Training: Transformer ###########

${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/Transformer/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${ONEDSLIDE_DATA_DIR}/train_memory_1k_random_agent.npy \
		--val-memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_random_agent.npy \
		${EVAL_SETTINGS_PATH}/onedslide_transformer.gin

${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/Transformer/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${ONEDSLIDE_DATA_DIR}/train_memory_1k_her_agent.npy \
		--val-memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_her_agent.npy \
		${EVAL_SETTINGS_PATH}/onedslide_transformer.gin

${TRAIN_PNP_DDPG_5K_DIR}/Transformer/%:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${PNP_DATA_DIR}/memory_5k_her_agent_v2.npy \
		--val-memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v2.npy \
		${EVAL_SETTINGS_PATH}/pickandplace_transformer.gin

${TRAIN_PNP_RANDOM_5K_DIR}/Transformer/%:

noise_train_dirs_transformer := $(foreach NOISE,$(noises), $(foreach SEED,$(seeds),${TRAIN_PNP_DDPG_NOISY_5K_DIR}/${NOISE}/Transformer/${SEED}))
${noise_train_dirs_transformer}:
	cd ${PROJ_ROOT}; \
	python -m cid.influence_estimation.train_model \
		--no-logging-subdir --log-dir $@ --seed $(@F) \
		--memory-path ${PNP_DATA_DIR}/memory_5k_her_agent_v3.npy \
		--val-memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v3.npy \
		${EVAL_SETTINGS_PATH}/pickandplace_transformer_noisy.gin \
		"NOISE_LEVEL=$(shell echo $@ | sed -n 's/.*\(0\.[0-9][0-9]\).*/\1/p')" \
		"fetch_with_object_state_noise_fn.only_object_noise=$(shell if [ `echo $@ | grep -c obj` -eq 1 ]; then echo True; else echo False; fi)"

########### Scoring ###########

SCORING_SCRIPT := ${PROJ_ROOT}/experiments/1-influence/pickandplace_cmi.py
SCORES_DIR := ${EXP_DIR}/scores

scorers := var_prod_approx entropy contact

SCORES_EVAL_ONEDSLIDE_RANDOM_1K_DIR := ${SCORES_DIR}/eval-${TRAIN_ONEDSLIDE_RANDOM_1K}
SCORES_EVAL_ONEDSLIDE_DDPG_1K_DIR := ${SCORES_DIR}/eval-${TRAIN_ONEDSLIDE_DDPG_1K}
SCORES_TEST_ONEDSLIDE_4K_DIR := ${SCORES_DIR}/test-OneDSlide-4k
SCORES_EVAL_PNP_RANDOM_5K_DIR := ${SCORES_DIR}/eval-${TRAIN_PNP_RANDOM_5K}
SCORES_EVAL_PNP_DDPG_5K_DIR := ${SCORES_DIR}/eval-${TRAIN_PNP_DDPG_5K}
SCORES_TEST_PNP_30K_DIR := ${SCORES_DIR}/test-FetchPickAndPlace-30k
SCORES_TEST_PNP_RANDOM_5K_DIR := ${SCORES_DIR}/test-FetchPickAndPlace-Random-5k

methods_and_seeds := $(addprefix CMI/,$(seeds)) $(addprefix Transformer/,$(seeds))
folders := $(addprefix ${SCORES_EVAL_ONEDSLIDE_DDPG_1K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/,$(methods_and_seeds))
folders += $(addprefix ${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/,$(methods_and_seeds))
folders += $(addprefix ${SCORES_EVAL_PNP_DDPG_5K_DIR}/${TRAIN_PNP_DDPG_5K}/,$(methods_and_seeds))
folders += $(addprefix ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_5K}/,$(methods_and_seeds))
folders += $(addprefix ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_5K}/,$(methods_and_seeds))
folders += $(foreach NOISE,$(noises), $(foreach METHOD_AND_SEED,$(methods_and_seeds),${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/${METHOD_AND_SEED}))
folders += $(foreach NOISE,$(noises), $(foreach METHOD_AND_SEED,$(methods_and_seeds),${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/${METHOD_AND_SEED}))

score_dirs := ${SCORES_EVAL_ONEDSLIDE_DDPG_1K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K} \
	${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K} \
	${SCORES_EVAL_PNP_DDPG_5K_DIR}/${TRAIN_PNP_DDPG_5K} \
	${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_5K} \
	${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_5K} \
	$(addprefix ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/,${noises}) \
	$(addprefix ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/,${noises})

scoring: $(score_dirs)

$(score_dirs): % : $(addprefix %/CMI/,$(seeds)) $(addprefix %/Transformer/,$(seeds))

########### Scoring: CMI ###########

target_pattern = ${SCORES_EVAL_ONEDSLIDE_RANDOM_1K_DIR}/${TRAIN_ONEDSLIDE_RANDOM_1K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_random_agent.npy --variants $(scorers)

target_pattern = ${SCORES_EVAL_ONEDSLIDE_DDPG_1K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_her_agent.npy --variants $(scorers)

target_pattern = ${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_RANDOM_1K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/test_memory_4k_random+her_agent.npy --variants $(scorers)

target_pattern = ${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/test_memory_4k_random+her_agent.npy --variants $(scorers)

target_pattern = ${SCORES_EVAL_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_RANDOM_5K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_RANDOM_5K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/val_memory_2k_random_agent_v2.npy --variants $(scorers)

target_pattern = ${SCORES_EVAL_PNP_DDPG_5K_DIR}/${TRAIN_PNP_DDPG_5K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v2.npy --variants $(scorers)

target_pattern = ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_RANDOM_5K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_RANDOM_5K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_7.5kof30k_her_agent_v2.npy --variants $(scorers)

target_pattern = ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_5K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_7.5kof30k_her_agent_v2.npy --variants $(scorers)

target_pattern = ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_5K}/CMI/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/CMI/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_5k_random_agent_v2.npy --variants $(scorers)

target_pattern = $(foreach NOISE,$(noises),${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/CMI/%)
$(filter $(target_pattern),$(folders)): ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/% : ${TRAIN_PNP_DDPG_NOISY_5K_DIR}/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_7.5kof30k_her_agent_v2.npy --extra-args ForwardGoalDataset.use_state_noise=False --variants $(scorers)

target_pattern = $(foreach NOISE,$(noises),${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/CMI/%)
$(filter $(target_pattern),$(folders)): ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/% : ${TRAIN_PNP_DDPG_NOISY_5K_DIR}/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_5k_random_agent_v2.npy --extra-args ForwardGoalDataset.use_state_noise=False --variants $(scorers)

########### Scoring: Transformer ###########

target_pattern = ${SCORES_EVAL_ONEDSLIDE_RANDOM_1K_DIR}/${TRAIN_ONEDSLIDE_RANDOM_1K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_random_agent.npy --variants attention_onedslide

target_pattern = ${SCORES_EVAL_ONEDSLIDE_DDPG_1K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/val_memory_1k_her_agent.npy --variants attention_onedslide

target_pattern = ${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_RANDOM_1K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_RANDOM_1K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/test_memory_4k_random+her_agent.npy --variants attention_onedslide

target_pattern = ${SCORES_TEST_ONEDSLIDE_4K_DIR}/${TRAIN_ONEDSLIDE_DDPG_1K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_ONEDSLIDE_DDPG_1K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${ONEDSLIDE_DATA_DIR}/test_memory_4k_random+her_agent.npy --variants attention_onedslide

target_pattern = ${SCORES_EVAL_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_RANDOM_5K}/Transformer/%
$(filter $(target_pattern),$(folders)):

target_pattern = ${SCORES_EVAL_PNP_DDPG_5K_DIR}/${TRAIN_PNP_DDPG_5K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/val_memory_2kof5k_her_agent_v2.npy --variants attention_fetchpnp

target_pattern = ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_RANDOM_5K}/Transformer/%
$(filter $(target_pattern),$(folders)):

target_pattern = ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_5K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_7.5kof30k_her_agent_v2.npy --variants attention_fetchpnp

target_pattern = ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_5K}/Transformer/%
$(filter $(target_pattern),$(folders)): $(target_pattern) : ${TRAIN_PNP_DDPG_5K_DIR}/Transformer/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_5k_random_agent_v2.npy --variants attention_fetchpnp

target_pattern = $(foreach NOISE,$(noises),${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/Transformer/%)
$(filter $(target_pattern),$(folders)): ${SCORES_TEST_PNP_30K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/% : ${TRAIN_PNP_DDPG_NOISY_5K_DIR}/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_7.5kof30k_her_agent_v2.npy --extra-args FactorizedForwardDataset.use_state_noise=False --variants attention_fetchpnp

target_pattern = $(foreach NOISE,$(noises),${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/${NOISE}/Transformer/%)
$(filter $(target_pattern),$(folders)): ${SCORES_TEST_PNP_RANDOM_5K_DIR}/${TRAIN_PNP_DDPG_NOISY_5K}/% : ${TRAIN_PNP_DDPG_NOISY_5K_DIR}/%
	python ${SCORING_SCRIPT} --output-path $@ --model-path $< --settings-path $</eval_settings.gin \
		--memory-path ${PNP_DATA_DIR}/test_memory_5k_random_agent_v2.npy --extra-args FactorizedForwardDataset.use_state_noise=False --variants attention_fetchpnp