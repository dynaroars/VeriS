EPOCHS=50
GEN_SPEC_DIR=generated_benchmark_new/
RESULTS_DIR=../exp/results_optimized/

GEN_SPEC_BASELINE_DIR=generated_benchmark_baseline/
RESULTS_DIR_BASELINE=../exp/results_baseline/

GEN_SPEC_UNOPTIMIZED_DIR=generated_benchmark_unoptimized/
RESULTS_DIR_UNOPTIMIZED=../exp/results_unoptimized/

NEURALSAT_DIR=../exp/neuralsat/src/
ABCROWN_DIR=../exp/abcrown/complete_verifier/

all: train spec verify

train: train_kws_m5 train_kws_m3 train_ecg_m5 train_ecg_m3 train_geometric

spec: spec_kws_m5 spec_kws_m3 spec_ecg_m5 spec_ecg_m3 spec_geometric

verify_neuralsat: verify_neuralsat_invariant verify_neuralsat_varying

verify_abcrown: verify_abcrown_invariant verify_abcrown_varying verify_abcrown_A_invariant verify_abcrown_A_varying

verify: verify_neuralsat verify_abcrown

# Training
train_kws_m5: 
	python train.py --task kws --model m5 --epochs ${EPOCHS} --n_channel 64
	python train.py --task kws --model m5 --epochs ${EPOCHS} --n_channel 32

train_kws_m3:
	python train.py --task kws --model m3 --epochs ${EPOCHS} --n_channel 64
	python train.py --task kws --model m3 --epochs ${EPOCHS} --n_channel 32

train_ecg_m5:
	python train.py --task ecg --model m5 --epochs ${EPOCHS} --n_channel 64
	python train.py --task ecg --model m5 --epochs ${EPOCHS} --n_channel 32

train_ecg_m3:
	python train.py --task ecg --model m3 --epochs ${EPOCHS} --n_channel 64
	python train.py --task ecg --model m3 --epochs ${EPOCHS} --n_channel 32

train_geometric:
	python train.py --task geometric --model f2 --epochs ${EPOCHS}
	python train.py --task geometric --model f4 --epochs ${EPOCHS}

# Generating specs
spec_kws_m5:
	python gen_spec.py --task kws --model m5 --n_channel 64 --sample_per_class 1 --spec_dir ${GEN_SPEC_DIR}
	python gen_spec.py --task kws --model m5 --n_channel 32 --sample_per_class 1 --spec_dir ${GEN_SPEC_DIR}

spec_kws_m3:
	python gen_spec.py --task kws --model m3 --n_channel 64 --sample_per_class 1 --spec_dir ${GEN_SPEC_DIR}
	python gen_spec.py --task kws --model m3 --n_channel 32 --sample_per_class 1 --spec_dir ${GEN_SPEC_DIR}

spec_ecg_m5:
	python gen_spec.py --task ecg --model m5 --n_channel 64 --sample_per_class 2 --spec_dir ${GEN_SPEC_DIR}
	python gen_spec.py --task ecg --model m5 --n_channel 32 --sample_per_class 2 --spec_dir ${GEN_SPEC_DIR}

spec_ecg_m3:
	python gen_spec.py --task ecg --model m3 --n_channel 64 --sample_per_class 2 --spec_dir ${GEN_SPEC_DIR}
	python gen_spec.py --task ecg --model m3 --n_channel 32 --sample_per_class 2 --spec_dir ${GEN_SPEC_DIR}

spec_geometric:
	python gen_spec.py --task geometric --model f2 --sample_per_class 4 --spec_dir ${GEN_SPEC_DIR}
	python gen_spec.py --task geometric --model f4 --sample_per_class 4 --spec_dir ${GEN_SPEC_DIR}

spec_baseline:
	python gen_spec_baseline.py --task kws --model m3 --n_channel 32 --sample_per_class 1 --spec_dir ${GEN_SPEC_BASELINE_DIR}
	python gen_spec_baseline.py --task kws --model m5 --n_channel 32 --sample_per_class 1 --spec_dir ${GEN_SPEC_BASELINE_DIR}

	python gen_spec_baseline.py --task kws --model m3 --n_channel 64 --sample_per_class 1 --spec_dir ${GEN_SPEC_BASELINE_DIR}
	python gen_spec_baseline.py --task kws --model m5 --n_channel 64 --sample_per_class 1 --spec_dir ${GEN_SPEC_BASELINE_DIR}

	python gen_spec_baseline.py --task ecg --model m3 --n_channel 32 --sample_per_class 2 --spec_dir ${GEN_SPEC_BASELINE_DIR}
	python gen_spec_baseline.py --task ecg --model m5 --n_channel 32 --sample_per_class 2 --spec_dir ${GEN_SPEC_BASELINE_DIR}

	python gen_spec_baseline.py --task ecg --model m3 --n_channel 64 --sample_per_class 2 --spec_dir ${GEN_SPEC_BASELINE_DIR}
	python gen_spec_baseline.py --task ecg --model m5 --n_channel 64 --sample_per_class 2 --spec_dir ${GEN_SPEC_BASELINE_DIR}

verify_baseline:
	python verify.py --benchmark_type time_invariant --verifier neuralsat --verifier_dir ${NEURALSAT_DIR} --output_dir ${RESULTS_DIR_BASELINE} --timeout 30 --benchmark_dir ${GEN_SPEC_BASELINE_DIR}

verify_neuralsat_invariant:
	python verify.py --benchmark_type time_invariant --verifier neuralsat --verifier_dir ${NEURALSAT_DIR} --output_dir ${RESULTS_DIR} --timeout 30 --benchmark_dir ${GEN_SPEC_DIR}

verify_neuralsat_varying:
	python verify.py --benchmark_type time_varying   --verifier neuralsat --verifier_dir ${NEURALSAT_DIR} --output_dir ${RESULTS_DIR} --timeout 60 --benchmark_dir ${GEN_SPEC_DIR}

verify_abcrown_invariant:
	python verify.py --benchmark_type time_invariant --verifier abcrown --verifier_dir ${ABCROWN_DIR} --output_dir ${RESULTS_DIR} --timeout 30 --benchmark_dir ${GEN_SPEC_DIR}

verify_abcrown_varying:
	python verify.py --benchmark_type time_varying   --verifier abcrown --verifier_dir ${ABCROWN_DIR} --output_dir ${RESULTS_DIR} --timeout 60 --benchmark_dir ${GEN_SPEC_DIR}

verify_abcrown_A_invariant:
	python verify.py --benchmark_type time_invariant --verifier abcrown_A --verifier_dir ${ABCROWN_DIR} --output_dir ${RESULTS_DIR} --timeout 30 --benchmark_dir ${GEN_SPEC_DIR}

verify_abcrown_A_varying:
	python verify.py --benchmark_type time_varying   --verifier abcrown_A --verifier_dir ${ABCROWN_DIR} --output_dir ${RESULTS_DIR} --timeout 60 --benchmark_dir ${GEN_SPEC_DIR}

create_csv:
	python plot/export_results.py --output_dir ${RESULTS_DIR} --benchmark_dir ${GEN_SPEC_DIR}
	# python plot/export_results.py --output_dir ${RESULTS_DIR_BASELINE} --benchmark_dir ${GEN_SPEC_BASELINE_DIR} --postfix baseline
	# python plot/export_results.py --output_dir ${RESULTS_DIR_UNOPTIMIZED} --benchmark_dir ${GEN_SPEC_UNOPTIMIZED_DIR} --postfix unoptimized

export_csv_baseline:
	python plot/export_results.py --output_dir ${RESULTS_DIR_BASELINE} --benchmark_dir ${GEN_SPEC_BASELINE_DIR} --postfix baseline
	python plot/export_results.py --output_dir ${RESULTS_DIR_UNOPTIMIZED} --benchmark_dir ${GEN_SPEC_UNOPTIMIZED_DIR} --postfix unoptimized

create_table:
	python plot/create_tables.py --csv plot/neuralsat_results.csv
	# python plot/create_tables.py --csv plot/neuralsat_results_baseline.csv

create_plot:
	python3 -m plot.time_varying_example
	python3 -m plot.invariant_vs_varying
	python3 -m plot.compatibility


# Cleaning
clean:
	rm -rf checkpoints/
	rm -rf generated_benchmark/
	rm -rf log_*.txt
