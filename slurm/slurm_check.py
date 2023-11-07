"""Check progress of slurm batch experiments"""
import os
import json
import argparse


# constants
SLURM_DN = "slurm_logs"

TRANSFER_TO_PREFIX = "Moving input data to the compute node's scratch space: "
PREPROCESS_PROMPT = "Pre-processing data in scratch space"
RUNNING_PREFIX = 'Running provided command: '
FAILED_PROMPT = 'Command failed!'
TRANSFER_FROM_PROMPT = "Moving output data back to DFS"
#WANDB_PROMPT = "Log uploaded to wandb"
DELETE_PROMPT="Deleting output files in scratch space"
FINISHED_PROMPT = 'Job finished successfully!'
TIMEOUT_PROMPT = 'CANCELLED'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Experiment Checker"
    )
    parser.add_argument(
        "-j", "--job", 
        required=True,
        metavar="INT",
        help="Slurm Job ID",
        type=int,
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="PATH",
        help="Absolute path to path config file",
        type=str,
    )
    args = parser.parse_args()
    if args.job < 1:
        raise ValueError('Job ID is not positive!')
    if not os.path.exists(args.config):
        raise FileNotFoundError('Config file not found!')
    cfg = {}
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    slurm_log_path = os.path.join(cfg['EDI']['HOME'], 
                                  cfg['EDI']['USER'], 
                                  SLURM_DN)
    project_path = os.path.join(cfg['EDI']['HOME'], 
                                cfg['EDI']['USER'], 
                                cfg['EDI']['PROJECT'])
    slurm_path = os.path.join(project_path, cfg['SLURM_DN'])
    exp_fp = os.path.join(slurm_path, cfg['EXP']['TSV']['DEFAULT_FN'])
    exp_fail_fp = os.path.join(slurm_path, cfg['EXP']['TSV']['FAILED_FN'])
    exp_timeout_fp = os.path.join(slurm_path, cfg['EXP']['TSV']['TIMEOUT_FN'])

    # different job status
    queuing_ids = []
    transferTo_ids = []
    preprocess_ids = []
    running_ids = []
    transferFrom_ids = []
    #wandb_ids = []
    delete_ids = []
    finished_ids = []
    failed_ids = []
    timeout_ids = []

    # get experiments
    lines = []
    with open(exp_fp, 'r') as f:
        lines = f.readlines()

    # look at status of each experiment
    for id, line in enumerate(lines[1:], start=1):
        slurm_log_fp = os.path.join(slurm_log_path, f'slurm-{args.job}_{id}.out') 
        if not os.path.exists(slurm_log_fp):
            queuing_ids.append(id)
            continue

        process_flags = [False for _ in range(8)]
        with open(slurm_log_fp, 'r') as f:
            log_line = f.readline()
            while log_line:
                if TRANSFER_TO_PREFIX in log_line:
                    process_flags[0] = True
                elif PREPROCESS_PROMPT  in log_line:
                    process_flags[1] = True
                elif (RUNNING_PREFIX + line) in log_line:
                    process_flags[2] = True
                elif TRANSFER_FROM_PROMPT  in log_line:
                    process_flags[3] = True
                #elif WANDB_PROMPT  in log_line:
                #    process_flags[4] = True
                elif DELETE_PROMPT  in log_line:
                    process_flags[4] = True
                elif FINISHED_PROMPT in log_line:
                    process_flags[5] = True
                    break
                elif FAILED_PROMPT in log_line:
                    process_flags[6] = True
                    break
                elif TIMEOUT_PROMPT in log_line:
                    process_flags[7] = True
                    break
                log_line = f.readline()

        # sort flags
        if process_flags[7]:
            timeout_ids.append(id)
        elif process_flags[6]:
            failed_ids.append(id)
        elif process_flags[5]:
            finished_ids.append(id)
        elif process_flags[4]:
            delete_ids.append(id)
        #elif process_flags[4]:
        #    wandb_ids.append(id)
        elif process_flags[3]:
            transferFrom_ids.append(id)
        elif process_flags[2]:
            running_ids.append(id)
        elif process_flags[1]:
            preprocess_ids.append(id)
        elif process_flags[0]:
            transferTo_ids.append(id)

    print('QUEUING/LOG_IN_OTHER_DIR ----------')
    print(queuing_ids)
    print()

    print('TRANSFER TO GPU NODE ----------')
    print(transferTo_ids)
    print()

    print('PREPROCESSING ON GPU NODE ----------')
    print(preprocess_ids)
    print()

    print('RUNNING ----------')
    print(running_ids)
    print()

    print('TRANSFER FROM GPU_NODE ----------')
    print(transferFrom_ids)
    print()

    #print('WANDB_LOGGING ----------')
    #print(wandb_ids)
    #print()

    print('DELETING OUTPUT FROM GPU_NODE ----------')
    print(transferFrom_ids)
    print()

    print('FINISHED ----------')
    print(finished_ids)
    print()

    print('FAILED ----------')
    print(failed_ids)
    print()

    print('CANCELLED ----------')
    print(timeout_ids)
    print()

    any_fails = len(failed_ids) > 0
    any_timeouts = len(timeout_ids) > 0

    if any_fails or any_timeouts:
        # saving details
        lines = []
        with open(exp_fp, 'r') as f:
            lines = f.readlines()

        if any_fails:
            with open(exp_fail_fp, 'w') as f:
                # header
                f.write(lines[0])
                for id in failed_ids:
                    f.write(lines[id])
            print(f'Saved failed experiment details in: {exp_fail_fp}')

        if any_timeouts:
            with open(exp_fail_fp, 'w') as f:
                # header
                f.write(lines[0])
                for id in timeout_ids:
                    f.write(lines[id])
            print(f'Saved cancelled experiment details in: {exp_fail_fp}')
