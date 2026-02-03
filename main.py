import platform, os

def main_thread_import_torch():
    import psutil
    n_phy_cores= psutil.cpu_count(logical=False)
    os.environ['NUM_THREADS'] = str(n_phy_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_phy_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_phy_cores)
    os.environ['OMP_NUM_THREADS'] = str(n_phy_cores)
    import torch
    torch.set_num_threads(n_phy_cores)
    torch.manual_seed(cfg.seed)
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available")


if __name__ == '__main__':
    from conf_system import init_conf_system, load_global_conf
    load_global_conf()  # avoid import torch from ALGORITHM
    from config import GlobalConfig as cfg
    run_RL = (cfg.runner == 'rl_runner')

    # Set numpy seed
    import numpy
    numpy.random.seed(cfg.seed)

    if run_RL:
        # Init remote process, create environments also
        from MISSION.env_router import make_parallel_envs, load_ScenarioConfig
        envs = make_parallel_envs()

        # pytorch can be init AFTER the creation of remote process, set pytorch seed
        main_thread_import_torch()
        
        init_conf_system(prepare_logdir=True, print_summary=True)

        cfg.ScenarioConfig = load_ScenarioConfig(cfg.env_name)

        # Prepare everything else
        from rl_runner import Runner
        print("CREATE RUNNER")
        runner = Runner(envs=envs)
        print("RUN")
        runner.run()
    else: # run il_runner
        main_thread_import_torch()
        init_conf_system(prepare_logdir=True, print_summary=True)
        from il_runner import Runner
        runner = Runner()
        runner.run()

elif platform.system()!="Linux":
    # Linux uses fork for multi-processing, but Windows does not, reload config for Windows
    from conf_system import init_conf_system
    cfg = init_conf_system()
