import time, numpy as np, traceback, threading
import multiprocessing as mp
from typing import List, Dict, Any
from uhtk.imitation.utils import safe_load_traj_pool, safe_dump_traj_pool
from uhtk.imitation.trajPool_sampler import get_container_from_traj_pool
from uhtk.print_pack import *
from uhtk.siri.utils.print_nan import check_nan, print_nan
from data_converter import convert_campas_action_to_discrete
from config import GlobalConfig as cfg

def extract_name(traj_pool: List[Any], req_dict_name: List[str]):
    for traj in traj_pool:
        for name in req_dict_name:
            check_nan(getattr(traj, name))
            # print(len(getattr(traj, name)))
    container = get_container_from_traj_pool(traj_pool, req_dict=req_dict_name, req_dict_rename=req_dict_name)
    return container

def try_extract_name(traj_pool: List[Any], req_dict_name: List[str]):
    try:
        container = extract_name(traj_pool, req_dict_name)
    except Exception as e:
        container = None
        print红(f"[data_loader] Error extracting names: {e}")
        traceback.print_exc()
    return container

def get_data(traj_pool):
    req_keys_all = []
    req_keys_all.append(['obs', 'campas_action_and_shoot'])  # BVR 3D recorded data
    # req_keys_all.append(['obs', 'action'])
    # req_keys_all.append(['obs', 'action', 'actionLogProb', 'return', 'reward', 'value'])
    # req_keys_all.append(['key', 'mouse', 'FRAME_raw'])
    

    container = None
    for req_keys in req_keys_all:
        container = try_extract_name(traj_pool, req_keys)
        if container is not None:
            break
    # print_dict(container)

    print("obs shape: ", container['obs'].shape)

    ### convert if needed

    # Check if we have BVR 3D compass action data that needs conversion
    if 'campas_action_and_shoot' in container and 'action' not in container:
        # print("[data_loader] Converting compass actions to discrete indices...")
        converted = []
        for seq in container['campas_action_and_shoot']:
            converted.append(convert_campas_action_to_discrete(seq))
        action_indices = np.array(converted)
        container['action'] = action_indices
        # print(f"[data_loader] Converted actions sample: {action_indices[-1, :5]}")

    data = {
        'obs': container['obs'],
        'action': container['action'],
        'traj_mask': container['traj_mask'],
    }
    return data

# def data_loader_process(traj_dirs, dataset_dir, n_traj, queue: mp.Queue):
#     print蓝("[data_loader_process] started")
#     load = safe_load_traj_pool(traj_dir=traj_dirs, logdir=dataset_dir)
#     while True:
#         while queue.full():
#             # print蓝(f"[data_loader_process] waiting, queue.qsize()={qsz}")
#             time.sleep(1)
#             qsz = queue.qsize()
#         print蓝(f"[data_loader_process] start loading {len(traj_dirs)} trajs_dirs")
#         pool = load(n_samples=n_traj)
#         # datas = []
#         # for traj in pool:
#         #     datas.append(get_data([traj]))
#         # print蓝(f"[data_loader_process] load completed")
#         # queue.put_nowait((datas, traj_dirs,)) 
#         data = get_data(pool)
#         queue.put_nowait((data, traj_dirs,)) 
#         del pool

def loader_thread(traj_dirs, dataset_dir, n_traj, raw_queue, raw_queue_size, final_queue, final_queue_size, load, thread_id):
    while True:
        try:
            while final_queue.qsize() >= final_queue_size:
                time.sleep(0.01)
            # start_time = time.time()
            pool = load(n_samples=n_traj)
            # end_time = time.time()
            # print_green(f"[loader-{thread_id}] load time: {end_time - start_time}")
            raw_queue.put(pool)
            del pool
        except Exception as e:
            print(f"[loader-{thread_id}] Exception: {e}")
            traceback.print_exc()
            time.sleep(2)

def processor_process(traj_dirs, raw_queue, final_queue, process_id):
    while True:
        try:
            pool = raw_queue.get()
            # time_start = time.time()
            data = get_data(pool)
            # time_end = time.time()
            # print_green(f"[processor-{process_id}] get_data time: {time_end - time_start}")
            final_queue.put((data, traj_dirs))
            del pool; del data
        except Exception as e:
            print(f"[processor-{process_id}] Exception: {e}")
            traceback.print_exc()
            time.sleep(2)

def data_loader_process(traj_dirs, dataset_dir, n_traj, final_queue: mp.Queue, final_queue_size):
    # n_loader_threads = 8
    # n_processor_processes = max(n_loader_threads, mp.cpu_count())
    n_processor_processes = mp.cpu_count()//2

    raw_queue_size=final_queue_size

    load = safe_load_traj_pool(
        traj_dir=traj_dirs,
        logdir=dataset_dir,
        use_cache=True,
        preload_cache=False,
        preload_cache_percent=1.0,
    )
    raw_queue = mp.Queue(maxsize=raw_queue_size)

    # for i in range(n_loader_threads):
    #     t = threading.Thread(target=loader_thread, args=(traj_dirs, dataset_dir, n_traj, raw_queue, raw_queue_size, final_queue, final_queue_size, load, i), daemon=True)
    #     t.start()

    for i in range(n_processor_processes):
        p = mp.Process(target=processor_process, args=(traj_dirs, raw_queue, final_queue, i), daemon=True)
        p.start()

    try:
        loader_thread(traj_dirs, dataset_dir, n_traj, raw_queue, raw_queue_size, final_queue, final_queue_size, load, 0)
        while True:
            time.sleep(0.1)
            # print_bold_blue(f"[data_loader_process] raw_queue.qsize()={raw_queue.qsize()}, final_queue.qsize()={final_queue.qsize()}")
    except KeyboardInterrupt:
        print("[data_loader_process] KeyboardInterrupt, exiting...")
    #     load.save_cache()
    # except Exception as e:
    #     load.save_cache()
    #     raise e