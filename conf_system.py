import argparse, os, time, inspect
from shutil import copyfile
from uhtk.UTIL.colorful import *
from uhtk.siri.utils.mx_info import register_mx_info

#CONF_SYSTEM#
def get_conf(CLS, attr: str, default=None):
    assert inspect.isclass(CLS)
    if hasattr(CLS, attr): return getattr(CLS, attr)
    return default

def init_conf_system(prepare_logdir=False, print_summary=False):
    # parse args
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-c', '--cfg', help='Path of the configuration file', required=True)
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0: print亮红('Warning! Arg "%s" is ignored'%str(unknown))

    # override configs (python classes)
    load_and_override_via_json(args.cfg, global_only=False, print_summary=print_summary)

    # prepare logdir
    if prepare_logdir:
        from config import GlobalConfig as cfg
        check_log_path(cfg.logdir)
        if not os.path.exists(cfg.logdir): os.makedirs(cfg.logdir)
        copyfile(args.cfg, '%s/config_backup.jsonc'%cfg.logdir)
        cfg.machine_info = register_mx_info(cfg.logdir)
        cfg.cfg_ready = True

def load_global_conf():
    # parse args
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-c', '--cfg', help='Path of the configuration file', required=True)
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0: print亮红('Warning! Arg "%s" is ignored'%str(unknown))

    # override configs (python classes)
    load_and_override_via_json(args.cfg, global_only=True)

def load_and_override_via_json(json_path, global_only=False, print_summary=False):
    # read jsonc config file
    import commentjson as json
    if print_summary: print('reading configuration at', json_path)
    with open(json_path, encoding='utf8') as f: json_data = json.load(f)

    for cfg_group in json_data:
        if global_only and cfg_group!='config.py->GlobalConfig': continue
        dependency = override_static_class(cfg_group, json_data[cfg_group], print_summary=print_summary)
        if dependency is not None:
            for dep in dependency:
                assert any([dep in k for k in json_data.keys()]), 'Arg check failure, There is something missing!'


def override_static_class(class_spec: str, new_cfg: dict, print_summary=False):
    import importlib
    assert '->' in class_spec, "Class specifier should be some_path.some_py->SomeClass"
    if print_summary:
        str_pro = '------------- %s -------------'%class_spec
        print绿(str_pro)
    file_, class_ = class_spec.split('->')
    if '.py' in file_: 
        # replace it with removesuffix('.py') if you have python>=3.9
        if file_.endswith('.py'): file_ = file_[:-3]    
    default_configs = getattr(importlib.import_module(file_), class_)
    for key in new_cfg:
        safe_setattr(conf_class=default_configs, key=key, new_value=new_cfg[key], print_summary=print_summary)

    if print_summary:
        print绿(''.join(['-']*len(str_pro)),)
        conf_summary(default_configs, new_cfg)
        print绿(''.join(['-']*len(str_pro)),'\n\n\n')

    if 'TEAM_NAMES' in new_cfg:
        return [item.split('->')[0] for item in new_cfg['TEAM_NAMES']]
    return None



def check_log_path(logdir):
    if os.path.exists(logdir):
        print亮红('Current log path:', logdir)
        print亮红('Warning! you will overwrite old logs if continue!')
        print亮红('>>', end='')
        time.sleep(2.0)


def conf_summary(config_class, modify_dict = {}):
    for key in config_class.__dict__: 
        if '__' in key: continue
        if (not key in modify_dict) or (modify_dict[key] is None): 
            print绿(key.center(25), '-->', str(getattr(config_class,key)))
        else: 
            print红(key.center(25), '-->', str(getattr(config_class,key)))

def safe_setattr(conf_class, key, new_value, print_summary=False):
    assert hasattr(conf_class, key), (conf_class, 'has no such config item: **%s**'%key)
    setting_name = key
    replace_item = new_value
    original_item = getattr(conf_class, setting_name)
    if print_summary:
        print绿('[conf_system] override %s:'%setting_name, original_item, '-->', replace_item)
    if isinstance(original_item, float):
        replace_item = float(replace_item)
    elif isinstance(original_item, bool):
        if replace_item == 'True':
            replace_item = True
        elif replace_item == 'False':
            replace_item = False
        elif isinstance(replace_item, bool):
            replace_item = replace_item
        else:
            assert False, ('enter True or False, but have:', replace_item)
    elif isinstance(original_item, int):
        assert int(replace_item) == float(replace_item), ("warning, this var **%s** has an int default, but given a float override!"%key)
        replace_item = int(replace_item)
    elif isinstance(original_item, str):
        replace_item = replace_item
    elif isinstance(original_item, list):
        assert isinstance(replace_item, list)
    elif isinstance(original_item, dict):
        assert isinstance(replace_item, dict)
    elif isinstance(original_item, tuple):
        if isinstance(replace_item, tuple):
            pass
        elif isinstance(replace_item, list):
            replace_item = tuple(replace_item)
        else:
            assert False, ('not support this type', type(replace_item))
    elif replace_item is None:
        pass
    elif original_item is None:
        pass
    else:
        print(key)
        assert False, ('not support this type %s %s'%(str(type(original_item)), str(original_item)))
    setattr(conf_class, setting_name, replace_item)
    return

